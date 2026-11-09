#pragma once

// [QLH-SPLIT] v2.3: VRAM budget pre-check + post-load spill check.
//
// Why this layer exists: the Windows display driver (WDDM) lets an allocation exceed
// dedicated video memory and quietly backs the excess with system RAM. llama.cpp therefore
// still prints "offloaded N/N layers", the process neither errors nor aborts -- it is
// **merely slow**. Measured 2026-11-09 on an 8188 MiB RTX 4060 Laptop: `head -ngl 8` plus a
// fully offloaded tail needs 8.4 GiB against 7.9 GiB free, and **not one line of log said
// what happened**; tg fell from 10.76 to 0.343 t/s (27x slower).
//
// This is the project's recurring "silent wrong" family: no error, a plausible result (it
// runs), just 27x slower. The one job of this layer is to **name the spill**.
//
// Two mechanisms, ported from the v2 implementation (chain_client.py + docs/MEMORY-LIMITS.md
// section 3-4):
//   1. pre-check, before the model is loaded: estimate ``want`` from the bytes this segment
//      will actually touch, compare it with the currently free VRAM, and print
//      ``VMEM: need ~X GiB  free Y GiB  of Z GiB``.
//   2. post-check, after the model is loaded: compare the measured VRAM delta (``landed``)
//      against the weights that were supposed to go there; ``landed < want x slack`` is a
//      spill.
//
// **The post-check never exits the process.** The spill has already happened by then; the
// only thing left to do is tell the operator why every step is slow. v2's conclusion stands:
// "a spill warning does not mean the configuration is unusable -- the criterion is
// throughput, not the warning" (a gpu_layers=6 tier was still spilling and was still the
// fastest of four measured tiers). That is also why the pre-check only warns by default:
// refusing would make "is this spilled tier actually faster?" untestable, and that question
// is exactly what this warning exists to let people answer. QLH_VRAM_STRICT=1 restores
// v2's refuse-to-start behaviour for CI and batch scripts; QLH_VRAM_ALLOW=1 forces a
// start even then.
//
// CPU-only builds (GGML_CUDA=OFF, no GPU device at all) must skip the whole thing: "cannot
// query VRAM" is not a reason to refuse to start, and treating it as one would break every
// CPU deployment.
//
// Single source of truth: this header is included by both entry points that load a segment
// on their own -- tools/split-stage (a stage, links only `llama`) and the llama-server head
// (tools/server). It is header-only on purpose so the stage keeps its "links against llama
// and nothing else" property; it needs only ggml/gguf and the standard library.

#include "llama.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <regex>
#include <string>
#include <vector>

namespace qlh_vram {

// --- constants, verbatim from v2 chain_client.py -------------------------------------------
//
// VRAM_PER_BYTE: measured on an 8188 MiB RTX 4060 Laptop. Two stages that fit both cost
//   exactly 597 MiB on top of their file size (4640 MiB file -> 5237 MiB VRAM delta,
//   4542 -> 5139, both 1.13x), so the cost is linear: file size times a fixed factor plus a
//   fixed per-stage overhead. Both constants are rounded up from those measurements.
// GPU_VRAM_OVERHEAD: the context + compute buffer, measured 597 MiB, rounded up.
// VRAM_LANDED_SLACK: a stage that *fits* occupies MORE video memory than its weights (the
//   compute buffer comes on top); a stage that *spilled* occupies LESS (the driver backed
//   the remainder with system RAM). The driver's own reservations move a few percent either
//   way (measured 5.44 GiB landed for 5.58 GiB of weights), so the comparison allows 15%.
//   A real spill misses by a factor, not by 3%.
inline constexpr double VRAM_PER_BYTE     = 1.13;
inline constexpr size_t GPU_VRAM_OVERHEAD = (size_t) 640 << 20;
inline constexpr double VRAM_LANDED_SLACK = 0.85;

inline constexpr size_t MIB = (size_t) 1 << 20;
inline constexpr size_t GIB = (size_t) 1 << 30;

inline const char * ENV_STRICT  = "QLH_VRAM_STRICT";
inline const char * ENV_ALLOW   = "QLH_VRAM_ALLOW";
// Kill switch. A diagnostic layer that cannot be switched off is a layer that gets blamed for
// every unexplained slowdown, so -- like every other check in this tree -- it can be turned off
// without rebuilding: `QLH_VRAM_DISABLE=1` makes both halves print one line and do nothing.
inline const char * ENV_DISABLE = "QLH_VRAM_DISABLE";

inline bool env_flag(const char * name) {
    const char * v = std::getenv(name);
    if (v == nullptr || *v == '\0' || std::strcmp(v, "0") == 0) {
        return false;
    }
    return std::strcmp(v, "false") != 0 && std::strcmp(v, "no") != 0;
}

inline void fmt_gib(char * buf, size_t n, size_t bytes) {
    std::snprintf(buf, n, "%.2f", (double) bytes / (double) GIB);
}

// --- device memory -------------------------------------------------------------------------

struct device_memory {
    bool        ok    = false;
    size_t      total = 0;
    size_t      free  = 0;
    std::string name;
};

// Does this build have a GPU backend at all?
//
// This is asked *before* anything reads a memory number, and it is what keeps a CPU-only build
// (GGML_CUDA=OFF) working on a machine that happens to have a card: no offload device means
// nothing can be offloaded, so "-ngl all" here is a no-op and there is no budget to check.
// Reading nvidia-smi instead would compute a want for weights that will never leave the CPU and
// then refuse to start a deployment that has no GPU in the first place.
inline bool has_gpu_backend(std::string * name = nullptr) {
    const size_t n = ggml_backend_dev_count();
    for (size_t i = 0; i < n; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (dev == nullptr) {
            continue;
        }
        const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
        if (type == GGML_BACKEND_DEVICE_TYPE_GPU ||
            type == GGML_BACKEND_DEVICE_TYPE_IGPU ||
            type == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            if (name != nullptr) {
                const char * nm = ggml_backend_dev_name(dev);
                *name = nm != nullptr ? nm : "?";
            }
            return true;
        }
    }
    return false;
}

// Free/total video memory as the *driver* sees it -- nvidia-smi, deliberately, and only.
//
// Read this before "simplifying" it to a CUDA API call.
//
// Windows lets an application allocate past dedicated video memory: the display driver (WDDM)
// backs the excess with system RAM, so the allocation *succeeds*, llama.cpp still prints
// "offloaded N/N layers", and the run merely crawls. That is the spill this whole header exists
// to name. It also means every "how much VRAM is left" API answers the wrong question, because
// they report what the driver is still willing to hand out -- over-commit budget included --
// not what dedicated memory is actually still free. Measured here 2026-11-09, RTX 4060 Laptop
// 8188 MiB, tail stage already resident with 5.2 GiB of weights:
//
//   nvidia-smi               memory.used 5440 MiB   memory.free 2517 MiB   <- the truth
//   cudaMemGetInfo           a fresh process reports                     7106 MiB free
//   (llama-server --list-devices, i.e. ggml_backend_dev_memory)
//
// 7106 MiB "free" on a card that has 2517 MiB left. A budget check built on that number
// concluded the very configuration that spilled (8.4 GiB against 7.9 GiB) was safe -- the first
// version of this file did exactly that and D failed silently in a different way.
//
// `ggml_backend_dev_memory` is the same trap (on NVIDIA it *is* cudaMemGetInfo), which is why it
// is not used here at all -- see query_device(). nvidia-smi is the only source that reports
// dedicated usage the driver has already committed, and committed is what decides the next
// spill. v2 reached the same conclusion: "nvidia-smi is the only source that reflects
// *dedicated* usage the driver has already committed" (chain_client.vram_info).
//
// MiB units on purpose: `nounits` + MiB is what nvidia-smi prints and what v2 parsed.
inline device_memory query_nvidia_smi() {
    device_memory m;
#ifdef _WIN32
    FILE * p = _popen("nvidia-smi --query-gpu=memory.total,memory.used "
                      "--format=csv,noheader,nounits 2>nul", "r");
#else
    FILE * p = popen("nvidia-smi --query-gpu=memory.total,memory.used "
                     "--format=csv,noheader,nounits 2>/dev/null", "r");
#endif
    if (p == nullptr) {
        return m;
    }
    char line[256] = {0};
    if (std::fgets(line, sizeof(line), p) != nullptr) {
        long long total_mib = -1;
        long long used_mib  = -1;
        if (std::sscanf(line, "%lld, %lld", &total_mib, &used_mib) == 2 &&
            total_mib > 0 && used_mib >= 0) {
            m.ok    = true;
            m.total = (size_t) total_mib * MIB;
            m.free  = (size_t) std::max<long long>(total_mib - used_mib, 0) * MIB;
        }
    }
#ifdef _WIN32
    _pclose(p);
#else
    pclose(p);
#endif
    return m;
}

// The device with the most free memory, or ok == false when this build cannot offload at all.
//
// Two steps, and the second one is deliberately narrow:
//   1. is there a GPU backend? A CPU-only build has nothing to offload, whatever card the
//      machine happens to carry -- "-ngl all" there is a no-op and there is no budget to check.
//   2. nvidia-smi, and nothing else. NOT `ggml_backend_dev_memory`: under an over-committing
//      driver it answers with the over-commit budget (7106 MiB "free" on a card with 2517 MiB
//      left), and a check built on it declares the spilling configuration safe. The whole
//      point of query_nvidia_smi is that it is the one number that cannot be inflated that way.
//      When nvidia-smi is absent (an AMD/Intel/Vulkan device) this returns ok == false and the
//      caller skips with a stated reason -- "cannot read committed VRAM" is an honest answer;
//      a wrong "it fits" is not.
inline device_memory query_device() {
    device_memory best;
    std::string name;
    if (!has_gpu_backend(&name)) {
        // CPU-only build: nothing to check, and the caller must not treat this as a refusal
        return best;
    }
    best = query_nvidia_smi();
    if (!best.ok) {
        // GPU backend present but no nvidia-smi: skip, do not guess (see above)
        return best;
    }
    best.name = name;
    return best;
}

// --- what a segment will touch -------------------------------------------------------------
//
// Read from the GGUF index rather than assumed from the file size. Two things made the naive
// "file size x layer fraction" wrong:
//   - a range load touches only its own blocks, and the block sizes differ (MoE layers carry
//     ffn_*_exps, the MTP nextn block carries a slightly different set);
//   - `-ncmoe N` keeps the expert tensors of N layers on the host, so a stage can be entirely
//     "on the GPU" and still put only ~9% of its bytes there.
//
// The index walk is cheap (metadata only, no_alloc) and runs once per load.

struct tensor_row {
    std::string name;
    size_t      bytes = 0;
    int         blk   = -1; // -1 = a global tensor (token_embd / output / output_norm)
    bool        exps  = false;
};

struct model_index {
    bool                   ok      = false;
    int                    max_blk = -1;
    std::vector<tensor_row> tensors;
};

inline bool tensor_name_is_exps(const char * name) {
    // v2's rule: "exps" in tensor.name -- it catches all four spellings llama.cpp knows
    // (ffn_{up,down,gate,gate_up}_(ch|)exps) without hard-coding the architecture's names.
    return std::strstr(name, "exps") != nullptr;
}

inline int tensor_name_block(const char * name) {
    // "blk.<N>.<rest>" -- the one part of the naming every architecture shares, which is why
    // this file never needs the architecture name or a block_count KV key.
    if (std::strncmp(name, "blk.", 4) != 0) {
        return -1;
    }
    char * end = nullptr;
    const long v = std::strtol(name + 4, &end, 10);
    if (end == name + 4 || end == nullptr || *end != '.') {
        return -1;
    }
    return (int) v;
}

inline model_index read_model_index(const char * path) {
    model_index idx;
    if (path == nullptr || *path == '\0') {
        return idx;
    }
    gguf_init_params params{};
    params.no_alloc = true;  // metadata only: this runs before the model is loaded
    params.ctx      = nullptr;
    gguf_context * g = gguf_init_from_file(path, params);
    if (g == nullptr) {
        return idx;
    }
    const int64_t n = gguf_get_n_tensors(g);
    idx.tensors.reserve((size_t) std::max<int64_t>(n, 0));
    for (int64_t i = 0; i < n; ++i) {
        const char * nm = gguf_get_tensor_name(g, i);
        if (nm == nullptr) {
            continue;
        }
        tensor_row row;
        row.name  = nm;
        row.bytes = gguf_get_tensor_size(g, i);
        row.blk   = tensor_name_block(nm);
        row.exps  = tensor_name_is_exps(nm);
        if (row.blk > idx.max_blk) {
            idx.max_blk = row.blk;
        }
        idx.tensors.push_back(std::move(row));
    }
    gguf_free(g);
    idx.ok = idx.max_blk >= 0;
    return idx;
}

// The blocks this process loads, in the order llama.cpp assigns its local layer indices.
//
// A range [lo, hi) covers blocks lo..hi-1 of the source. The MTP nextn block is the *largest*
// block id and sits past the trunk, so the range that reaches the top of the model also
// carries it -- observed directly in the stage log:
//     `[C-scheme] layers [32, 40) of 40, block_count -> 9, nextn -> 1`
// i.e. 8 trunk blocks plus the nextn block for `--layer-range 32:40`. Missing it would make
// every tail estimate 1/40 low, and the tail is exactly the segment that spilled.
inline std::vector<int> local_blocks(const model_index & idx, int32_t lo, int32_t hi) {
    std::vector<int> out;
    if (!idx.ok) {
        return out;
    }
    if (lo < 0 || hi < 0) {
        // whole model
        for (const auto & t : idx.tensors) {
            if (t.blk < 0) {
                continue;
            }
            if (out.empty() || out.back() != t.blk) {
                if (std::find(out.begin(), out.end(), t.blk) == out.end()) {
                    out.push_back(t.blk);
                }
            }
        }
        std::sort(out.begin(), out.end());
        return out;
    }
    for (int i = lo; i < hi && i <= idx.max_blk; ++i) {
        out.push_back(i);
    }
    // nextn travels with the range that reaches the top of the source model
    if (hi >= idx.max_blk && idx.max_blk >= lo &&
        (out.empty() || out.back() != idx.max_blk)) {
        out.push_back(idx.max_blk);
    }
    return out;
}

struct estimate_input {
    const char * model_path   = nullptr;
    int32_t      lo           = -1;  // -1/-1 = whole model
    int32_t      hi           = -1;
    int32_t      n_gpu_layers = 0;   // < 0 = every local block (+ the output slot)
    // `-ncmoe` and friends arrive as tensor buffer-type overrides; the ones that send a
    // tensor to the CPU take it off the card. Matching the patterns the same way the loader
    // does is the only way a `-ncmoe 32` head is not charged for experts that never leave RAM.
    const llama_model_tensor_buft_override * overrides = nullptr;
    // `--moe-cache-mib N` puts N MiB of the *host-resident* experts on the card. It is real
    // video memory and it is invisible to a weights-only estimate: a head with 1.19 GiB of
    // weights and a 1024 MiB cache occupies ~2.4 GiB, which is the difference between "fits"
    // and "spills" on a card where the tail already took 5.3 GiB.
    size_t       moe_cache    = 0;
};

struct estimate {
    bool        ok         = false; // false = nothing can be said (no GPU layers, or no index)
    size_t      weights    = 0;     // bytes of weights that go to the card
    size_t      want       = 0;     // weights x VRAM_PER_BYTE + per-segment overhead
    int         n_local    = 0;     // blocks this process loads
    int         n_gpu_blk  = 0;     // of those, how many go to the card
    bool        output_gpu = false;
    std::string why;                // human-readable reason when ok == false
};

// Replicates llama_model::load_tensors()' placement rules (src/llama-model.cpp):
//
//   i_gpu_start   = max((range ? n_layer_all : n_layer_all + 1) - n_gpu_layers, 0)
//   act_gpu_layers= min(n_gpu_layers, n_layer_all + 1)
//   offload_output= range ? (n_gpu_layers >= n_layer_all) : (n_gpu_layers > 0)
//
// and the input layer is always CPU ("there is very little benefit to offloading the input
// layer, so always keep it on the CPU"), which is why token_embd is never charged to VRAM.
inline estimate estimate_want(const estimate_input & in) {
    estimate est;

    if (in.n_gpu_layers == 0) {
        est.why = "no layers offloaded";
        return est;
    }

    const model_index idx = read_model_index(in.model_path);
    if (!idx.ok) {
        est.why = "GGUF index unreadable";
        return est;
    }

    const bool    is_range = in.lo >= 0 && in.hi > in.lo;
    const std::vector<int> blocks = local_blocks(idx, in.lo, in.hi);
    if (blocks.empty()) {
        est.why = "layer range selects no blocks";
        return est;
    }
    const int n_local = (int) blocks.size();
    est.n_local = n_local;

    // llama_model::n_gpu_layers(): a negative request means "this range plus the output slot"
    const int n_gpu_layers = in.n_gpu_layers >= 0 ? in.n_gpu_layers : n_local + 1;
    const int i_gpu_start  = std::max((is_range ? n_local : n_local + 1) - n_gpu_layers, 0);
    const int act_gpu      = std::min(n_gpu_layers, n_local + 1);
    const bool offload_output = is_range ? (n_gpu_layers >= n_local) : (n_gpu_layers > 0);

    est.output_gpu = offload_output;

    // Which local layer indices end up on the card.
    std::vector<bool> on_gpu((size_t) n_local, false);
    int n_gpu_blk = 0;
    for (int il = 0; il < n_local; ++il) {
        const bool gpu = il >= i_gpu_start && il - i_gpu_start < act_gpu;
        on_gpu[(size_t) il] = gpu;
        if (gpu) {
            ++n_gpu_blk;
        }
    }
    est.n_gpu_blk = n_gpu_blk;

    // compile the CPU-side override patterns once
    std::vector<std::regex> cpu_patterns;
    if (in.overrides != nullptr) {
        for (const llama_model_tensor_buft_override * o = in.overrides; o->pattern != nullptr; ++o) {
            if (o->buft == ggml_backend_cpu_buffer_type()) {
                try {
                    cpu_patterns.emplace_back(std::string(o->pattern), std::regex::ECMAScript);
                } catch (const std::regex_error &) {
                    // a pattern we cannot compile is a pattern we cannot second-guess
                }
            }
        }
    }

    auto overridden_to_cpu = [&](const std::string & name) {
        for (const auto & pat : cpu_patterns) {
            if (std::regex_search(name, pat)) {
                return true;
            }
        }
        return false;
    };

    // global tensors that belong to this segment
    const bool has_input  = !is_range || in.lo == 0;          // token_embd: head / whole model
    const bool has_output = !is_range || in.hi >= idx.max_blk; // output: tail / whole model

    size_t weights = 0;
    for (const auto & t : idx.tensors) {
        bool counted = false;
        if (t.blk >= 0) {
            const auto it = std::find(blocks.begin(), blocks.end(), t.blk);
            if (it == blocks.end()) {
                continue;
            }
            const int il = (int) std::distance(blocks.begin(), it);
            counted = on_gpu[(size_t) il];
        } else if (std::strcmp(t.name.c_str(), "output.weight") == 0 ||
                   std::strcmp(t.name.c_str(), "output_norm.weight") == 0) {
            counted = has_output && offload_output;
        } else if (std::strcmp(t.name.c_str(), "token_embd.weight") == 0) {
            // always CPU, see the comment above
            counted = false;
        } else {
            // any other global tensor is small and stays with the loader's own decision
            counted = has_input || has_output;
        }
        if (!counted) {
            continue;
        }
        if (!cpu_patterns.empty() && overridden_to_cpu(t.name)) {
            continue;
        }
        weights += t.bytes;
    }

    est.weights = weights;
    // The expert cache is video memory too. Counted in `want` (what the card must hold) but not
    // in `weights` (what the loader reports as offloaded), because the post-load check compares
    // the driver's delta against the offloaded weights and the cache legitimately sits on top.
    est.want    = (size_t) std::ceil((double) weights * VRAM_PER_BYTE) + GPU_VRAM_OVERHEAD
                + in.moe_cache;
    est.ok      = true;
    return est;
}

// --- pre-check -----------------------------------------------------------------------------

enum verdict {
    VERDICT_OK = 0,          // fits (or nothing to check)
    VERDICT_SPILL_EXPECTED,  // does not fit; warn and continue
    VERDICT_REFUSED,         // does not fit and QLH_VRAM_STRICT asked to refuse
    VERDICT_SKIPPED,         // no GPU device: CPU-only build, say so and continue
};

struct precheck_result {
    verdict       v        = VERDICT_SKIPPED;
    device_memory dev;
    estimate      est;
};

// Prints the budget line and decides. `tag` names the caller ("split-stage", "head").
inline precheck_result precheck(const estimate_input & in, const char * tag) {
    precheck_result r;
    if (env_flag(ENV_DISABLE)) {
        std::fprintf(stderr, "VMEM: %s: check disabled by %s\n", tag, ENV_DISABLE);
        r.v = VERDICT_SKIPPED;
        return r;
    }
    r.dev = query_device();

    if (!r.dev.ok) {
        // Two different "cannot check" cases, and the operator needs to tell them apart:
        // no offload device at all (CPU-only build -- mandatory to continue, or every CPU
        // deployment breaks), versus a GPU backend whose committed usage is unreadable.
        if (has_gpu_backend()) {
            std::fprintf(stderr, "VMEM: %s: GPU backend present but no nvidia-smi, cannot read "
                                 "committed VRAM; skipping budget check\n", tag);
        } else if (in.n_gpu_layers != 0) {
            std::fprintf(stderr, "VMEM: %s: no GPU device, skipping budget check\n", tag);
        }
        r.v = VERDICT_SKIPPED;
        return r;
    }

    r.est = estimate_want(in);
    if (!r.est.ok) {
        if (in.n_gpu_layers != 0) {
            std::fprintf(stderr, "VMEM: %s: cannot estimate offloaded weights (%s), skipping "
                                 "budget check\n", tag, r.est.why.c_str());
        }
        r.v = VERDICT_SKIPPED;
        return r;
    }

    char want_s[32], free_s[32], total_s[32], short_s[32];
    fmt_gib(want_s,  sizeof(want_s),  r.est.want);
    fmt_gib(free_s,  sizeof(free_s),  r.dev.free);
    fmt_gib(total_s, sizeof(total_s), r.dev.total);

    std::fprintf(stderr, "VMEM: %s: need ~%s GiB  free %s GiB  of %s GiB (%s; %d/%d blocks)\n",
                 tag, want_s, free_s, total_s, r.dev.name.c_str(),
                 r.est.n_gpu_blk, r.est.n_local);

    if (r.est.want <= r.dev.free) {
        r.v = VERDICT_OK;
        return r;
    }

    fmt_gib(short_s, sizeof(short_s), r.est.want - r.dev.free);

    // Does not fit. v2 refused here; we name it and continue unless asked to be strict,
    // because a spilled tier is sometimes still the fastest one and refusing destroys the
    // measurement that would show it.
    const bool strict = env_flag(ENV_STRICT);
    const bool allow  = env_flag(ENV_ALLOW);

    if (strict && !allow) {
        std::fprintf(stderr,
            "refusing to start: GPU stages need an estimated %s GiB of video memory but only "
            "%s GiB is free (short by %s GiB).\n"
            "  Lower -ngl, or set %s=1 to start anyway.\n"
            "  Windows does not refuse the allocation -- it spills to system RAM and runs "
            "slower than pure CPU.\n",
            want_s, free_s, short_s, ENV_ALLOW);
        r.v = VERDICT_REFUSED;
        return r;
    }

    std::fprintf(stderr,
        "!! VMEM: expected spill at load: need ~%s GiB but only %s GiB is free (short by %s GiB).\n"
        "   Windows will not refuse the allocation: WDDM backs the excess with system RAM, "
        "llama.cpp will still print \"offloaded N/N\", and decode will be slower than pure CPU.\n"
        "   The post-load check below will confirm it. Set %s=1 to refuse instead.\n",
        want_s, free_s, short_s, ENV_STRICT);
    r.v = VERDICT_SPILL_EXPECTED;
    return r;
}

// --- post-check ----------------------------------------------------------------------------

struct after_result {
    bool   measurable = false;
    size_t landed     = 0;
    size_t want       = 0;
    bool   spill      = false;
};

// Compare the VRAM delta against the weights that were supposed to land there.
//
// `before` must have been sampled before any model was loaded in this process (the delta is
// then attributable to this run rather than to whatever else shares the card). The estimate
// keeps a deliberately generous `want` for the budget line, so only a gross miss is caught
// there; this catches the case the estimate got wrong, using the same weight number.
inline after_result after_load(const estimate & est, const device_memory & before, const char * tag) {
    after_result r;
    if (!est.ok || !before.ok) {
        return r;
    }
    const device_memory now = query_device();
    if (!now.ok) {
        return r;
    }
    r.measurable = true;
    r.want       = est.weights;
    // used-memory delta; a shrink (negative) is clamped to zero rather than wrapped
    const long long delta = (long long) now.free - (long long) before.free;
    r.landed = delta >= 0 ? 0 : (size_t) (-delta);
    r.spill  = est.weights > 0 && r.landed < (size_t) (est.weights * VRAM_LANDED_SLACK);

    char landed_s[32], want_s[32];
    fmt_gib(landed_s, sizeof(landed_s), r.landed);
    fmt_gib(want_s,   sizeof(want_s),   r.want);

    if (!r.spill) {
        std::fprintf(stderr, "VMEM verified: %s: %s GiB resident for %s GiB of GPU weights\n",
                     tag, landed_s, want_s);
        return r;
    }

    std::fprintf(stderr,
        "!! VMEM spill detected after load (%s): %s GiB resident for %s GiB of GPU weights.\n"
        "   %d of %d blocks did not fully fit on the card. Windows spilled the rest to system "
        "RAM; llama.cpp reported them all as offloaded and decode will be slower than pure CPU.\n"
        "   This is a warning, not a failure: a spilled configuration can still be the fastest "
        "one measured -- judge it by throughput, not by this line.\n",
        tag, landed_s, want_s, est.n_gpu_blk, est.n_local);
    return r;
}

} // namespace qlh_vram
