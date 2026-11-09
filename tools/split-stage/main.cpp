// [QLH-SPLIT] split-stage: serve ONE stage of a split chain over TCP.
//
// This process owns a single model range and nothing else. It listens, and for each connection
// serves decode requests until the peer says BYE. Everything it does with llama.cpp is the
// ordinary public API -- the point of this tool is to prove that a stage needs no privileged
// access, only the ability to take hidden rows in and give hidden rows (or logits) out.
//
// Deliberately minimal:
//  - one connection at a time, blocking IO. A chain drives its stages in order, so concurrency
//    would buy nothing and would add a class of bugs this prototype does not need.
//  - no reconnect, no TLS, no heartbeat, no authentication. A dropped connection is a failed
//    chain; the caller surfaces it as a decode error rather than retrying behind the user's
//    back. The lack of authentication is on purpose: --host exists for a stage reached over an
//    internal network or a tunnel, and auth belongs to whatever terminates that tunnel. Do not
//    add a password flag here -- it would be the only credential in the tree and would rot.
//  - the wire format is fixed (see llama-split-proto.h); both ends come from the same build.
#include "llama.h"
// [QLH-SPLIT] v2.3: the VRAM budget check, shared with the llama-server head so both ends of a
// chain judge a segment the same way (see src/qlh-vram-budget.h for why the check exists).
#include "qlh-vram-budget.h"
// [QLH-SPLIT] v2.4: the op_offload default, shared with the llama-server head so a stage and the
// head that drives it cannot disagree about the switch (see the header for the measurement).
#include "qlh-split-defaults.h"
// [QLH-SPLIT] v8: the staging API. Only two declarations are needed here -- the MTP h_nextn
// channel (llama_set_embeddings_nextn / llama_get_embeddings_nextn_ith), which has not moved into
// llama.h -- and they come from the same library this tool already links, exactly as
// tests/dflash-taps.cpp does. Nothing here reaches into context internals.
#include "llama-ext.h"
#include "llama-split-net.h"
#include "llama-split-proto.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iterator>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace qlh_split;

// The socket plumbing is shared with the client (src/llama-split-net.h) instead of living here:
// two copies of send_all/recv_all is how one end ends up handling a short write and the other
// not. The header also brings in winsock2/ws2tcpip or their POSIX equivalents.
using sock_t = qlh_split::net::sock_t;

// [QLH-SPLIT] Per-frame tracing is off by default.
//
// This process serves every decode of a production chain, so a line per frame (and one per
// decode batch) is sustained IO into whatever log file the deployment keeps, and it looks like
// leftover debugging. State-changing operations (seq_rm / state save / state restore) still log
// unconditionally: they are few, they are the audit trail an operator lines up against the
// head's own log, and the acceptance script greps for OP_SEQ_RM to prove the remote KV is
// actually trimmed. Everything else goes through vlog() and needs -v / --verbose.
static bool g_verbose = false;

// [QLH-SPLIT] v5: bounds on the candidate request's variable-length sections. They exist to turn
// a corrupt length into a refused frame instead of a multi-gigabyte allocation inside the stage;
// the real values are tiny (a merged logit-bias list of a few dozen entries, a 64-token window).
static constexpr uint32_t kMaxCandBias = 65536;
static constexpr uint32_t kMaxCandHist = 65536;

static void vlog(const char * fmt, ...) {
    if (!g_verbose) {
        return;
    }
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
}

// [QLH-SPLIT] v7: the token a greedy sampler would pick from one logits row.
//
// Four independent running maxima over the 4-float stride of the row, then one combine. The
// obvious `if (v > best) { best = v; best_i = i; }` loop carries a single dependency chain
// through every one of the 248320 elements; four lanes let the CPU compare four at a time. The
// reply is one int32 per row, so this scan is the whole cost of the answer -- worth writing once,
// carefully, rather than as the branchy version.
//
// Ties keep the *lowest* index, because only a strictly greater value moves a maximum. That is
// exactly what llama_sampler_greedy_apply does (src/llama-sampler.cpp: "if (logit > logit of
// selected)"), so a tail token id and a locally sampled greedy token agree even on an exact tie.
static int32_t argmax_row(const float * v, uint32_t n) {
    GGML_ASSERT(n > 0);

    if (n < 8) {
        int32_t best_i = 0;
        for (uint32_t i = 1; i < n; ++i) {
            if (v[i] > v[best_i]) {
                best_i = (int32_t) i;
            }
        }
        return best_i;
    }

    float    m[4] = { v[0], v[1], v[2], v[3] };
    uint32_t k[4] = { 0, 1, 2, 3 };

    uint32_t i = 4;
    for (; i + 4 <= n; i += 4) {
        const float a = v[i + 0];
        const float b = v[i + 1];
        const float c = v[i + 2];
        const float d = v[i + 3];
        if (a > m[0]) { m[0] = a; k[0] = i + 0; }
        if (b > m[1]) { m[1] = b; k[1] = i + 1; }
        if (c > m[2]) { m[2] = c; k[2] = i + 2; }
        if (d > m[3]) { m[3] = d; k[3] = i + 3; }
    }
    // The remainder goes into lane 0: every index it sees is above every index lane 0 already
    // recorded, so a plain `>` keeps lane 0's "lowest index of its own maximum" property.
    for (; i < n; ++i) {
        if (v[i] > m[0]) { m[0] = v[i]; k[0] = i; }
    }

    // Combine by (value, index) so an exact tie still resolves to the lowest token id.
    int32_t best_i = (int32_t) k[0];
    float   best   = m[0];
    for (int lane = 1; lane < 4; ++lane) {
        if (m[lane] > best || (m[lane] == best && k[lane] < (uint32_t) best_i)) {
            best   = m[lane];
            best_i = (int32_t) k[lane];
        }
    }
    return best_i;
}

// [QLH-SPLIT] v9: diagnosis knobs, read once from the environment so a stage that does not set
// them pays one branch per frame and nothing else.
//
// They are environment variables rather than flags on purpose: they are about *measuring* a stage,
// not about configuring one, and the two questions they answer are both "what did this stage
// actually do" -- which row of a verify frame flipped its argmax, and where the per-frame time
// went. A flag would imply a supported operating mode; these two write a log and change nothing.
//
//   QLH_STAGE_TRACE_ARGMAX=1  one line per answered row: position, argmax, top-1, top-2, gap
//   QLH_STAGE_TIMING=1        a running summary every 16 frames: build/decode/post/reply ms
static bool env_flag(const char * name) {
    const char * v = std::getenv(name);
    return v != nullptr && v[0] != '\0' && std::strcmp(v, "0") != 0;
}

static bool g_trace_argmax = false; // QLH_STAGE_TRACE_ARGMAX
static bool g_timing       = false; // QLH_STAGE_TIMING

static double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

// The top two logits of a row and the token they belong to. A greedy argmax flip between two
// paths can only happen where the two candidates are close, so the gap *is* the diagnosis: 1e-6
// is a different floating-point accumulation order, 0.5 is a bug in the row itself.
struct row_top2 {
    int32_t id   = -1;
    int32_t id2  = -1;
    float   top1 = 0.0f;
    float   top2 = 0.0f;
};

static row_top2 top2_of(const float * v, uint32_t n) {
    row_top2 r;
    for (uint32_t i = 0; i < n; ++i) {
        const float x = v[i];
        if (r.id < 0 || x > r.top1) {
            r.top2 = r.top1;
            r.id2  = r.id;
            r.top1 = x;
            r.id   = (int32_t) i;
        } else if (r.id2 < 0 || x > r.top2) {
            r.top2 = x;
            r.id2  = (int32_t) i;
        }
    }
    return r;
}

// One line per answered row, greppable and stable: `[trace] frame=.. kind=.. pos=.. row=.. id=..
// top1=.. top2=.. gap=.. hm=.. h0=..`. The same format is printed by the plain decode path (one
// row) and by a verify frame (one row per token), so the two paths can be diffed on (pos) alone.
//
// `hm` / `h0` are the L2 norm and the first element of the *input* h row this stage decoded. That
// is the one number that separates "the tail computed the same row and got a different answer"
// (h identical, logits differ -> a compute/kernel problem) from "the head handed over a different
// row" (h differs -> the divergence is upstream of this stage). Comparing logits alone cannot tell
// those apart, and the whole question of where a split chain's batched verify stops agreeing is
// decided by that distinction.
static void trace_row(uint64_t frame, const char * kind, uint32_t n_tok, int32_t pos, uint32_t row,
                      const float * logits, uint32_t n_vocab, int32_t argmax,
                      const float * hrow = nullptr, uint32_t n_embd = 0) {
    if (!g_trace_argmax) {
        return;
    }
    const row_top2 t = top2_of(logits, n_vocab);
    double hm = 0.0;
    float  h0 = 0.0f;
    if (hrow != nullptr && n_embd > 0) {
        h0 = hrow[0];
        double acc = 0.0;
        for (uint32_t i = 0; i < n_embd; ++i) {
            const double x = hrow[i];
            acc += x * x;
        }
        hm = std::sqrt(acc);
    }
    fprintf(stderr, "[trace] frame=%llu kind=%s n_tok=%u pos=%d row=%u argmax=%d top2_argmax=%d "
                    "top1=%.9g top2=%.9g gap=%.9g hm=%.9g h0=%.9g\n",
            (unsigned long long) frame, kind, n_tok, pos, row, (int) argmax, (int) t.id2,
            (double) t.top1, (double) t.top2, (double) (t.top1 - t.top2), hm, (double) h0);
}

// Per-frame phase counters. Reported as means over two kinds of frame -- `plain` (one row, the
// non-speculative decode) and `verify` (drafts + target, answered per row) -- because the whole
// question "what does the per-row channel cost per frame" is a comparison between those two.
struct frame_timer {
    double t0 = 0.0;

    uint64_t n_plain = 0, n_verify = 0, n_draft = 0, n_cand = 0, n_prefill = 0;
    double plain_build = 0, plain_decode = 0, plain_post = 0, plain_reply = 0;
    double verify_build = 0, verify_decode = 0, verify_post = 0, verify_reply = 0;
    double cand_build = 0, cand_decode = 0, cand_post = 0, cand_reply = 0;
    double prefill_decode = 0;
    double verify_argmax = 0, draft_total = 0, draft_decode = 0;
};

static void timing_report(const frame_timer & t) {
    if (!g_timing) {
        return;
    }
    auto mean = [](double sum, uint64_t n) { return n == 0 ? 0.0 : sum / (double) n; };
    // [QLH-SPLIT] v9: `plain` is a 1-row decode answered with the whole logits row, `cand` is the
    // same 1-row decode answered through the candidate channel (what llama-server's sampler asks
    // for by default), `verify` is the k+1-row frame of the per-row channel, `prefill` is any frame
    // wider than MAX_VERIFY_ROWS (excluded from `plain`: averaging a 512-row prompt frame into the
    // per-token numbers is how a "decode costs 40 ms" claim gets invented).
    fprintf(stderr, "[timing] plain n=%llu build=%.2f decode=%.2f post=%.2f reply=%.2f ms | "
                    "cand n=%llu build=%.2f decode=%.2f post=%.2f reply=%.2f ms | "
                    "verify n=%llu build=%.2f decode=%.2f post=%.2f reply=%.2f ms argmax=%.2f ms | "
                    "draft n=%llu round=%.2f (decode %.2f) ms | prefill n=%llu decode=%.2f ms\n",
            (unsigned long long) t.n_plain, mean(t.plain_build, t.n_plain),
            mean(t.plain_decode, t.n_plain), mean(t.plain_post, t.n_plain),
            mean(t.plain_reply, t.n_plain),
            (unsigned long long) t.n_cand, mean(t.cand_build, t.n_cand),
            mean(t.cand_decode, t.n_cand), mean(t.cand_post, t.n_cand),
            mean(t.cand_reply, t.n_cand),
            (unsigned long long) t.n_verify, mean(t.verify_build, t.n_verify),
            mean(t.verify_decode, t.n_verify), mean(t.verify_post, t.n_verify),
            mean(t.verify_reply, t.n_verify), mean(t.verify_argmax, t.n_verify),
            (unsigned long long) t.n_draft, mean(t.draft_total, t.n_draft),
            mean(t.draft_decode, t.n_draft),
            (unsigned long long) t.n_prefill, mean(t.prefill_decode, t.n_prefill));
}

// [QLH-SPLIT] A reply body that fits next to the header in one segment goes out in the SAME
// write as the header; anything bigger keeps the two-write path.
//
// Why this exists: the reply was always "send(header) then send(body)". Over loopback that is
// free; over a real link the candidate answer (32 B header + 320 B body) hits Nagle -- the
// header leaves, the 320 B body is shorter than one MSS so the kernel holds it until the peer's
// delayed ACK lands. Measured cross-host: 48-52 ms per token, which is what made the candidate
// path (A) *slower* than shipping the 993 KB logits row (B) even though A moves 22.4 ms/token
// less data. A small frame therefore must not be split into two writes.
//
// Why the threshold is one MTU and not "always": a body larger than a segment is sent as
// full-MSS segments, which Nagle does not delay, so merging it would buy nothing and would mean
// either a memcpy of up to 993 KB per token or a scatter/gather call the kernel then has to
// copy anyway for no gain. Below the threshold the merge is what keeps header+body in a single
// segment; above it we keep the plain path (and net::set_nodelay in main() covers the general
// case: any write that follows unacknowledged data leaves immediately).
static constexpr size_t kCoalesceMax = 1400; // one Ethernet MTU minus a little, header included

static bool reply(sock_t s, status st, uint32_t n_tokens, uint32_t n_embd, uint32_t n_vocab,
                  const void * data, size_t data_bytes, int32_t pos_max = -1, uint32_t n_bytes = 0,
                  uint8_t flags = 0, uint8_t n_taps = 0) {
    rsp_header h{};
    h.magic    = MAGIC;
    h.version  = PROTO_VER;
    h.status   = (uint8_t) st;
    h.flags    = flags;
    h.n_taps   = n_taps;
    h.n_tokens = n_tokens;
    h.n_embd   = n_embd;
    h.n_vocab  = n_vocab;
    h.pos_max  = pos_max;
    h.n_bytes  = n_bytes;
    if (data_bytes > 0 && data_bytes <= kCoalesceMax) {
        return net::send_two(s, &h, sizeof(h), data, data_bytes);
    }
    if (!net::send_all(s, &h, sizeof(h))) {
        return false;
    }
    if (data_bytes > 0 && !net::send_all(s, data, data_bytes)) {
        return false;
    }
    return true;
}

// [QLH-SPLIT] v8: a reply whose body is up to three sections, sent as one header plus each
// non-empty section.
//
// The two-section cases are the ones v8 introduces: a verify frame answers token ids *and* h_nextn
// rows (id array, then rows), and a plain frame answers the logits row *and* one h row. Neither may
// be collapsed into one buffer: the logits row is ~1 MB, and copying it to glue 16 KB onto it
// would be a memcpy on the per-token path. The coalescing rule of reply() is kept for the small
// case (header + candidates), which is the frame Nagle actually delayed.
static bool reply_parts(sock_t s, status st, uint32_t n_tokens, uint32_t n_embd, uint32_t n_vocab,
                        int32_t pos_max, uint32_t n_bytes, uint8_t flags, uint8_t n_taps,
                        const void * p1, size_t n1,
                        const void * p2, size_t n2,
                        const void * p3, size_t n3) {
    rsp_header h{};
    h.magic    = MAGIC;
    h.version  = PROTO_VER;
    h.status   = (uint8_t) st;
    h.flags    = flags;
    h.n_taps   = n_taps;
    h.n_tokens = n_tokens;
    h.n_embd   = n_embd;
    h.n_vocab  = n_vocab;
    h.pos_max  = pos_max;
    h.n_bytes  = n_bytes;

    const std::pair<const void *, size_t> parts[3] = { { p1, n1 }, { p2, n2 }, { p3, n3 } };

    const size_t total = n1 + n2 + n3;
    if (total <= kCoalesceMax) {
        // Small frame: header and body in one segment (see reply()).
        std::vector<uint8_t> frame(sizeof(h) + total);
        memcpy(frame.data(), &h, sizeof(h));
        size_t off = sizeof(h);
        for (const auto & part : parts) {
            if (part.second > 0) {
                memcpy(frame.data() + off, part.first, part.second);
                off += part.second;
            }
        }
        return net::send_all(s, frame.data(), frame.size());
    }

    if (!net::send_all(s, &h, sizeof(h))) {
        return false;
    }
    for (const auto & part : parts) {
        if (part.second > 0 && !net::send_all(s, part.first, part.second)) {
            return false;
        }
    }
    return true;
}

// [QLH-SPLIT] v5: build the sampler steps a tail runs before it can answer with candidates.
//
// The steps arrive in the order the driving side's chain has them (cand_req::steps), because that
// order is the only thing that makes the candidate set equal to the one the row path would
// produce: penalties before top_k, top_n_sigma before top_k, top_k last. Anything the driving
// side cannot describe this way never reaches here -- it falls back to the whole row there.
//
// They are rebuilt on every frame, from the plain arrays in the frame, instead of being kept
// alive across frames: the penalty state is a window of recent tokens (cand_req::n_hist), so
// rebuilding it makes the answer a pure function of the frame. A tail that was reconnected, or a
// sequence whose KV was trimmed, cannot then hold a stale history that quietly changes the
// penalty of the next token.
//
// Returns nullptr when the frame describes a step this build does not know. That is a protocol
// error, not a fallback: the two ends are built from the same tree (PROTO_VER), so it means a
// corrupt or forged frame, and guessing a chain from it would be worse than refusing it.
static llama_sampler * build_cand_chain(const cand_req & c,
                                        const std::vector<llama_logit_bias> & bias,
                                        const std::vector<llama_token> & hist,
                                        uint32_t n_vocab) {
    llama_sampler_chain_params cp = llama_sampler_chain_default_params();
    cp.no_perf = true; // the tail measures the decode, not the chain
    llama_sampler * chain = llama_sampler_chain_init(cp);
    if (chain == nullptr) {
        return nullptr;
    }

    // logit_bias is always first on the driving side (common_sampler_init), so it is always first
    // here; it is not in steps[] because it has no parameters that vary per frame beyond its own
    // list. Suppression (-INFINITY biases) belongs to this step: a suppressed token must be gone
    // before top_k ranks, or it takes a candidate slot the row path would have given to a token
    // that can actually be sampled.
    if (!bias.empty()) {
        llama_sampler_chain_add(chain,
                llama_sampler_init_logit_bias((int32_t) n_vocab, (int32_t) bias.size(), bias.data()));
    }

    for (uint32_t i = 0; i < c.n_steps; ++i) {
        switch (c.steps[i]) {
            case CAND_STEP_PENALTIES: {
                llama_sampler * p = llama_sampler_init_penalties((int32_t) n_vocab,
                        c.penalty_last_n, c.penalty_repeat, c.penalty_freq, c.penalty_present);
                // Replay the window oldest-first so the internal ring buffer ends up holding the
                // same tokens, in the same order, as the driving side's own penalties sampler.
                for (const llama_token t : hist) {
                    llama_sampler_accept(p, t);
                }
                llama_sampler_chain_add(chain, p);
                break;
            }
            case CAND_STEP_TOP_N_SIGMA:
                llama_sampler_chain_add(chain, llama_sampler_init_top_n_sigma(c.top_n_sigma));
                break;
            case CAND_STEP_TOP_K:
                llama_sampler_chain_add(chain, llama_sampler_init_top_k(c.top_k));
                break;
            default:
                llama_sampler_free(chain);
                return nullptr;
        }
    }

    return chain;
}

// [QLH-SPLIT] v6: the stage's own checkpoint store. See --state-cache and OP_STATE_SAVE.
//
// A context checkpoint of a split chain has to cover every stage, and this stage's part is its
// recurrent state -- 6,733,996 B for an 8-layer qwen35 tail, 13,172,908 B for the same 8 layers
// of the 35B qwen35moe, both independent of the prompt length (the checkpoint flavour writes no
// KV). Before v6 that blob went down the wire on every checkpoint and back up on every restore:
// 0.8-1.2 s per checkpoint on a 16 MB/s link, 2.8 s at the 5 MB/s of the field report -- more
// than the decode the checkpoint exists to protect. The head never reads those bytes, so it only
// needs a name for them, and the state is already here.
//
// The cache is bounded, and that is the whole trade-off: an evicted entry cannot be restored, and
// the stage answers ST_STALE instead of failing. The head treats that as "no checkpoint" and
// re-processes the prompt -- a prefill, never a wrong token. --state-cache sizes it; the default
// (32) matches llama-server's own per-slot checkpoint bound, so the common case never evicts.
struct stage_snapshot {
    int32_t  pos_max = -1;
    std::vector<uint8_t> blob;
};

// =====================================================================================
// [QLH-SPLIT] The command line: what the operator typed, and why each value is checked.
// =====================================================================================

struct stage_options {
    std::string model_path;
    std::string role;
    std::string host     = "127.0.0.1";
    int32_t  layer_lo    = -1;
    int32_t  layer_hi    = -1;
    int32_t  port        = 9330;
    uint32_t rs_seq      = 0;
    bool     rs_seq_set  = false;
    int32_t  n_ctx       = 512;
    int32_t  n_batch     = 256;
    uint32_t state_cache = 32; // v6: how many of the head's checkpoints this stage keeps

    // [QLH-SPLIT] v9: thread counts. -1 means "not given", and not given is the *default* this
    // tool always had: llama_context_default_params() leaves both at GGML_DEFAULT_N_THREADS (4),
    // so a stage spawned no arguments about threads keeps behaving exactly like v8 did. A stage is
    // the only place that can fix this -- the head's -t does not reach across the wire -- and on a
    // 4C8T laptop the difference between 4 threads and all 8 is a whole core's worth of throughput
    // per token.
    //
    // Two knobs, not one, because the two halves of a decode want different things: prefill and a
    // multi-row verify frame go through n_threads_batch (sized by rows, one row per thread is
    // wasted), a single-token decode goes through n_threads. -tb exists so the two can be measured
    // against each other; the default keeps llama.cpp's own rule ("n_threads_batch follows
    // n_threads unless set"), which is what llama-server does for -t.
    int32_t  n_threads       = -1; // -t  / --threads
    int32_t  n_threads_batch = -1; // -tb / --threads-batch
    // [QLH-SPLIT] v9: -ub, the physical batch. Its reason to exist here is a measurement, not a
    // tuning knob: llama_decode() splits a batch into ubatches of n_ubatch rows, so -ub 1 makes a
    // verify frame run one row at a time inside one frame -- the shape a plain decode has. That is
    // the only way to ask "does the *tail*'s batched compute account for the argmax flips, or does
    // the head's?" without writing a second decoding loop.
    // [QLH-SPLIT] v2.2: -ngl, the whole point of a stage being a *range*: a segment can fit in
    // VRAM even when the model cannot (tail 32:40 is ~4.2 GiB against 21.11 GiB, on an 8 GiB
    // RTX 4060). The default is 0 and NOT llama.cpp's -1 ("all layers"): -1 is right for a tool
    // that loads a model the user asked to run, and wrong for a stage, where the head is 16.9 GiB
    // and silently trying to put it on an 8 GiB card is an OOM, not a request. A stage offloads
    // only when the operator says so.
    int32_t  n_gpu_layers = 0;   // -ngl N | all (-1)
    // [QLH-SPLIT] v2.2: --no-host, the control for the host-buffer question. CUDA builds a pinned
    // host buffer type too, and make_cpu_buft_list puts it ahead of the plain CPU one for every
    // device, so "GPU backend loaded, -ngl 0" is measurable as a second CPU baseline. See
    // docs/IGPU-SLOWDOWN.md §1.1 for the Vulkan measurement this exists to repeat on CUDA.
    bool     no_host      = false;
    // [QLH-SPLIT] v2.2: op_offload is the switch that decides whether the GPU may take over the
    // *host* tensors' big operators. It is ON upstream, and on a range load it is the difference
    // between a positive and a negative head stage: with it on, the CPU-resident blocks' weights
    // go into the pinned CUDA host buffer and the GPU runs their prefill matmuls, which costs
    // them ~31% of decode speed (measured: -ngl 0 tg 11.25 vs 18.23 with it off, same build).
    // A stage already decides per block where it runs, so it wants the GPU for its offloaded
    // blocks and the CPU's own kernels for the rest -- default off, --op-offload restores upstream.
    bool     no_op_offload = QLH_SPLIT_DEFAULT_NO_OP_OFFLOAD;
    // [QLH-SPLIT] v2.2: the MoE expert cache and the weight-repack switch, which only matter once
    // a *part* of this segment is on the GPU: the experts of the CPU-resident blocks stay in host
    // memory, and llama_moe_cache uploads the hot ones into a device budget (upstream e95426f1).
    // On a fully offloaded stage both are no-ops by construction (there are no host experts left
    // to cache), so the defaults change nothing for the CPU stages this tool has always served.
    int32_t  moe_cache_mib = 0;      // --moe-cache-mib N (0 = disabled)
    float    moe_cache_cpu_share = 0.0f; // --moe-cache-cpu-share F (0 = upload every miss)
    bool     no_repack = false;      // -nr / --no-repack
    int32_t  n_ubatch        = 0;  // 0 = follow -b, as before
};


// [QLH-SPLIT] v9: a thread count, parsed so that a non-number is a sentence and not an exception.
static int32_t parse_int(const std::string & v, const char * flag) {
    char * end = nullptr;
    const long x = std::strtol(v.c_str(), &end, 10);
    if (end == v.c_str() || (end != nullptr && *end != '\0')) {
        fprintf(stderr, "%s expects an integer (got '%s')\n", flag, v.c_str());
        exit(1);
    }
    return (int32_t) x;
}

static int parse_args(int argc, char ** argv, stage_options & o) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char * name) -> std::string {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for %s\n", name);
                exit(1);
            }
            return argv[++i];
        };
        if      (a == "-m" || a == "--model")           { o.model_path = next("-m"); }
        else if (a == "--role")                         { o.role    = next("--role"); }
        else if (a == "--host")                         { o.host    = next("--host"); }
        else if (a == "--layer-range") {
            // A half-parsed range used to start a stage on the default -1:-1 (or on a lo that
            // sscanf wrote before it stopped), which is a stage serving a range nobody asked
            // for. Refuse to start instead of running on a value the operator did not type.
            const std::string v = next("--layer-range");
            int32_t lo = -1;
            int32_t hi = -1;
            if (sscanf(v.c_str(), "%d:%d", &lo, &hi) != 2) {
                fprintf(stderr, "--layer-range expects lo:hi, two integers separated by ':' (got '%s')\n",
                        v.c_str());
                return 1;
            }
            o.layer_lo = lo;
            o.layer_hi = hi;
        }
        else if (a == "--port")                         { o.port    = std::stoi(next("--port")); }
        else if (a == "--rs-seq")                       { o.rs_seq  = (uint32_t) std::stoul(next("--rs-seq")); o.rs_seq_set = true; }
        else if (a == "-v" || a == "--verbose")         { g_verbose = true; }
        else if (a == "-c" || a == "--ctx-size")        { o.n_ctx   = std::stoi(next("-c")); }
        else if (a == "-b" || a == "--batch-size")      { o.n_batch = std::stoi(next("-b")); }
        else if (a == "-ub" || a == "--ubatch-size")    { o.n_ubatch = std::stoi(next("-ub")); }
        // [QLH-SPLIT] v9: thread count. Parse into int32 and check the value in validate_options,
        // where the refusal is a sentence instead of a std::stoi exception: `-t abc` throwing out
        // of main() prints "terminate called after throwing" and nothing about threads, and a
        // stage that exits for the wrong stated reason is how the next person loses an hour.
        else if (a == "-t" || a == "--threads")         { o.n_threads = parse_int(next("-t"), "-t"); }
        else if (a == "-tb" || a == "--threads-batch")  { o.n_threads_batch = parse_int(next("-tb"), "-tb"); }
        // [QLH-SPLIT] v2.2: -ngl, the same spelling common/arg.cpp uses. "all" maps to -1, which
        // llama_model::n_gpu_layers() expands to "every block of this range + the output slot";
        // an explicit N counts the last N blocks of *this range* (see load_tensors in
        // src/llama-model.cpp for why that base is the range's high bound and not the whole model).
        else if (a == "-ngl" || a == "--gpu-layers" || a == "--n-gpu-layers") {
            const std::string v = next("-ngl");
            o.n_gpu_layers = (v == "all") ? -1 : parse_int(v, "-ngl");
        }
        else if (a == "--no-host")                      { o.no_host = true; }
        else if (a == "--moe-cache-mib")                { o.moe_cache_mib = parse_int(next("--moe-cache-mib"), "--moe-cache-mib"); }
        else if (a == "--moe-cache-cpu-share")          { o.moe_cache_cpu_share = (float) std::atof(next("--moe-cache-cpu-share").c_str()); }
        else if (a == "-nr" || a == "--no-repack")      { o.no_repack = true; }
        else if (a == "--no-op-offload")                { o.no_op_offload = true; }
        else if (a == "--op-offload")                   { o.no_op_offload = false; }
        else if (a == "--state-cache")                  { o.state_cache = (uint32_t) std::stoul(next("--state-cache")); }
        else {
            fprintf(stderr, "unknown argument: %s\n", a.c_str());
            return 1;
        }
    }
    return 0;
}

// The checks that need the whole command line, done before anything is loaded.
static int validate_options(const stage_options & o) {
    // [QLH-SPLIT] --role and --rs-seq are required, not inferred or defaulted.
    //
    // The role used to be derived from llama_model_n_layer(), but that returns the *local*
    // depth (hi - lo) of a range load, so the test "layer_hi >= n_layer_local" is true for
    // every range with lo >= 0: a relay stage silently ran as the tail (no hidden exported,
    // logits returned) and the client kept its carrier unchanged. Nothing errored; the
    // answer was just wrong. Only the caller knows which end of the chain it is serving.
    //
    // rs_seq > 0 is required *here* even though 0 is a supported value for a local chain: a
    // local chain falls back to a context checkpoint (llama-server restores one, and a chain's
    // checkpoint covers every stage), while a remote stage process has nothing to fall back on
    // -- the truncating seq_rm from a slot reuse arrives over the wire, and with 0 snapshots
    // this process can only answer "cannot roll back", which common.cpp turns into GGML_ABORT.
    // That dependency is real only because the driving side now broadcasts seq_rm over the wire
    // (llama_context installs a llama_memory_split that holds the remote stages); before that,
    // this comment claimed the same thing while no seq_rm ever reached a stage, so it was true
    // in intent and false in fact. Do not infer the value from llama_model_is_recurrent():
    // qwen35 is a KV+recurrent hybrid and that predicate is false for it.
    if (o.model_path.empty() || !o.rs_seq_set) {
        fprintf(stderr, "usage: split-stage -m MODEL --layer-range lo:hi --role relay|tail --rs-seq N "
                        "[--host ADDR] [--port N] [-c N] [-b N] [-ub N] [-t N] [-tb N] "
                        "[-ngl N|all] [--op-offload] [--no-host] [--moe-cache-mib N] [--moe-cache-cpu-share F] [-nr] "
                        "[--state-cache N] [-v]\n");
        return 1;
    }
    // [QLH-SPLIT] v9: the thread counts, checked here so a typo is refused with a number in the
    // message instead of a thread pool of that size. Two bounds, and both are real:
    //  - n < 1: ggml does not define what a pool of zero threads means (the barrier waits for
    //    nobody), so it is never a configuration, it is a missing value.
    //  - n > 4 * hardware_concurrency (and never below 64, so a container that reports 1 CPU
    //    still lets an operator ask for a reasonable number): a stage that wants more than four
    //    times the cores is a typo -- `-t 999` on a 4C8T laptop is the example that started this.
    // The upper bound is a sanity rail, not a promise: oversubscribing at all is usually slower,
    // and the honest way to find the plateau is to measure the values that pass.
    {
        const unsigned hw  = std::thread::hardware_concurrency();
        const int32_t  max = (int32_t) std::max(64u, hw == 0 ? 64u : hw * 4u);
        for (const auto & pair : { std::make_pair("-t", o.n_threads),
                                   std::make_pair("-tb", o.n_threads_batch) }) {
            if (pair.second == -1) {
                continue;
            }
            if (pair.second < 1) {
                fprintf(stderr, "%s must be at least 1 (got %d); omit it to keep llama.cpp's "
                                "default of %d threads\n", pair.first, pair.second,
                        GGML_DEFAULT_N_THREADS);
                return 1;
            }
            if (pair.second > max) {
                fprintf(stderr, "%s %d is out of range: this machine reports %u hardware threads, "
                                "so the accepted range is 1..%d (omit the flag to keep the default)\n",
                        pair.first, pair.second, hw, max);
                return 1;
            }
        }
    }
    // -ub is either absent or a real physical batch. It is bounded by the same rule llama_context
    // applies (n_ubatch <= n_batch); a value larger than -b would be silently clamped there, so it
    // is refused here where the operator can see which of the two numbers to change.
    if (o.n_ubatch < 0 || (o.n_ubatch > 0 && o.n_ubatch > o.n_batch)) {
        fprintf(stderr, "-ub must be in 1..%d (the -b value) or 0 to follow -b (got %d)\n",
                o.n_batch, o.n_ubatch);
        return 1;
    }
    // [QLH-SPLIT] v2.2: -ngl is either "all" (-1, expanded by llama_model_n_gpu_layers() to this
    // range's block count + output slot) or a count of this range's last blocks. A negative value
    // other than "all" is not a smaller number, it is a typo: llama.cpp would read every negative
    // as "all" and the stage would try to put a 16.9 GiB head on an 8 GiB card.
    if (o.n_gpu_layers < -1) {
        fprintf(stderr, "-ngl expects N >= 0 or 'all' (got %d)\n", o.n_gpu_layers);
        return 1;
    }
    if (o.moe_cache_mib < 0) {
        fprintf(stderr, "--moe-cache-mib must be >= 0 (got %d)\n", o.moe_cache_mib);
        return 1;
    }
    if (o.moe_cache_cpu_share < 0.0f || o.moe_cache_cpu_share > 1.0f) {
        fprintf(stderr, "--moe-cache-cpu-share must be in [0, 1] (got %.3f)\n", (double) o.moe_cache_cpu_share);
        return 1;
    }
    // Say what is wrong with the range, not the whole usage line: this is the one argument an
    // operator gets wrong by hand, and "usage:" does not tell them which half.
    if (o.layer_lo < 0 && o.layer_hi < 0) {
        fprintf(stderr, "--layer-range is required, as lo:hi (e.g. 8:16)\n");
        return 1;
    }
    if (o.layer_hi <= o.layer_lo) {
        fprintf(stderr, "--layer-range must be lo:hi with lo < hi (got %d:%d)\n",
                o.layer_lo, o.layer_hi);
        return 1;
    }
    if (o.role != "relay" && o.role != "tail") {
        fprintf(stderr, "--role must be 'relay' or 'tail' (got '%s')\n", o.role.empty() ? "" : o.role.c_str());
        return 1;
    }
    // [QLH-SPLIT] v6: a stage keeps the checkpoints it is asked for, so the bound is a memory
    // bound, not a correctness one -- but it must not be 0. Every handle would answer ST_STALE,
    // the head would drop each checkpoint it just made, and every request would re-process its
    // prompt: a chain that is correct and useless. Refuse to start instead.
    if (o.state_cache == 0) {
        fprintf(stderr, "--state-cache must be at least 1 (a checkpoint names a snapshot this "
                        "process has to keep); 0 would make every restore STALE\n");
        return 1;
    }
    if (o.layer_lo <= 0) {
        // Layers before lo belong to the head, and the head runs in the driving process: it is
        // the process that has the tokens. A split-stage only ever consumes hidden rows.
        fprintf(stderr, "layer-lo must be > 0 (got %d): the head owns the front of the model and "
                        "runs in the driving process, a split-stage only consumes hidden rows\n",
                o.layer_lo);
        return 1;
    }
    return 0;
}

// =====================================================================================
// [QLH-SPLIT] The running stage: one loaded range, its context, the numbers the wire and the
// identity handshake need, and the scratch the decode loop reuses across frames.
// =====================================================================================

struct stage {
    stage_options opt;

    llama_model *        model = nullptr;
    llama_context *      ctx   = nullptr;
    // [QLH-SPLIT] v8: the MTP draft context, built over this same model when the range owns the
    // NextN block. It is a *second* context with its own KV (a handful of cells: it is cleared
    // every draft round and never sees the prompt), and it is what OP_MTP_DRAFT drives. Null means
    // "this stage cannot draft" -- no NextN block in the model, this range is not the tail, or the
    // context failed to build -- and that is reported in the identity handshake, never aborting a
    // process that can still serve decodes.
    llama_context *      ctx_mtp = nullptr;
    const llama_vocab *  vocab = nullptr;
    llama_context_params cparams{};

    int32_t  n_layer_src    = 0;
    uint32_t n_embd         = 0;
    uint32_t n_vocab        = 0;
    uint32_t n_pos_per_embd = 1;
    bool     relay          = false;
    bool     has_nextn      = false;
    stage_id self{};

    std::vector<float>            hidden;
    std::vector<llama_pos>        positions;
    // [QLH-SPLIT] v5 candidate scratch, allocated once. sized to the vocabulary (a few MB at
    // most) and reused: a fresh 248k-entry vector per token would put an allocation and a
    // zero-fill on the decode path for no reason.
    std::vector<llama_token_data> cand_cur;
    std::vector<token_cand>       cand_out;
    // [QLH-SPLIT] v7: one token id per row of a verify frame. Reused like the candidate scratch,
    // and sized per frame (at most MAX_VERIFY_ROWS entries, 68 B).
    std::vector<int32_t>          verify_out;
    // One-shot diagnostic: the first decode of this process prints the position the model will
    // actually consume (see the comment in OP_DECODE).
    bool logged_pos = false;
    // [QLH-SPLIT] v7: and the first per-row verify frame prints its shape, so an operator can
    // tell "the channel is in use" from "the channel is off" without a packet capture.
    bool logged_verify = false;
    // [QLH-SPLIT] v8: and the first MTP draft frame prints how many drafts crossed the wire, so
    // "MTP is drafting" is visible in the stage's own log without a packet capture.
    bool logged_mtp = false;
    // [QLH-SPLIT] v8: the k draft ids OP_MTP_DRAFT answers with, and the single h row it decodes
    // per step. Both are reused across frames, like the scratches above.
    std::vector<int32_t>          mtp_out;
    std::vector<float>            mtp_h;

    // [QLH-SPLIT] v8: one h_nextn row per output row of the last decode, when the frame asked for
    // them (DECODE_FLAG_NEXTN_ROWS). Reused across frames; a prefill frame fills exactly one row.
    std::vector<float>            nextn_out;
    // One-shot diagnostic for a head that asked for rows this stage cannot produce (see the note
    // in decode_read_frame): the answer omits them, and this says so once.
    bool logged_no_nextn = false;

    // [QLH-SPLIT] v9: the layer-input tap channel. `tap_on` is one flag per local layer saying
    // whether extraction is already enabled for it: set_embeddings_layer_inp() marks the scheduler
    // for a re-reserve, so calling it once per frame would re-reserve the graph once per frame. The
    // set only ever grows (the head asks for the same layers on every decode), so the reserve is
    // paid once, on the first tap frame.
    //
    // `tap_ask` is the frame's request scratch, `tap_ids_out`/`tap_rows_out` the answer scratch, and
    // both are reused across frames like the other scratches above.
    std::vector<char>             tap_on;
    std::vector<int32_t>          tap_ask;
    std::vector<int32_t>          tap_ids_out;
    std::vector<float>            tap_rows_out;
    // One-shot diagnostic: the first frame that asked for taps prints how many this stage owns, so
    // "the channel is in use" and "the head's tap layers are all below this range" look different
    // without a packet capture.
    bool logged_tap = false;

    // [QLH-SPLIT] v9: diagnosis only, both inert unless QLH_STAGE_TRACE_ARGMAX / QLH_STAGE_TIMING
    // are set. `frame_id` numbers every OP_DECODE frame so trace lines and timing lines can be
    // lined up against the head's own log without timestamps.
    uint64_t    frame_id = 0;
    frame_timer timing;

    // [QLH-SPLIT] v6: the snapshots this stage has been asked to keep, oldest first (that is also
    // the eviction order).
    //
    // Handles are *random* 64-bit names, not counters. A counter would restart at 1 in a new
    // process, so a handle the head still holds from before a stage restart would name some
    // unrelated snapshot minted after it -- silently restoring the wrong state, which is the one
    // failure mode this whole layer exists to prevent. A random name makes "not the process that
    // minted it" indistinguishable from "evicted", i.e. both answer ST_STALE, which is exactly the
    // answer the head knows how to recover from. The collision check below is for the free
    // in-process case (a duplicate would make the lookup return the older entry).
    std::deque<std::pair<uint64_t, stage_snapshot>> snapshots;
    std::mt19937_64 snap_rng;
};

static void stage_free(stage & s) {
    // [QLH-SPLIT] v8: the MTP context borrows the model, so it goes first -- and it does not own
    // the model, exactly like the main context.
    if (s.ctx_mtp != nullptr) {
        llama_free(s.ctx_mtp);
        s.ctx_mtp = nullptr;
    }
    if (s.ctx != nullptr) {
        llama_free(s.ctx);
        s.ctx = nullptr;
    }
    if (s.model != nullptr) {
        llama_model_free(s.model);
        s.model = nullptr;
    }
    llama_backend_free();
}

// Load the range, build the context, and prove the arguments describe something this process can
// actually serve. Returns false with the reason already printed; the caller frees and exits 1.
static bool stage_init(const stage_options & opt, stage & s) {
    s.opt = opt;

    llama_backend_init();

    auto mparams = llama_model_default_params();
    mparams.split_layer_lo = opt.layer_lo;
    mparams.split_layer_hi = opt.layer_hi;
    // [QLH-SPLIT] v2.2: the offload knobs. -1 ("all") is passed through: llama_model::n_gpu_layers()
    // turns it into the local block count + 1, so "all" means the whole range without this tool
    // having to know the range's depth before it loads the GGUF.
    mparams.n_gpu_layers = opt.n_gpu_layers;
    mparams.no_host      = opt.no_host;
    mparams.use_extra_bufts = !opt.no_repack;

    // [QLH-SPLIT] v2.3: VRAM budget pre-check, before a single weight is read.
    //
    // Here rather than after the load because the number that decides it -- how much VRAM is
    // free -- is only meaningful while this stage has not taken any yet, and because a stage
    // started second in a chain sees the first one's card already occupied: that is exactly the
    // configuration that spilled silently on 2026-11-09 (`head -ngl 8` + fully offloaded tail,
    // 8.4 GiB needed against 7.9 GiB free, tg 10.76 -> 0.343).
    //
    // WDDM does not fail an overshoot, it pages, and llama.cpp still prints "offloaded N/N" --
    // so without this line the only symptom is a 27x slowdown. A segment with -ngl 0 has
    // nothing to check and says nothing.
    qlh_vram::estimate_input vram_in;
    vram_in.model_path   = opt.model_path.c_str();
    vram_in.lo           = opt.layer_lo;
    vram_in.hi           = opt.layer_hi;
    vram_in.n_gpu_layers = opt.n_gpu_layers;
    vram_in.overrides    = mparams.tensor_buft_overrides;
    vram_in.moe_cache    = (size_t) opt.moe_cache_mib * 1024 * 1024;
    const qlh_vram::precheck_result vram_pre = qlh_vram::precheck(vram_in, "split-stage");
    if (vram_pre.v == qlh_vram::VERDICT_REFUSED) {
        return false;
    }

    s.model = llama_model_load_from_file(opt.model_path.c_str(), mparams);
    if (s.model == nullptr) {
        fprintf(stderr, "failed to load model\n");
        return false;
    }

    // [QLH-SPLIT] Validate the declared role against the source model the loader saw, not
    // against this stage's local layer count (see the note at the argument check).
    s.n_layer_src = llama_model_n_layer_src(s.model);
    if (s.n_layer_src <= 0) {
        fprintf(stderr, "model was not loaded as a layer range; cannot determine the source layer count\n");
        return false;
    }
    s.relay = (opt.role == "relay");

    // [QLH-SPLIT] The compat conversion of an embedding batch reads n_pos_per_embd positions
    // per token, section-major (src/llama-batch.cpp, llama_batch_compat::init). There is no
    // public getter for it, so mirror llama_hparams::n_pos_per_embd() from the rope type.
    const llama_rope_type rope_type = llama_model_rope_type(s.model);
    s.n_pos_per_embd = (rope_type == LLAMA_ROPE_TYPE_MROPE ||
                        rope_type == LLAMA_ROPE_TYPE_IMROPE)
                     ? GGML_MROPE_SECTIONS : 1;

    if (s.relay) {
        if (opt.layer_hi >= s.n_layer_src) {
            fprintf(stderr, "role=relay requires layer-hi < %d (the source model's trunk depth); "
                            "layers [%d, %d) reach the model's last layer and must be served as --role tail\n",
                    s.n_layer_src, opt.layer_lo, opt.layer_hi);
            return false;
        }
    } else if (opt.layer_hi != s.n_layer_src) {
        fprintf(stderr, "role=tail requires layer-hi == %d (the source model's trunk depth), got %d; "
                        "a range that stops earlier does not produce logits\n",
                s.n_layer_src, opt.layer_hi);
        return false;
    }

    s.cparams = llama_context_default_params();
    s.cparams.n_ctx     = (uint32_t) opt.n_ctx;
    s.cparams.n_batch   = (uint32_t) opt.n_batch;
    s.cparams.n_ubatch  = (uint32_t) (opt.n_ubatch > 0 ? opt.n_ubatch : opt.n_batch);
    s.cparams.n_rs_seq  = opt.rs_seq;
    // [QLH-SPLIT] v2.2: MoE 专家缓存（upstream e95426f1）。只有「本段有一部分留在 CPU」时才
    // 有意义：留在 host 的专家权重由缓存放进设备预算，热专家上传、冷专家按 cpu_share 在 CPU 算。
    s.cparams.moe_cache_size      = (size_t) opt.moe_cache_mib * 1024 * 1024;
    s.cparams.moe_cache_cpu_share = opt.moe_cache_cpu_share;
    s.cparams.op_offload          = !opt.no_op_offload;

    // [QLH-SPLIT] v9: the thread counts, applied before the context exists (they are constructor
    // fields: n_threads_batch also decides how the threadpool is created, so setting them later
    // through llama_set_n_threads would size the pool from the default).
    //
    // Nothing given => nothing changed: both stay at llama_context_default_params(), which is
    // GGML_DEFAULT_N_THREADS (4) and not, as the older notes in this tree assumed, the number of
    // physical cores. That default is what a stage ran with until now, so "no -t" is bit-for-bit
    // the old behaviour and the five acceptance scripts keep meaning what they meant.
    //
    // -t alone sets both, mirroring llama-server (-t sets cpuparams, and cpuparams_batch defaults
    // to "follow"). -tb then overrides the batch half so the two can be measured apart: a plain
    // decode of one row uses n_threads, while prefill and a multi-row verify frame use
    // n_threads_batch, and those are the two shapes this stage serves.
    if (opt.n_threads > 0) {
        s.cparams.n_threads = opt.n_threads;
        s.cparams.n_threads_batch = opt.n_threads;
    }
    if (opt.n_threads_batch > 0) {
        s.cparams.n_threads_batch = opt.n_threads_batch;
    }

    // [QLH-SPLIT] The output-row cap is a property of the role, not a tuning knob, and it is
    // what bounds the logits tensor the *reserve* pass allocates: sched_reserve reserves the pp
    // graph with n_outputs = min(n_tokens, cparams.n_outputs_max), so the default (0, which the
    // context constructor reads as n_batch) makes that node [n_vocab, n_batch] f32. At -b 2048
    // with this model's 248k vocab that is ~1.9 GiB of compute buffer, held for the life of the
    // process, for a stage whose reply() carries exactly one row of logits per frame
    // ("reply(peer, ST_OK, 1, 0, n_vocab, ...)"). The batch already marks one token as output
    // (the one the caller samples from), so 1 is the real worst case.
    //
    //  - relay: every token of the hop needs a hidden row (llama_get_embeddings_ith() per token),
    //    so it keeps the default that follows n_batch.
    //  - tail: 1 row. The floor is n_seq_max, not 1: output_reserve() raises its own bound to
    //    n_seq_max() and then asserts it fits in cparams.n_outputs_max (GGML_ASSERT at the end
    //    of output_reserve), so a stage that ever gains a second sequence must not ask for less
    //    than that. This tool builds a single-sequence context, which is why the max is 1 today.
    //
    // [QLH-SPLIT] v7: except for MAX_VERIFY_ROWS, which is what a per-row verify frame needs. That
    // is not "the tail often answers k rows" -- a plain decode still answers one -- it is the cap
    // the reserve graph has to be *built* with, because cparams is read once at construction and
    // output_reserve() cannot grow the graph later. MAX_VERIFY_ROWS is the shared constant in
    // llama-split-proto.h: the head checks a frame against the same number, so the two ends cannot
    // disagree about what fits.
    if (!s.relay) {
        s.cparams.n_outputs_max = std::max<uint32_t>(MAX_VERIFY_ROWS, s.cparams.n_seq_max);
    }

    if (s.relay) {
        // Middle stages relay the pre-norm hidden, exactly like the local chain does.
        s.cparams.embeddings          = true;
        s.cparams.embeddings_pre_norm = true;
    }

    s.ctx = llama_init_from_model(s.model, s.cparams);
    if (s.ctx == nullptr) {
        fprintf(stderr, "failed to create context\n");
        return false;
    }

    // The constructor clamps n_rs_seq to 0 for architectures without recurrent rollback, so
    // the flag surviving validation is not enough -- the context is the thing that answers.
    if (llama_n_rs_seq(s.ctx) == 0) {
        fprintf(stderr, "[split-stage] this stage has n_rs_seq = 0; it must be configured with "
                        "--rs-seq N. A llama-server driving this chain calls seq_rm() on the "
                        "first request and aborts the process when the stage cannot roll back.\n");
        return false;
    }

    s.vocab   = llama_model_get_vocab(s.model);
    s.n_embd  = (uint32_t) llama_model_n_embd(s.model);
    s.n_vocab = (uint32_t) llama_vocab_n_tokens(s.vocab);

    // [QLH-SPLIT] The chain exchanges fixed-width hidden rows, and each side names that width
    // with a different accessor: the exporter writes n_embd_out (output_reserve sizes t_embd
    // that way), the consumer is validated against n_embd_inp (set_token_embd), and the wire
    // check below uses n_embd. They are equal for both qwen35 models this fork ships, but a
    // model where they differ would copy the wrong number of floats into the reply, or be
    // refused by the peer mid-request. Refuse to start instead, where the reason is legible
    // (the driving side runs the same check in llama_context::split_attach / _remote).
    const uint32_t n_embd_inp = (uint32_t) llama_model_n_embd_inp(s.model);
    const uint32_t n_embd_out = (uint32_t) llama_model_n_embd_out(s.model);
    if (n_embd_out != n_embd_inp || n_embd_inp != s.n_embd) {
        fprintf(stderr, "[split-stage] this model's hidden widths disagree (n_embd=%u, "
                        "n_embd_inp=%u, n_embd_out=%u); a chain hop carries one width\n",
                s.n_embd, n_embd_inp, n_embd_out);
        return false;
    }

    // [QLH-SPLIT] v4 identity: what OP_HELLO answers with. Built once, from numbers the loader
    // already validated, so the value this stage reports cannot disagree with the model it
    // actually runs. The driving side compares it against the stage it expects at this position
    // in the chain (range, source depth, width, vocabulary, role); a wrong stage is then refused
    // at connect time instead of being decoded into.
    s.self = [&] {
        stage_id id{};
        id.layer_lo    = opt.layer_lo;
        id.layer_hi    = opt.layer_hi;
        id.n_layer_src = s.n_layer_src;
        id.n_embd_inp  = n_embd_inp;
        id.n_embd_out  = n_embd_out;
        id.n_vocab     = s.n_vocab;
        id.role        = (uint8_t) (s.relay ? ROLE_RELAY : ROLE_TAIL);
        // [QLH-SPLIT] v8: filled in below, after the MTP context has actually been built. Set here
        // too, so the value the handshake reports is built in one place from the one thing that
        // decides it (whether ctx_mtp exists), not from "the model says it has NextN layers".
        id.has_nextn   = 0;
        return id;
    }();

    // ============================================================================
    // [QLH-SPLIT] v8: the MTP draft context.
    //
    // Only the tail can have one, and only a tail *range* does: the loader keeps blk.<n_layer>.
    // nextn.* exclusively for the range that ends at the source model's last layer (nextn_keep in
    // llama-model-loader.cpp), so a head or middle range loads no NextN block and has nothing to
    // draft with. That is also why this is not a command-line option: whether a stage *can* draft
    // is a property of the range it was given, not of how the operator felt that morning.
    //
    // What the block needs that a tail range does not otherwise load: output_norm / output (the
    // tail has them, they produce its logits) and tok_embd (the MTP block re-embeds the draft token
    // and concatenates that with RMSNorm(h)). The range loader leaves tok_embd unread but mapped
    // (llama-model-loader.cpp, read_lazy), so it is readable on demand; with --no-mmap it is loaded
    // in full, which is the price of drafting from a stage.
    //
    // Failure here must not stop the stage: a chain whose fast path is unavailable is still a
    // correct chain (the same rule as every other remote failure in this tree). The identity
    // reports has_nextn = 0 and OP_MTP_DRAFT answers ST_UNSUPPORTED.
    //
    // Note that cparams are *not* reused: a MTP context is a different graph type, it wants its own
    // KV (only the NextN layer's, which is why the whole thing is a few MiB), no embeddings, no
    // recurrent rollback and n_outputs_max = 1 (it decodes one row per step).
    if (!s.relay) {
        const int32_t n_nextn = llama_model_n_layer_nextn(s.model);
        if (n_nextn <= 0) {
            fprintf(stderr, "[split-stage] this model's tail range has no NextN/MTP block "
                            "(n_layer_nextn = %d): MTP draft is unavailable, remote speculation "
                            "on this chain can only use a drafter that needs no MTP head "
                            "(ngram)\n", n_nextn);
        } else {
            llama_context_params mp = llama_context_default_params();
            mp.ctx_type               = LLAMA_CONTEXT_TYPE_MTP;
            mp.n_ctx                  = s.cparams.n_ctx;
            mp.n_batch                = s.cparams.n_batch;
            mp.n_ubatch               = s.cparams.n_ubatch;
            // [QLH-SPLIT] v9: the draft context gets the same thread counts. Each draft step is one
            // decoded row through the NextN block *plus the full output projection* (the argmax is
            // over the whole 248k vocabulary), so on a slow tail the draft loop is a real share of
            // the round -- it must be tuned with the trunk it shares cores with, not against it.
            mp.n_threads              = s.cparams.n_threads;
            mp.n_threads_batch        = s.cparams.n_threads_batch;
            // One decoded row per draft step, one sequence (the chain is single-sequence by
            // construction). n_outputs_max is what output_reserve() sizes the logits node with;
            // asking for more would hold a k-row buffer for a loop that never uses it.
            mp.n_seq_max              = 1;
            mp.n_outputs_max          = 1;
            mp.n_outputs_max_per_seq  = 1;
            // No recurrent rollback: the draft memory is cleared every round, so there is no state
            // to rewind, and n_rs_seq would size a recurrent cache this context does not have.
            mp.n_rs_seq               = 0;
            mp.embeddings             = false;
            mp.embeddings_pre_norm    = false;
            // The MTP block borrows nothing: this stage's own model carries output.weight, and the
            // draft never converts a token to embeddings through a shared vocab.
            mp.ctx_other              = nullptr;

            s.ctx_mtp = llama_init_from_model(s.model, mp);
            if (s.ctx_mtp == nullptr) {
                fprintf(stderr, "[split-stage] failed to create the MTP draft context over this "
                                "range (n_layer_nextn = %d); MTP draft is unavailable on this "
                                "chain, decodes keep working\n", n_nextn);
            } else {
                // The draft loop chains h rows: each step's input is the previous step's
                // t_h_nextn, so the graph has to export them. Masked, because the loop reads one
                // row (the row it marked as output) per step.
                llama_set_embeddings_nextn(s.ctx_mtp, true, /*masked*/ true);
                s.has_nextn      = true;
                s.self.has_nextn = 1;
                fprintf(stderr, "[split-stage] MTP draft context: %d NextN block(s), n_ctx=%u, "
                                "own KV, n_outputs_max=%u; OP_MTP_DRAFT enabled\n",
                        n_nextn, mp.n_ctx, mp.n_outputs_max);
            }

            // The main tail context also has to export t_h_nextn, because the h row the draft
            // starts from is RMSNorm(last trunk layer output) and only this stage computes that
            // layer. Masked for the same reason: only the rows the frame answers for are shipped
            // back, so a prefill frame costs one row instead of one per prompt token.
            if (s.has_nextn) {
                llama_set_embeddings_nextn(s.ctx, true, /*masked*/ true);
            }
        }
    }

    // [QLH-SPLIT] v10: this line used to end with "listening on host:port", and that string was
    // what a supervisor waited for to decide the stage was up. The bind now happens before the
    // load (see main), so the wait signal moved here: "ready" means the model, the contexts and
    // the offload plan are settled and the accept loop is about to run. The listening line above
    // says only that the port is ours -- it is printed before a single weight is read.
    fprintf(stderr, "[split-stage] ready: layers [%d, %d) of %d, role=%s, n_embd=%u, n_rs_seq=%u\n",
            opt.layer_lo, opt.layer_hi, s.n_layer_src, s.relay ? "relay" : "tail", s.n_embd,
            llama_n_rs_seq(s.ctx));
    // [QLH-SPLIT] v2.2: the offload plan, on one line. A range loads its own block count (tail
    // includes the MTP block), so "-ngl 8" on the tail 32:40 offloads blocks 1..8 of 9 and *not*
    // the whole segment -- a number this line is the only place to read without a verbose log.
    // "all" is reported as the count it expands to so it can be compared against the block count
    // printed next to it.
    {
        // n_layer counts trunk blocks only; a range's local block count includes the NextN
        // head when this is the tail (block_count was overridden to (hi-lo)+nextn), and that is
        // what -ngl counts. Without the +nextn this line said "last 8 of 8" for a 9-block tail.
        const int n_local = llama_model_n_layer(s.model) + llama_model_n_layer_nextn(s.model);
        const int n_eff   = opt.n_gpu_layers < 0 ? n_local : opt.n_gpu_layers;
        fprintf(stderr, "[split-stage] offload: -ngl %s -> last %d of %d local block(s) on GPU%s%s\n",
                opt.n_gpu_layers < 0 ? "all" : std::to_string(opt.n_gpu_layers).c_str(),
                std::min(n_eff, n_local), n_local,
                n_eff >= n_local ? " (+output slot)" : "",
                opt.no_host ? ", host buffer bypassed (--no-host)" : "");
        fprintf(stderr, "[split-stage] moe: cache=%d MiB cpu_share=%.2f repack=%s op_offload=%s "
                        "(expert cache only acts on blocks left in host memory; op_offload makes the "
                        "GPU take over host tensors)\n",
                opt.moe_cache_mib, (double) opt.moe_cache_cpu_share, opt.no_repack ? "off" : "on",
                opt.no_op_offload ? "off" : "on");
    }
    fprintf(stderr, "[split-stage] identity: layers [%d, %d) of %d, n_embd_inp=%u, n_embd_out=%u, "
                    "n_vocab=%u, role=%s, has_nextn=%u (reported in OP_HELLO)\n",
            s.self.layer_lo, s.self.layer_hi, s.self.n_layer_src, s.self.n_embd_inp,
            s.self.n_embd_out, s.self.n_vocab, s.relay ? "relay" : "tail",
            (unsigned) s.self.has_nextn);
    fprintf(stderr, "[split-stage] ctx: role=%s embeddings=%d pre_norm=%d n_outputs_max=%u "
                    "n_outputs_max_per_seq=%u n_batch=%u\n",
            s.relay ? "relay" : "tail", (int) s.cparams.embeddings, (int) s.cparams.embeddings_pre_norm,
            s.cparams.n_outputs_max, s.cparams.n_outputs_max_per_seq, s.cparams.n_batch);
    // [QLH-SPLIT] v9: the numbers a throughput report needs, on one line, at startup.
    //
    // Which threads actually run is the first thing to check when a stage is slower than the
    // machine it is on -- and it is invisible without this line: two stages can be on identical
    // hardware and differ by 20% because one was started with -t and one was not. n_ubatch is here
    // for the same reason: it decides how a verify frame is split, i.e. whether a k-row frame is
    // one batched forward or k single-row ones.
    fprintf(stderr, "[split-stage] threads: n_threads=%d n_threads_batch=%d (%s) "
                    "hardware_concurrency=%u -- a single-row decode uses n_threads, prefill and a "
                    "multi-row verify frame use n_threads_batch\n",
            s.cparams.n_threads, s.cparams.n_threads_batch,
            opt.n_threads > 0 || opt.n_threads_batch > 0 ? "set on the command line"
                                                         : "llama.cpp default (no -t given)",
            std::thread::hardware_concurrency());
    fprintf(stderr, "[split-stage] ubatch: n_ubatch=%u of n_batch=%u%s\n",
            s.cparams.n_ubatch, s.cparams.n_batch,
            s.cparams.n_ubatch < s.cparams.n_batch ? " (frames are split into that many rows per graph)" : "");
    // The handshake identifies the stage; it does not measure it. -c and -b are capacity, not
    // identity, so a stage whose -c is smaller than the driving server's is still accepted here
    // and shows up as ST_NO_SPACE on the first prompt that does not fit -- a request-level
    // failure rather than a broken chain. Say so once, where an operator will see it.
    fprintf(stderr, "[split-stage] note: -c and -b must be at least the driving llama-server's "
                    "-c and -ubatch-size; the identity handshake does not check capacity, so "
                    "anything larger than this stage can hold answers ST_NO_SPACE instead of "
                    "growing the chain's context\n");

    std::random_device snap_rd;
    s.snap_rng = std::mt19937_64(((uint64_t) snap_rd() << 32) ^ (uint64_t) snap_rd());

    // [QLH-SPLIT] v2.3: post-load VRAM check, the last thing before this stage starts serving.
    //
    // The pre-check's estimate is deliberately generous, so it can only catch a gross miss; this
    // compares the VRAM this load actually took against the weights that were supposed to land
    // there and catches the case the estimate got wrong. It deliberately does NOT exit: the spill
    // has already happened, and a warning is the only useful thing left. A spilled tier is
    // sometimes still the fastest one measured -- the criterion is throughput, not this line.
    qlh_vram::after_load(vram_pre.est, vram_pre.dev, "split-stage");

    return true;
}

// =====================================================================================
// [QLH-SPLIT] The listener: bind what the operator spelled out, and nothing wider.
// =====================================================================================

static bool listen_open(const stage_options & o, sock_t & listener) {
    if (!net::global_init()) {
        fprintf(stderr, "WSAStartup failed\n");
        return false;
    }

    listener = socket(AF_INET, SOCK_STREAM, 0);
    if (!net::valid(listener)) {
        fprintf(stderr, "socket() failed\n");
        return false;
    }
    int yes = 1;
#ifdef _WIN32
    // SO_REUSEADDR is not the POSIX flag here: it lets a second process bind a port that is
    // already listening, and connections are then split between the two. Measured: a stale stage
    // and a fresh one both accepted work and the chain returned wrong logits with nothing
    // logged. SO_EXCLUSIVEADDRUSE is the semantics we want on Windows -- no hijacking, while a
    // re-bind right after close still works.
    setsockopt(listener, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, (const char *) &yes, sizeof(yes));
#else
    setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, (const char *) &yes, sizeof(yes));
#endif

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons((uint16_t) o.port);

    // [QLH-SPLIT] Loopback by default, and only a host the caller spells out can widen that.
    // The chain carries no authentication (see the file header), so the safe default is the
    // only default: a stage that binds 0.0.0.0 by accident is an open compute service.
    if (inet_pton(AF_INET, o.host.c_str(), &addr.sin_addr) != 1) {
        fprintf(stderr, "invalid --host '%s': expected an IPv4 address\n", o.host.c_str());
        net::close_sock(listener);
        listener = net::invalid;
        return false;
    }
    // [QLH-SPLIT] v10: the caller now reaches this before the model load, so a failure here is
    // the whole cost of a bad port -- close the socket and hand nothing back. The errno is worth
    // printing because "bind failed" and "that port is already in use" are different problems
    // for the operator, and the first word was not distinguishable from the log before.
    if (bind(listener, (sockaddr *) &addr, sizeof(addr)) != 0) {
#ifdef _WIN32
        const int err = WSAGetLastError();
        const bool in_use = (err == WSAEADDRINUSE);
#else
        const int err = errno;
        const bool in_use = (err == EADDRINUSE);
#endif
        fprintf(stderr, "bind on %s:%d failed (socket error %d%s)\n", o.host.c_str(), o.port, err,
                in_use ? ", port already in use" : "");
        net::close_sock(listener);
        listener = net::invalid;
        return false;
    }
    if (listen(listener, 1) != 0) {
        fprintf(stderr, "listen on %s:%d failed\n", o.host.c_str(), o.port);
        net::close_sock(listener);
        listener = net::invalid;
        return false;
    }
    return true;
}

// =====================================================================================
// [QLH-SPLIT] One connection: the v4 handshake gate, the frame loop, and one function per op.
//
// Each op returns true when the connection can keep serving and false when the frame (or the
// stream) says it cannot: a stream that failed mid-frame is not resynchronised by replying to
// it, which is why the read failures below answer nothing.
// =====================================================================================

struct conn {
    sock_t peer;

    // [QLH-SPLIT] v4: the first frame on a connection must be OP_HELLO. That is what makes
    // the identity check unavoidable -- a caller that skipped it would otherwise be driven
    // as if it had proved which stage this is. `greeted` is per connection: a rebuild opens
    // a new one.
    bool greeted = false;
};

// The header checks every frame has to pass before an op may look at it.
static bool check_frame(conn & c, const req_header & req) {
    if (req.magic != MAGIC) {
        fprintf(stderr, "[split-stage] protocol error: first-frame/header magic %08x, "
                        "expected %08x; the peer is not a split-stage client\n",
                req.magic, MAGIC);
        reply(c.peer, ST_BAD_REQ, 0, 0, 0, nullptr, 0);
        return false;
    }
    // [QLH-SPLIT] Say "version", not "bad request": a version skew is always a build
    // problem (the two ends must come from the same source tree), and an operator who
    // reads only "bad request" starts looking at firewalls.
    if (req.version != PROTO_VER) {
        fprintf(stderr, "[split-stage] protocol version mismatch: the client speaks v%u, "
                        "this build speaks v%u. Both ends must be built from the same "
                        "source tree (rebuild llama-server and split-stage together); "
                        "this is not a network problem\n",
                req.version, PROTO_VER);
        reply(c.peer, ST_BAD_REQ, 0, 0, 0, nullptr, 0);
        return false;
    }
    if (!c.greeted && req.op != OP_HELLO) {
        fprintf(stderr, "[split-stage] protocol error: the first frame is op=%u, not "
                        "OP_HELLO; a stage only accepts work after the v4 identity "
                        "handshake\n", (unsigned) req.op);
        reply(c.peer, ST_BAD_REQ, 0, 0, 0, nullptr, 0);
        return false;
    }
    if (c.greeted && req.op == OP_HELLO) {
        fprintf(stderr, "[split-stage] protocol error: a second OP_HELLO on the same "
                        "connection\n");
        reply(c.peer, ST_BAD_REQ, 0, 0, 0, nullptr, 0);
        return false;
    }
    if (req.n_tokens > MAX_TOKENS || req.n_embd > MAX_EMBD) {
        reply(c.peer, ST_BAD_REQ, 0, 0, 0, nullptr, 0);
        return false;
    }
    return true;
}

// [QLH-SPLIT] A sequence id is an int32 off the wire and it goes straight into the memory layer
// (seq_rm, state save/load, the batch's seq_id), so it is checked here before an op may use it.
//
// Not in check_frame(): that function has no stage, and this bound is the stage's -- it comes
// from the context that was built for this model range. The call sites are the ops that use
// req.seq_id, after the op has consumed its frame body, so a refused frame leaves the stream
// parseable and the connection keeps serving (a negative id used to reach llama_memory_seq_rm).
static bool check_seq_id(conn & c, const stage & s, const req_header & req) {
    const int32_t n_seq_max = (int32_t) llama_n_seq_max(s.ctx);
    if (req.seq_id < 0 || req.seq_id >= n_seq_max) {
        fprintf(stderr, "[split-stage] op=%u carries seq_id=%d, out of range [0, %d); this "
                        "stage's context holds %d sequence(s)\n",
                (unsigned) req.op, req.seq_id, n_seq_max, n_seq_max);
        reply(c.peer, ST_BAD_REQ, 0, 0, 0, nullptr, 0);
        return false;
    }
    return true;
}

static bool op_hello(conn & c, stage & s, const req_header & req) {
    // [QLH-SPLIT] v4 identity handshake. The request carries what the caller
    // expects at this position; the reply is what this stage actually is. The
    // comparison happens on the caller (it is the only side that knows the
    // chain), but the expectation is logged here too, so a rejected connection
    // leaves a trace on both ends instead of only in the head's log.
    if (req.n_bytes != sizeof(stage_id)) {
        fprintf(stderr, "[split-stage] OP_HELLO carried n_bytes=%u, expected %zu\n",
                req.n_bytes, sizeof(stage_id));
        reply(c.peer, ST_BAD_REQ, 0, 0, 0, nullptr, 0);
        return false;
    }
    stage_id expect{};
    if (!net::recv_all(c.peer, &expect, sizeof(expect))) {
        return false;
    }
    {
        const std::string exp_hi = expect.layer_hi == ID_ANY_I32
                ? std::string("?")
                : std::to_string(expect.layer_hi);
        fprintf(stderr, "[split-stage] OP_HELLO: client expects layers [%d, %s) of "
                        "%d, n_embd_inp=%u, n_embd_out=%u, n_vocab=%u, role=%s; "
                        "this stage answers layers [%d, %d) of %d, n_embd_inp=%u, "
                        "n_embd_out=%u, n_vocab=%u, role=%s, has_nextn=%u\n",
                expect.layer_lo, exp_hi.c_str(), expect.n_layer_src,
                expect.n_embd_inp, expect.n_embd_out, expect.n_vocab,
                expect.role == ROLE_TAIL ? "tail" :
                expect.role == ROLE_RELAY ? "relay" : "any",
                s.self.layer_lo, s.self.layer_hi, s.self.n_layer_src,
                s.self.n_embd_inp, s.self.n_embd_out, s.self.n_vocab,
                s.self.role == ROLE_TAIL ? "tail" : "relay",
                (unsigned) s.self.has_nextn);
    }
    c.greeted = true;
    // [QLH-SPLIT] v6: a fresh connection is a fresh chain. The head either just
    // attached (nothing to drop) or just rebuilt a chain it had declared dead,
    // which clears every stage -- so every checkpoint the head kept from before
    // is worthless, and keeping 32 x 12.56 MiB of it for restores that would all
    // be refused ("the stage is behind") is pure waste. Dropping here also means
    // a reconnect cannot resurrect a snapshot the new chain never had.
    if (!s.snapshots.empty()) {
        fprintf(stderr, "[split-stage] OP_HELLO: dropping %zu snapshot(s) from the "
                        "previous chain\n", s.snapshots.size());
        s.snapshots.clear();
    }
    reply(c.peer, ST_OK, 0, 0, 0, &s.self, sizeof(s.self), -1, (uint32_t) sizeof(s.self));
    return true;
}

static bool op_clear(conn & c, stage & s, const req_header & req) {
    (void) req;
    llama_memory_clear(llama_get_memory(s.ctx), true);
    reply(c.peer, ST_OK, 0, 0, 0, nullptr, 0);
    return true;
}

static bool op_seq_rm(conn & c, stage & s, const req_header & req) {
    // [QLH-SPLIT] The observable half of the fix that made this stage's KV
    // trimable at all: before the aggregate memory held remote stages, nobody
    // ever sent this op, so the stage's cache grew until it refused to decode.
    // Logging p0/p1 with the position before and after is what lets an operator
    // line it up against the head's own "memory_seq_rm [p0, end)" line.
    if (!check_seq_id(c, s, req)) {
        return true;
    }
    llama_memory_t mem = llama_get_memory(s.ctx);
    const llama_pos pos_max_before = llama_memory_seq_pos_max(mem, req.seq_id);
    const bool ok = llama_memory_seq_rm(mem, req.seq_id, req.pos0, req.pos1);
    const llama_pos pos_max_after = llama_memory_seq_pos_max(mem, req.seq_id);
    fprintf(stderr, "[split-stage] OP_SEQ_RM seq=%d p0=%d p1=%d -> %s, pos_max %d -> %d\n",
            req.seq_id, req.pos0, req.pos1, ok ? "ok" : "refused",
            (int) pos_max_before, (int) pos_max_after);
    reply(c.peer, ok ? ST_OK : ST_INTERNAL, 0, 0, 0, nullptr, 0);
    return true;
}

static bool op_state_save(conn & c, stage & s, const req_header & req) {
    // [QLH-SPLIT] The checkpoint channel. llama-server asks the head for its
    // sequence state when it creates a context checkpoint; the request comes
    // here for every stage after it. The size query is the same op with
    // STATE_FLAG_SIZE_ONLY set: the caller has to size its buffer before it asks
    // for the bytes.
    //
    // v6: with STATE_FLAG_HANDLE (which the head always sets) the snapshot stays
    // here and the reply is an 8-byte name for it instead of 12.56 MiB of state.
    // The size query carries the same flag, so "how big is the record" and "give
    // me the record" are answered in the same mode -- the head's buffer is sized
    // from the first and written by the second.
    if (!check_seq_id(c, s, req)) {
        return true;
    }
    const bool want_handle  = (req.flags & STATE_FLAG_HANDLE) != 0;
    const llama_state_seq_flags sflags = (llama_state_seq_flags)
            (req.flags & ~(uint8_t) (STATE_FLAG_SIZE_ONLY | STATE_FLAG_HANDLE));
    llama_memory_t mem = llama_get_memory(s.ctx);
    const int32_t pm = (int32_t) llama_memory_seq_pos_max(mem, req.seq_id);

    if (req.flags & STATE_FLAG_SIZE_ONLY) {
        const size_t n = want_handle
                ? (size_t) STATE_HANDLE_BYTES
                : llama_state_seq_get_size_ext(s.ctx, req.seq_id, sflags);
        if (n > MAX_STATE) {
            reply(c.peer, ST_INTERNAL, 0, 0, 0, nullptr, 0);
            return true;
        }
        fprintf(stderr, "[split-stage] OP_STATE_SAVE seq=%d flags=%u size-only -> %zu bytes, pos_max=%d\n",
                req.seq_id, (unsigned) sflags, n, (int) pm);
        reply(c.peer, ST_OK, 0, 0, 0, nullptr, 0, pm, (uint32_t) n);
        return true;
    }

    const size_t n = llama_state_seq_get_size_ext(s.ctx, req.seq_id, sflags);
    if (n > MAX_STATE) {
        reply(c.peer, ST_INTERNAL, 0, 0, 0, nullptr, 0);
        return true;
    }
    std::vector<uint8_t> blob(n);
    const size_t got = n > 0
            ? llama_state_seq_get_data_ext(s.ctx, blob.data(), n, req.seq_id, sflags)
            : 0;
    if (got != n) {
        fprintf(stderr, "[split-stage] OP_STATE_SAVE seq=%d failed: %zu of %zu bytes\n",
                req.seq_id, got, n);
        reply(c.peer, ST_INTERNAL, 0, 0, 0, nullptr, 0, pm, 0);
        return true;
    }

    if (!want_handle) {
        // Pre-v6 shape, kept because the op still means it: a caller that asks for
        // the blob gets the blob. Nothing in this tree sends it any more (the head
        // always asks for a handle) -- see the proto header's note on the flag.
        fprintf(stderr, "[split-stage] OP_STATE_SAVE seq=%d flags=%u -> %zu bytes, pos_max=%d\n",
                req.seq_id, (unsigned) sflags, n, (int) pm);
        reply(c.peer, ST_OK, 0, 0, 0, blob.data(), blob.size(), pm, (uint32_t) n);
        return true;
    }

    // Keep it and hand back a name. The bytes are the same ones the pre-v6 reply
    // carried -- this is the same snapshot, stored instead of sent -- which is
    // what keeps a restored checkpoint bit-identical to the old path.
    uint64_t handle = 0;
    do {
        handle = s.snap_rng();
    } while (handle == 0 ||
             std::find_if(s.snapshots.begin(), s.snapshots.end(),
                     [handle](const std::pair<uint64_t, stage_snapshot> & e) {
                         return e.first == handle;
                     }) != s.snapshots.end());
    s.snapshots.emplace_back(handle, stage_snapshot{ pm, std::move(blob) });
    while (s.snapshots.size() > s.opt.state_cache) {
        fprintf(stderr, "[split-stage] snapshot cache full (%u): evicting handle %llu "
                        "(pos_max=%d), any restore naming it will answer STALE\n",
                s.opt.state_cache, (unsigned long long) s.snapshots.front().first,
                (int) s.snapshots.front().second.pos_max);
        s.snapshots.pop_front();
    }
    fprintf(stderr, "[split-stage] OP_STATE_SAVE seq=%d flags=%u -> handle %llu "
                    "(%zu bytes kept, %zu/%u cached), pos_max=%d\n",
            req.seq_id, (unsigned) sflags, (unsigned long long) handle, n,
            s.snapshots.size(), s.opt.state_cache, (int) pm);
    reply(c.peer, ST_OK, 0, 0, 0, &handle, sizeof(handle), pm, STATE_HANDLE_BYTES);
    return true;
}

static bool op_state_load(conn & c, stage & s, const req_header & req) {
    // v6: with STATE_FLAG_HANDLE the frame is 8 bytes naming a snapshot in
    // `snapshots`; without it the frame carries the blob itself, which is only
    // reachable from a slot file written by a pre-v6 build (see state_read in
    // llama-memory-split.cpp -- it keeps that branch on purpose).
    const bool want_handle = (req.flags & STATE_FLAG_HANDLE) != 0;

    std::vector<uint8_t> blob;
    uint64_t handle = 0;
    if (want_handle) {
        if (!net::recv_all(c.peer, &handle, sizeof(handle))) {
            return false;
        }
    } else {
        if (req.n_bytes > MAX_STATE) {
            reply(c.peer, ST_BAD_REQ, 0, 0, 0, nullptr, 0);
            return true;
        }
        blob.resize(req.n_bytes);
        if (req.n_bytes > 0 && !net::recv_all(c.peer, blob.data(), blob.size())) {
            return false;
        }
    }

    const llama_state_seq_flags sflags = (llama_state_seq_flags)
            (req.flags & ~(uint8_t) STATE_FLAG_HANDLE);
    // After the body: a refused frame whose bytes were still unread would be parsed as the next
    // header.
    if (!check_seq_id(c, s, req)) {
        return true;
    }
    llama_memory_t mem = llama_get_memory(s.ctx);
    const int32_t pm = (int32_t) llama_memory_seq_pos_max(mem, req.seq_id);

    // Non-null once the handle resolved: the entry the bytes came from, so they can
    // be put back after the restore (a pointer into the deque stays valid across
    // emplace_back, unlike an iterator, and nothing else touches it here).
    stage_snapshot * snap = nullptr;
    if (want_handle) {
        auto it = std::find_if(s.snapshots.begin(), s.snapshots.end(),
                [handle](const std::pair<uint64_t, stage_snapshot> & e) {
                    return e.first == handle;
                });
        if (it == s.snapshots.end()) {
            // [QLH-SPLIT] The one answer this op can give that is not an error. The
            // snapshot was evicted, or this process minted no such handle: nothing
            // has been applied and nothing is inconsistent, so the head is told
            // "you have no checkpoint here" (ST_STALE) and re-processes the prompt
            // -- it must not treat this as a broken chain.
            fprintf(stderr, "[split-stage] OP_STATE_LOAD seq=%d handle %llu -> STALE "
                            "(%zu/%u cached); nothing applied\n",
                    req.seq_id, (unsigned long long) handle,
                    s.snapshots.size(), s.opt.state_cache);
            reply(c.peer, ST_STALE, 0, 0, 0, nullptr, 0, pm, 0);
            return true;
        }
        // Most recently used goes last: that is the eviction order the cache is
        // sized by, and a checkpoint the head keeps restoring should not be the
        // first thing to go.
        if (std::next(it) != s.snapshots.end()) {
            stage_snapshot entry = std::move(it->second);
            s.snapshots.erase(it);
            s.snapshots.emplace_back(handle, std::move(entry));
            it = std::prev(s.snapshots.end());
        }
        snap = &it->second;
        blob.swap(snap->blob);
    }

    // Validate before applying: a stage that has moved backwards (cleared, or
    // never saw the tokens this snapshot sits on) cannot host it, and applying
    // first would leave the chain's stages disagreeing even though the caller
    // then fails. The head is restored only after every stage has accepted.
    if (pm < req.pos0) {
        fprintf(stderr, "[split-stage] OP_STATE_LOAD seq=%d refused: pos_max=%d is "
                        "behind the snapshot's %d\n", req.seq_id, (int) pm, req.pos0);
        if (snap != nullptr) {
            // Put it back: the caller gets a refusal, not a lost snapshot.
            blob.swap(snap->blob);
        }
        reply(c.peer, ST_BAD_REQ, 0, 0, 0, nullptr, 0, pm, 0);
        return true;
    }

    const size_t got = llama_state_seq_set_data_ext(s.ctx, blob.data(), blob.size(),
            req.seq_id, sflags);
    const bool ok = got == blob.size();
    const std::string what = want_handle
            ? "handle " + std::to_string((unsigned long long) handle) + ", "
            : std::string();
    fprintf(stderr, "[split-stage] OP_STATE_LOAD seq=%d flags=%u %s%zu bytes -> %s, pos_max before=%d\n",
            req.seq_id, (unsigned) sflags, what.c_str(), blob.size(),
            ok ? "ok" : "refused", (int) pm);
    if (snap != nullptr) {
        // Restored from the cache, so the bytes stay: the same checkpoint can be
        // restored by a later request (the head keeps it until it is invalidated).
        blob.swap(snap->blob);
    }
    reply(c.peer, ok ? ST_OK : ST_INTERNAL, 0, 0, 0, nullptr, 0, pm, 0);
    return true;
}

// Read one OP_DECODE frame: positions, hidden rows, and the optional candidate request, with
// every check that has to happen before a single byte is decoded. Returns true when the whole
// frame has been consumed; a false means the stream is no longer parseable, so nothing is
// answered to it.
static bool decode_read_frame(conn & c, stage & s, const req_header & req, uint32_t n_tok,
                              bool & want_cand, cand_req & creq,
                              std::vector<llama_logit_bias> & cbias,
                              std::vector<llama_token> & chist,
                              bool & want_verify, bool & want_nextn,
                              std::vector<int32_t> & tap_ask) {
    // section-major, n_pos per token: the caller sends what the head's graph
    // actually used, so the stride has to be the one both ends agree on
    if (req.n_pos == 0 || req.n_pos > GGML_MROPE_SECTIONS) {
        reply(c.peer, ST_BAD_REQ, 0, 0, 0, nullptr, 0);
        return false;
    }
    if (req.n_pos != s.n_pos_per_embd) {
        fprintf(stderr, "[split-stage] OP_DECODE n_pos=%u but this stage uses %u; "
                        "the two ends were not built from the same model\n",
                req.n_pos, s.n_pos_per_embd);
        reply(c.peer, ST_BAD_REQ, 0, 0, 0, nullptr, 0);
        return false;
    }
    // [QLH-SPLIT] The hidden rows are read at *this* stage's width, not at the
    // width the header declares, so a frame from a head built for a different
    // model does not fail: it consumes n_tok * n_embd bytes, leaves the rest of
    // its hidden block in the stream, and every later header is parsed out of
    // that data. The client then blocks on a reply that never comes and gives up
    // after its 120 s timeout, with nothing logged on either side. The n_pos
    // check above exists for the same reason; n_embd was the half that was
    // missing. Reject before reading anything, exactly like n_pos.
    if (req.n_embd != s.n_embd) {
        fprintf(stderr, "[split-stage] OP_DECODE n_embd=%u but this stage is %u wide; "
                        "the two ends were not built from the same model\n",
                req.n_embd, s.n_embd);
        reply(c.peer, ST_BAD_REQ, 0, 0, 0, nullptr, 0);
        return false;
    }
    s.positions.resize((size_t) n_tok * s.n_pos_per_embd);
    if (!net::recv_all(c.peer, s.positions.data(), s.positions.size() * sizeof(llama_pos))) {
        return false;
    }

    // OP_DECODE always carries hidden rows: every stage of a chain consumes
    // them, the tail included. Reading them conditionally (on flags) left the
    // bytes in the stream and desynchronized the connection -- the next request
    // header was parsed out of hidden data.
    s.hidden.resize((size_t) n_tok * s.n_embd);
    if (!net::recv_all(c.peer, s.hidden.data(), s.hidden.size() * sizeof(float))) {
        return false;
    }

    // [QLH-SPLIT] v5 candidate request, only when the caller asked for one.
    // Read before anything can refuse the frame: the connection is a single
    // stream, so a reply sent while part of this frame is still unread would be
    // followed by those bytes on the next request (see the n_embd check above).
    want_cand = (req.flags & DECODE_FLAG_CANDIDATES) != 0;
    if (want_cand) {
        if (!net::recv_all(c.peer, &creq, sizeof(creq))) {
            return false;
        }
        if (creq.n_bias > kMaxCandBias || creq.n_hist > kMaxCandHist ||
            creq.n_steps == 0 || creq.n_steps > sizeof(creq.steps)) {
            fprintf(stderr, "[split-stage] OP_DECODE candidate request is out of "
                            "range (n_bias=%u n_hist=%u n_steps=%u)\n",
                    creq.n_bias, creq.n_hist, (unsigned) creq.n_steps);
            return false;
        }
        cbias.resize(creq.n_bias);
        if (creq.n_bias > 0 &&
            !net::recv_all(c.peer, cbias.data(), cbias.size() * sizeof(llama_logit_bias))) {
            return false;
        }
        chist.resize(creq.n_hist);
        if (creq.n_hist > 0 &&
            !net::recv_all(c.peer, chist.data(), chist.size() * sizeof(llama_token))) {
            return false;
        }
        // A relay exports hidden and cannot answer with candidates; the driving
        // side never asks (it only asks the last stage), so this is a frame from
        // something other than a head built from this tree. Refuse after the
        // whole frame is consumed, so the connection stays parseable.
        if (s.relay) {
            fprintf(stderr, "[split-stage] OP_DECODE asked a relay stage for "
                            "candidates; only a tail can answer those\n");
            reply(c.peer, ST_BAD_REQ, 0, 0, 0, nullptr, 0);
            return false;
        }
    }

    // [QLH-SPLIT] v7 per-row verify request. Unlike the candidate block this one adds no bytes to
    // the frame, so there is nothing to read; what it changes is how many rows this stage must
    // mark as output, which is decided in decode_run.
    want_verify = (req.flags & DECODE_FLAG_VERIFY_ROWS) != 0;
    if (want_verify) {
        // Both at once is a frame no build of this tree sends. Refuse it *after* the frame has
        // been consumed (the candidate block above was read) so the connection stays parseable.
        if (want_cand) {
            fprintf(stderr, "[split-stage] OP_DECODE set both DECODE_FLAG_CANDIDATES and "
                            "DECODE_FLAG_VERIFY_ROWS; they answer different questions and "
                            "cannot both be asked of one frame\n");
            reply(c.peer, ST_BAD_REQ, 0, 0, 0, nullptr, 0);
            return false;
        }
        if (s.relay) {
            fprintf(stderr, "[split-stage] OP_DECODE asked a relay stage for per-row results; "
                            "only a tail produces logits to take an argmax of\n");
            reply(c.peer, ST_BAD_REQ, 0, 0, 0, nullptr, 0);
            return false;
        }
    }

    // [QLH-SPLIT] v8: the h_nextn rows a decode answers with. Like the verify flag this adds no
    // bytes to the frame -- it only changes what the reply carries, which decode_run builds.
    //
    // A stage that *cannot* produce them does not refuse the frame: a decode must answer what it
    // can, and the head already knows from the handshake whether asking was worthwhile. Ignoring
    // the flag leaves the reply in exactly the v7 shape, which the head parses (it reads the rows
    // because of the reply flag, not because it asked), so a tail that lost its MTP context between
    // attach and now degrades to "no drafts" instead of failing the request. That is the same rule
    // OP_MTP_DRAFT follows with ST_UNSUPPORTED, one level earlier.
    want_nextn = (req.flags & DECODE_FLAG_NEXTN_ROWS) != 0;
    if (want_nextn) {
        if (s.relay) {
            fprintf(stderr, "[split-stage] OP_DECODE asked a relay stage for h_nextn rows; only a "
                            "tail owns the model's last layer and the NextN block\n");
            reply(c.peer, ST_BAD_REQ, 0, 0, 0, nullptr, 0);
            return false;
        }
        if (!s.has_nextn || s.ctx_mtp == nullptr) {
            if (!s.logged_no_nextn) {
                s.logged_no_nextn = true;
                fprintf(stderr, "[split-stage] OP_DECODE asked for h_nextn rows but this stage has "
                                "no NextN/MTP block; answering without them (the head will not "
                                "draft), which is not an error\n");
            }
            want_nextn = false;
        }
    }

    // [QLH-SPLIT] v9: the layer-input taps this frame asks for. Read last, after the candidate block,
    // because that is where the frame puts it (see layer_inp_req).
    //
    // Unlike the two flags above this one changes the frame: the request names the layers, so the
    // bytes have to be consumed even when this stage owns none of them -- a half-read frame is a
    // desynchronized connection, and a relay that owns no tap is a perfectly ordinary stage.
    //
    // The ids are *source-model* layer ids and this stage answers the ones inside its own range. That
    // is the whole reason they are source ids: the head does not have to know where the chain was cut,
    // and a three-stage chain needs no extra mapping.
    tap_ask.clear();
    if (req.flags & DECODE_FLAG_LAYER_INP) {
        layer_inp_req treq{};
        if (!net::recv_all(c.peer, &treq, sizeof(treq))) {
            return false;
        }
        if (treq.n_taps == 0 || treq.n_taps > MAX_LAYER_INP_TAPS || treq.n_embd == 0 ||
            treq.n_embd > MAX_EMBD) {
            fprintf(stderr, "[split-stage] OP_DECODE layer-input request is out of range "
                            "(n_taps=%u n_embd=%u, at most %u taps and %u wide)\n",
                    treq.n_taps, treq.n_embd, MAX_LAYER_INP_TAPS, MAX_EMBD);
            // The ids still follow the refused block, so read them before answering or the next
            // header would be parsed out of them.
            std::vector<int32_t> skip(treq.n_taps <= MAX_TOKENS ? treq.n_taps : 0);
            if (!skip.empty() &&
                !net::recv_all(c.peer, skip.data(), skip.size() * sizeof(int32_t))) {
                return false;
            }
            reply(c.peer, ST_BAD_REQ, 0, 0, 0, nullptr, 0);
            return false;
        }
        tap_ask.resize(treq.n_taps);
        if (!net::recv_all(c.peer, tap_ask.data(), tap_ask.size() * sizeof(int32_t))) {
            return false;
        }
        // A tap row is a layer input, which is always the trunk width (src/llama-context.cpp,
        // extract_layer_inputs asserts it). A frame from a different model is refused here rather
        // than answered at a stride nobody agrees on.
        if (treq.n_embd != s.n_embd) {
            fprintf(stderr, "[split-stage] OP_DECODE asked for layer-input taps of width %u but this "
                            "stage is %u wide; the two ends were not built from the same model\n",
                    treq.n_embd, s.n_embd);
            reply(c.peer, ST_BAD_REQ, 0, 0, 0, nullptr, 0);
            return false;
        }
    }

    return true;
}

// [QLH-SPLIT] v9: turn a frame's layer-input request into this stage's own layers.
//
// The ids in the request are *source-model* layer numbers, so the only thing this stage needs in order
// to answer them is its own offset: local = id - layer_lo. Ids outside [layer_lo, layer_hi) belong to
// another stage and are dropped -- that is what makes one frame sendable to every stage of a chain,
// and it is why the head does not have to know where the chain was cut.
//
// Extraction is enabled through llama_set_embeddings_layer_inp once per layer, never per frame (see
// stage::tap_on): that call marks the scheduler for a re-reserve, so repeating it on every decode
// would re-reserve the graph on every decode. The head asks for the same layers every time, so the
// reserve is paid once.
static void stage_prepare_taps(stage & s, const std::vector<int32_t> & ask) {
    s.tap_ids_out.clear();
    s.tap_rows_out.clear();
    if (ask.empty()) {
        return;
    }
    const int32_t lo = s.opt.layer_lo;
    const int32_t hi = s.opt.layer_hi;
    if (s.tap_on.empty()) {
        s.tap_on.assign((size_t) std::max(0, hi - lo), 0);
    }
    for (const int32_t id : ask) {
        if (id < lo || id >= hi) {
            continue;
        }
        const uint32_t lid = (uint32_t) (id - lo);
        s.tap_ids_out.push_back(id);
        if (!s.tap_on[lid]) {
            llama_set_embeddings_layer_inp(s.ctx, lid, true);
            s.tap_on[lid] = 1;
        }
    }
    if (!s.logged_tap) {
        s.logged_tap = true;
        fprintf(stderr, "[split-stage] first OP_DECODE layer-input request: %zu ids, %zu owned by "
                        "this stage, %u wide\n",
                ask.size(), s.tap_ids_out.size(), s.n_embd);
    }
}

// Read the rows of every tap this stage owns, into [row][tap][n_embd] -- the layout
// RSP_FLAG_LAYER_INP documents. Called after the decode and before the batch is freed, from the
// context's own host-side buffer: every row of the frame, because a draft injects every position of
// the batch into its KV, not just an anchor row.
static bool stage_collect_taps(stage & s, uint32_t n_tok) {
    if (s.tap_ids_out.empty()) {
        return true;
    }
    s.tap_rows_out.resize((size_t) n_tok * s.tap_ids_out.size() * s.n_embd);
    for (size_t t = 0; t < s.tap_ids_out.size(); ++t) {
        const uint32_t lid = (uint32_t) (s.tap_ids_out[t] - s.opt.layer_lo);
        const float * p = llama_get_embeddings_layer_inp(s.ctx, lid);
        if (p == nullptr) {
            // Nothing honest to answer: a fabricated row would be injected into the draft model's KV
            // as if it were this layer's input. The head reads "no taps" as "draft without the
            // injection", which is a slower request, not a wrong one.
            fprintf(stderr, "[split-stage] layer-input tap for source layer %d has no rows; this "
                            "context did not export them\n", s.tap_ids_out[t]);
            return false;
        }
        for (uint32_t r = 0; r < n_tok; ++r) {
            memcpy(s.tap_rows_out.data() + (((size_t) r * s.tap_ids_out.size()) + t) * s.n_embd,
                   p + (size_t) r * s.n_embd, s.n_embd * sizeof(float));
        }
    }
    return true;
}

// Build the batch for a frame that has already been read, decode it, and answer. Returns true
// when the connection can keep serving.
static bool decode_run(conn & c, stage & s, const req_header & req, uint32_t n_tok,
                       bool want_cand, const cand_req & creq,
                       const std::vector<llama_logit_bias> & cbias,
                       const std::vector<llama_token> & chist,
                       bool want_verify, bool want_nextn) {
    // [QLH-SPLIT] v9: resolve the frame's layer-input request into this stage's own layers before the
    // batch is built, because enabling extraction is a context setting that has to be in place when
    // the graph is reserved for this decode.
    stage_prepare_taps(s, s.tap_ask);
    const bool want_tap = !s.tap_ids_out.empty();
    // Compatibility batch, not llama_batch_ext. This is a reach for the stable
    // API, not a workaround: llama_decode() is the compat shell that builds an
    // ext batch internally (llama-context.cpp), both routes end in the same
    // decode_plain(), and both copy the hidden rows into an owned vector -- so
    // the ext path would buy nothing here. Its output side is still being
    // finished upstream (src/llama-batch.h: "TODO: have dedicated output
    // flags"), and a stage server is the wrong place to be its first user.
    // Whatever makes the tail's logits come out empty, it is not this choice:
    // tests/split-smoke.cpp runs a tail-only context through llama_batch_ext
    // with set_output_logits() and gets logits out.
    // [QLH-SPLIT] v9: frame counters and phase timers (both gated; no-op when the env knob is off).
    // The frame id is in every trace line, so "which frame did this row belong to" needs no clock.
    const uint64_t frame_id = ++s.frame_id;
    const double   t_frame  = g_timing ? now_ms() : 0.0;

    llama_batch batch = llama_batch_init((int32_t) n_tok, (int32_t) s.n_embd, 1);
    batch.n_tokens = (int32_t) n_tok;

    // [QLH-SPLIT] v7: the row count a verify frame asks for must fit in the cap the context was
    // built with, because output_reserve() cannot grow it later -- a decode that marks more rows
    // than n_outputs_max hits its closing GGML_ASSERT, i.e. an abort of this process, and the
    // caller sees a dropped connection. The head checks the same number (MAX_VERIFY_ROWS) before
    // it asks, so this only fires on a frame from something that is not a head of this tree.
    if (want_verify && n_tok > s.cparams.n_outputs_max) {
        fprintf(stderr, "[split-stage] OP_DECODE asks for %u per-row results but this stage was "
                        "built for %u; the frame is from a head that does not cap at "
                        "MAX_VERIFY_ROWS (%u)\n",
                n_tok, s.cparams.n_outputs_max, MAX_VERIFY_ROWS);
        llama_batch_free(batch);
        reply(c.peer, ST_BAD_REQ, 0, 0, 0, nullptr, 0);
        return true;
    }

    // [QLH-SPLIT] llama_batch_init sizes pos for one position per token, but the
    // compat conversion of an embedding batch reads n_pos_per_embd (4 for M-RoPE)
    // positions per token, section-major pos[j*n_tok + i]. Reading only n_tok
    // entries ran past the end of the allocation for three quarters of the
    // positions -- the zeroes the graph saw were whatever heap memory followed the
    // buffer, which is why this only ever worked by accident. Replace the array
    // with the section-major positions the caller sent.
    free(batch.pos);
    batch.pos = (llama_pos *) malloc((size_t) n_tok * s.n_pos_per_embd * sizeof(llama_pos));
    if (n_tok > 0 && batch.pos == nullptr) {
        llama_batch_free(batch);
        reply(c.peer, ST_INTERNAL, 0, 0, 0, nullptr, 0);
        return true;
    }
    if (!s.positions.empty()) {
        memcpy(batch.pos, s.positions.data(), s.positions.size() * sizeof(llama_pos));
    }

    for (uint32_t i = 0; i < n_tok; ++i) {
        memcpy(batch.embd + (size_t) i * s.n_embd, s.hidden.data() + (size_t) i * s.n_embd,
               s.n_embd * sizeof(float));
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = req.seq_id;
        // relay stages ask for hidden rows on every token; the tail needs logits
        // for the last token only, which is the one the caller samples from.
        //
        // [QLH-SPLIT] v7: a per-row verify frame is the one exception -- there the caller samples
        // from *every* row (k drafts + 1 target), so every row must keep its logits. Marking them
        // is also what allocates the rows: output_reserve() sizes the logits buffer from the
        // batch's output count, and cparams.n_outputs_max (MAX_VERIFY_ROWS for a tail) is the
        // ceiling that makes n_tok > MAX_VERIFY_ROWS a refusal instead of an assert.
        batch.logits[i] = s.relay ? 0 : (want_verify || i + 1 == n_tok ? 1 : 0);
    }

    // [QLH-SPLIT] Print the position the stage will actually use, once per
    // process: with M-RoPE the extra dimensions are easy to get wrong, and the
    // first thing to look at when a chain starts handling image tokens is whether
    // the stage sees the same [t, y, x, z] the head used. For a text token the
    // head's graph expands it to [t, t, t, 0], so that is what must show here.
    if (!s.logged_pos && n_tok > 0) {
        s.logged_pos = true;
        const size_t last = (size_t) n_tok - 1;
        fprintf(stderr, "[split-stage] first OP_DECODE: n_tok=%u n_embd=%u "
                        "n_pos_per_embd=%u; last-token pos = [",
                n_tok, req.n_embd, s.n_pos_per_embd);
        for (uint32_t j = 0; j < s.n_pos_per_embd; ++j) {
            fprintf(stderr, "%d%s", (int) batch.pos[(size_t) j * n_tok + last],
                    j + 1 == s.n_pos_per_embd ? "" : ", ");
        }
        fprintf(stderr, "]\n");
    }

    const double t_build = g_timing ? now_ms() : 0.0;

    const int rc = llama_decode(s.ctx, batch);

    if (rc != 0) {
        llama_batch_free(batch);
        // Say why, not just what: a refusal here used to be a 4-byte status and nothing else, and
        // the head's message ("stage rejected the request") is the only place an operator could
        // see it. The three cases are different problems (capacity, bad frame, compute failure).
        fprintf(stderr, "[split-stage] OP_DECODE n_tok=%u seq=%d failed rc=%d (%s)\n",
                n_tok, req.seq_id, (int) rc,
                rc == 1  ? "no KV room, -c of this stage is smaller than the driving server's"
              : rc == -1 ? "invalid batch, the frame does not describe what this context accepts"
                         : "compute failure");
        const status st = rc == 1 ? ST_NO_SPACE : (rc == -1 ? ST_BAD_REQ : ST_INTERNAL);
        reply(c.peer, st, 0, 0, 0, nullptr, 0);
        return true;
    }

    const double t_decode = g_timing ? now_ms() : 0.0;

    // [QLH-SPLIT] v9: the layer-input taps, read here for every role -- a relay owns interior layers
    // and is exactly where a DFlash tap lands on a three-stage chain, so unlike the h_nextn rows these
    // are not a tail privilege.
    const bool tap_ok = stage_collect_taps(s, n_tok);
    const uint8_t  tap_flag = (want_tap && tap_ok) ? RSP_FLAG_LAYER_INP : (uint8_t) 0;
    const uint8_t  tap_n    = (want_tap && tap_ok) ? (uint8_t) s.tap_ids_out.size() : (uint8_t) 0;
    const void *   tap_ids_p   = (want_tap && tap_ok) ? (const void *) s.tap_ids_out.data() : nullptr;
    const size_t   tap_ids_n   = (want_tap && tap_ok) ? s.tap_ids_out.size() * sizeof(int32_t) : 0;
    const void *   tap_rows_p  = (want_tap && tap_ok) ? (const void *) s.tap_rows_out.data() : nullptr;
    const size_t   tap_rows_n  = (want_tap && tap_ok) ? s.tap_rows_out.size() * sizeof(float) : 0;
    const uint32_t tap_embd    = (want_tap && tap_ok) ? s.n_embd : 0;
    const uint32_t tap_bytes   = (uint32_t) (tap_ids_n + tap_rows_n);

    if (s.relay) {
        std::vector<float> out((size_t) n_tok * s.n_embd);
        bool ok = true;
        for (uint32_t i = 0; i < n_tok; ++i) {
            const float * row = llama_get_embeddings_ith(s.ctx, (int32_t) i);
            if (row == nullptr) {
                ok = false;
                break;
            }
            memcpy(out.data() + (size_t) i * s.n_embd, row, s.n_embd * sizeof(float));
        }
        llama_batch_free(batch);
        if (!ok) {
            reply(c.peer, ST_INTERNAL, 0, 0, 0, nullptr, 0);
            return true;
        }
        // [QLH-SPLIT] v9: hidden rows, then the tap ids, then the tap rows -- the sections a reply that
        // carries both puts in that order (see RSP_FLAG_LAYER_INP). n_embd stays the hidden width, so
        // the tap width is the one the frame asked for.
        reply_parts(c.peer, ST_OK, n_tok, s.n_embd, 0, -1,
                    (uint32_t) (out.size() * sizeof(float) + tap_bytes), tap_flag, tap_n,
                    out.data(), out.size() * sizeof(float),
                    tap_ids_p, tap_ids_n, tap_rows_p, tap_rows_n);
        return true;
    }

    const int32_t last = (int32_t) n_tok - 1;
    const float * lg = llama_get_logits_ith(s.ctx, last);
    if (lg == nullptr) {
        llama_batch_free(batch);
        reply(c.peer, ST_INTERNAL, 0, 0, 0, nullptr, 0);
        return true;
    }

    // [QLH-SPLIT] v8: the h_nextn rows, when the frame asked for them. Only the rows this frame
    // answers for: all of them for a verify frame, the last one for a plain decode (the same rule
    // batch.logits followed when the batch was built above). That is what keeps a prefill from
    // shipping a row per prompt token -- the draft only ever starts from the position it last
    // accepted, which is the last row of the batch.
    if (want_nextn) {
        const uint32_t row0 = want_verify ? 0 : (uint32_t) last;
        s.nextn_out.resize((size_t) (n_tok - row0) * s.n_embd);
        for (uint32_t i = row0; i < n_tok; ++i) {
            const float * h = llama_get_embeddings_nextn_ith(s.ctx, (int32_t) i);
            if (h == nullptr) {
                // The context was built without the t_h_nextn channel, or the row was not marked
                // as output. Either way there is no honest answer, and a fabricated h row would
                // draft from another token's state -- refuse instead.
                fprintf(stderr, "[split-stage] OP_DECODE asked for h_nextn rows but row %u has "
                                "none; this context does not export t_h_nextn for it\n", i);
                llama_batch_free(batch);
                reply(c.peer, ST_INTERNAL, 0, 0, 0, nullptr, 0);
                return true;
            }
            memcpy(s.nextn_out.data() + (size_t) (i - row0) * s.n_embd, h,
                   s.n_embd * sizeof(float));
        }
    }

    const uint8_t  nextn_flag  = want_nextn ? RSP_FLAG_NEXTN_ROWS : (uint8_t) 0;
    const uint32_t nextn_embd  = want_nextn ? s.n_embd : 0;
    const uint32_t nextn_bytes = want_nextn ? (uint32_t) (s.nextn_out.size() * sizeof(float)) : 0;
    const void *   nextn_data  = want_nextn ? (const void *) s.nextn_out.data() : nullptr;

    // [QLH-SPLIT] v7: the per-row answer. One int32 per row, in row order: the argmax of that
    // row's logits. Nothing else is needed, because the only caller that asks is a greedy
    // verification (temp 0 / llama_sampler_init_greedy), where the token a row produces *is* its
    // argmax -- no sampling parameters, no candidate array, no comparison of distributions.
    //
    // The row is read through llama_get_logits_ith(i), which is only valid for rows the batch
    // marked as output; this branch is the only place a tail marks every row, so all n_tok rows
    // are there by construction (and n_tok <= n_outputs_max was checked above).
    if (want_verify) {
        const double t_argmax0 = g_timing ? now_ms() : 0.0;
        s.verify_out.resize(n_tok);
        for (uint32_t i = 0; i < n_tok; ++i) {
            const float * row = llama_get_logits_ith(s.ctx, (int32_t) i);
            if (row == nullptr) {
                // A row that was marked as output but has no logits means the context is not the
                // tail shape this frame needs (embeddings_pre_norm builds no t_logits). Refuse
                // rather than answer a fabricated id: a wrong token here is a wrong token in the
                // caller's output, not a fallback.
                fprintf(stderr, "[split-stage] OP_DECODE verify frame: row %u has no logits; "
                                "this context does not build logits (a tail started with the "
                                "relay shape?)\n", i);
                llama_batch_free(batch);
                reply(c.peer, ST_INTERNAL, 0, 0, 0, nullptr, 0);
                return true;
            }
            s.verify_out[i] = argmax_row(row, s.n_vocab);
            trace_row(frame_id, "verify", n_tok, (int32_t) batch.pos[i], i, row, s.n_vocab,
                      s.verify_out[i], s.hidden.data() + (size_t) i * s.n_embd, s.n_embd);
        }
        const double t_argmax1 = g_timing ? now_ms() : 0.0;
        llama_batch_free(batch);
        if (!s.logged_verify) {
            s.logged_verify = true;
            fprintf(stderr, "[split-stage] first OP_DECODE verify frame: %u rows -> %u int32 "
                            "(%zu B) instead of one %u-float logits row (%zu B)\n",
                    n_tok, n_tok, s.verify_out.size() * sizeof(int32_t),
                    s.n_vocab, (size_t) s.n_vocab * sizeof(float));
        }
        // [QLH-SPLIT] v8: the body is the ids and then the h rows -- the order documented on
        // RSP_FLAG_NEXTN_ROWS. n_bytes covers both sections.
        //
        // [QLH-SPLIT] v9: with layer-input taps the body is ids, then the tap ids, then the tap rows;
        // h_nextn and taps never share a reply (one reply has one n_embd field, and a frame that asked
        // for both would be answered at one of the two widths).
        if (tap_flag) {
            reply_parts(c.peer, ST_OK, n_tok, tap_embd, 0, -1,
                        (uint32_t) (s.verify_out.size() * sizeof(int32_t) + tap_bytes),
                        (uint8_t) (RSP_FLAG_VERIFY_ROWS | tap_flag), tap_n,
                        s.verify_out.data(), s.verify_out.size() * sizeof(int32_t),
                        tap_ids_p, tap_ids_n, tap_rows_p, tap_rows_n);
        } else {
            reply_parts(c.peer, ST_OK, n_tok, nextn_embd, 0, -1,
                        (uint32_t) (s.verify_out.size() * sizeof(int32_t) + nextn_bytes),
                        (uint8_t) (RSP_FLAG_VERIFY_ROWS | nextn_flag), 0,
                        s.verify_out.data(), s.verify_out.size() * sizeof(int32_t),
                        nextn_data, nextn_bytes, nullptr, 0);
        }
        if (g_timing) {
            const double t_now = now_ms();
            s.timing.n_verify++;
            s.timing.verify_build   += t_build - t_frame;
            s.timing.verify_decode  += t_decode - t_build;
            s.timing.verify_argmax  += t_argmax1 - t_argmax0;
            // "post" is everything between the decode returning and the answer being sent: the
            // nextn rows, the argmax scan, the reply assembly. Kept separate from `argmax` so the
            // 248k-element scan can be reported on its own.
            s.timing.verify_post    += (t_now - t_decode) - (t_argmax1 - t_argmax0);
            s.timing.verify_reply   += now_ms() - t_now;
            if ((s.timing.n_verify & 15u) == 0) {
                timing_report(s.timing);
            }
        }
        return true;
    }

    // [QLH-SPLIT] v5: the candidate answer. Everything up to and including
    // top_k runs here, on the whole row; the driving side runs the rest of
    // its chain on these k entries. That split is exact because top_k is the
    // one step that truncates the vocabulary: after it, every sampler only
    // ever looks at cur_p, and cur_p holds exactly these k entries in
    // descending logit order (llama-sampler.cpp, llama_sampler_top_k_impl).
    // Applying the same steps to this array is therefore not an
    // approximation of the row path -- it is the row path, one hop later.
    if (want_cand) {
        // [QLH-SPLIT] v9: the candidate path answers a whole logits row too, so it is the same
        // "one row per token" shape as a plain decode -- traced as well, or the trace would have
        // exactly six lines (the prompt frames) and say nothing about the generation. The argmax
        // printed is the row's own, not cur_p.data[0]: a penalty step can move the top candidate,
        // and the point of the trace is "what would this row give", matched across paths.
        trace_row(frame_id, "cand", n_tok, (int32_t) batch.pos[last], (uint32_t) last, lg, s.n_vocab,
                  argmax_row(lg, s.n_vocab), s.hidden.data() + (size_t) last * s.n_embd, s.n_embd);
        llama_sampler * chain = build_cand_chain(creq, cbias, chist, s.n_vocab);
        if (chain == nullptr) {
            fprintf(stderr, "[split-stage] OP_DECODE candidate request named a "
                            "sampler step this build does not have; refusing "
                            "the frame instead of guessing a chain\n");
            llama_batch_free(batch);
            reply(c.peer, ST_BAD_REQ, 0, 0, 0, nullptr, 0);
            return false;
        }

        s.cand_cur.resize(s.n_vocab);
        for (uint32_t i = 0; i < s.n_vocab; ++i) {
            s.cand_cur[i] = llama_token_data{(llama_token) i, lg[i], 0.0f};
        }
        llama_token_data_array cur_p = { s.cand_cur.data(), s.cand_cur.size(), -1, false };
        llama_sampler_apply(chain, &cur_p);
        llama_sampler_free(chain);

        if (cur_p.size == 0 || cur_p.size > s.n_vocab) {
            llama_batch_free(batch);
            reply(c.peer, ST_INTERNAL, 0, 0, 0, nullptr, 0);
            return true;
        }

        // Sent as the pairs themselves: same order, same f32 logits, so the
        // driving side's array equals the one this stage just built.
        s.cand_out.resize(cur_p.size);
        for (size_t i = 0; i < cur_p.size; ++i) {
            s.cand_out[i].id    = (int32_t) cur_p.data[i].id;
            s.cand_out[i].logit = cur_p.data[i].logit;
        }
        llama_batch_free(batch);
        const double t_post = g_timing ? now_ms() : 0.0;
        if (tap_flag) {
            // [QLH-SPLIT] v9: the candidate pairs first, then the taps (a tap section is always last).
            reply_parts(c.peer, ST_OK, 1, tap_embd, 0, -1,
                        (uint32_t) (s.cand_out.size() * sizeof(token_cand) + tap_bytes),
                        (uint8_t) (RSP_FLAG_CANDIDATES | tap_flag), tap_n,
                        s.cand_out.data(), s.cand_out.size() * sizeof(token_cand),
                        tap_ids_p, tap_ids_n, tap_rows_p, tap_rows_n);
        } else {
            reply_parts(c.peer, ST_OK, 1, nextn_embd, 0, -1,
                        (uint32_t) (s.cand_out.size() * sizeof(token_cand) + nextn_bytes),
                        (uint8_t) (RSP_FLAG_CANDIDATES | nextn_flag), 0,
                        nextn_data, nextn_bytes,
                        s.cand_out.data(), s.cand_out.size() * sizeof(token_cand), nullptr, 0);
        }
        if (g_timing) {
            s.timing.n_cand++;
            s.timing.cand_build  += t_build - t_frame;
            s.timing.cand_decode += t_decode - t_build;
            s.timing.cand_post   += t_post - t_decode;
            s.timing.cand_reply  += now_ms() - t_post;
            if ((s.timing.n_cand & 15u) == 0) {
                timing_report(s.timing);
            }
        }
        return true;
    }

    // [QLH-SPLIT] No staging copy: reply() writes synchronously (net::send_all/send_two, both
    // blocking until the kernel has the bytes), so the row is still valid while it is sent.
    //
    // v9: the trace prints only the row the caller samples from -- for a plain decode that is the
    // last row, and for a prefill frame it is the last prompt row, which is exactly the row the
    // first generated token comes from. Printing every prefill row would bury the decode in one
    // line per prompt token and answer nothing.
    trace_row(frame_id, "plain", n_tok, (int32_t) batch.pos[last], (uint32_t) last, lg, s.n_vocab,
              argmax_row(lg, s.n_vocab), s.hidden.data() + (size_t) last * s.n_embd, s.n_embd);
    llama_batch_free(batch);
    const double t_post = g_timing ? now_ms() : 0.0;
    // [QLH-SPLIT] v9: taps before h_nextn rows on a plain frame, mirroring the verify and candidate
    // branches above. The two cannot share a reply (one reply states one n_embd width), and only the
    // tap is irreplaceable: a plain frame's h_nextn row is the last row of the batch, which a dflash
    // chain never reads, while a prefill frame's tap is every row of the batch and the draft KV has
    // no other way to get them. Ordering this the other way was why a remote dflash chain reported
    // "answered 0 of 2 layer-input taps" on every request and drafted from an unseeded KV: the head
    // asks for both (its tail reports a NextN block, so server-context arms the row channel too), the
    // stage answered the row, and the taps were never sent. MTP chains are unaffected: they ask for
    // no taps, so tap_flag stays 0 and they take the row branch.
    if (tap_flag) {
        // [QLH-SPLIT] v9: the logits row first (it is what the caller samples from), then the taps.
        // Both are sent as pieces, so the ~1 MB row is never copied to glue the tap block onto it.
        reply_parts(c.peer, ST_OK, 1, tap_embd, s.n_vocab, -1,
                    (uint32_t) (s.n_vocab * sizeof(float) + tap_bytes), tap_flag, tap_n,
                    lg, (size_t) s.n_vocab * sizeof(float),
                    tap_ids_p, tap_ids_n, tap_rows_p, tap_rows_n);
    } else if (want_nextn) {
        // Body order: the h rows, then the logits row (see RSP_FLAG_NEXTN_ROWS). The logits row is
        // ~1 MB, so it must not be copied to append 16 KB to it -- reply_parts sends the pieces.
        reply_parts(c.peer, ST_OK, 1, nextn_embd, s.n_vocab, -1,
                    (uint32_t) (nextn_bytes + s.n_vocab * sizeof(float)), nextn_flag, 0,
                    nextn_data, nextn_bytes, lg, (size_t) s.n_vocab * sizeof(float), nullptr, 0);
    } else {
        reply(c.peer, ST_OK, 1, 0, s.n_vocab, lg, (size_t) s.n_vocab * sizeof(float));
    }
    if (g_timing) {
        if (n_tok > (uint32_t) MAX_VERIFY_ROWS) {
            // A prompt frame. Its decode is 512 rows at once and has nothing to do with the
            // per-token number this report exists to give.
            s.timing.n_prefill++;
            s.timing.prefill_decode += t_decode - t_build;
        } else {
            s.timing.n_plain++;
            s.timing.plain_build  += t_build - t_frame;
            s.timing.plain_decode += t_decode - t_build;
            s.timing.plain_post   += t_post - t_decode;
            // The reply of a plain tail frame is the ~1 MB logits row: it is the one phase whose
            // cost depends on the link rather than on the machine, which is why it is timed apart.
            s.timing.plain_reply  += now_ms() - t_post;
            if ((s.timing.n_plain & 15u) == 0) {
                timing_report(s.timing);
            }
        }
    }
    return true;
}

static bool op_decode(conn & c, stage & s, const req_header & req) {
    const uint32_t n_tok = req.n_tokens;

    bool          want_cand = false;
    bool          want_verify = false;
    bool          want_nextn = false;
    cand_req      creq{};
    std::vector<llama_logit_bias> cbias;
    std::vector<llama_token>      chist;
    if (!decode_read_frame(c, s, req, n_tok, want_cand, creq, cbias, chist, want_verify,
                           want_nextn, s.tap_ask)) {
        return false;
    }

    // The whole frame is consumed at this point, so refusing does not desynchronize
    // the connection.
    if (!check_seq_id(c, s, req)) {
        return true;
    }
    // A frame larger than this stage's n_batch used to reach
    // llama_decode and trip GGML_ASSERT(n_tokens_all <= n_batch) -- an abort of
    // the stage process, which the caller sees as a dropped connection. The
    // driving server sends ubatches, so a stage started with a smaller -b than
    // the server's -ub is a configuration error; answer it as "out of space",
    // which the server already understands as a failed request.
    if (n_tok > (uint32_t) s.opt.n_batch) {
        fprintf(stderr, "[split-stage] OP_DECODE n_tok=%u exceeds this stage's -b %d; "
                        "-b must be at least the driving server's -ubatch-size\n",
                n_tok, s.opt.n_batch);
        reply(c.peer, ST_NO_SPACE, 0, 0, 0, nullptr, 0);
        return true;
    }
    return decode_run(c, s, req, n_tok, want_cand, creq, cbias, chist, want_verify, want_nextn);
}

// =====================================================================================
// [QLH-SPLIT] v8: OP_MTP_DRAFT -- run the MTP head this stage owns and answer k draft tokens.
//
// This is the one op in the protocol that looks like a *model feature* rather than a memory
// operation, and the reason is structural: the MTP block is a full transformer layer appended to
// the trunk, its input is t_h_nextn = RMSNorm(last trunk layer output), and in a split chain that
// last layer runs here. There is no head-side computation that could produce the same thing, so
// either the stage drafts or the chain does not (that is what v7 did, and why a remote chain could
// only use ngram).
//
// The k drafts cannot be batched: step i's input h is step i-1's output h_nextn, so the loop is
// sequential inside the stage. What *can* be one round trip is the whole block, and that is the
// point of the op -- a draft token per frame would add k hops to every speculative round.
//
// The loop is the local chain's, unchanged (common/speculative.cpp,
// common_speculative_impl_draft_mtp::draft, the is_chain branch):
//   step 0: decode (token at pos, h from the caller's row)      -> draft 1
//   step i: decode (draft i at pos+i, h_nextn of step i-1)      -> draft i+1
// Greedy, because the local draft sampler is top_k = 10 and it takes cur_p->data[0], i.e. the
// argmax, and the verify channel this feeds is only armed for a greedy chain.
//
// The draft KV is cleared per frame, exactly like the local chain's per-round clear: a round must
// not inherit rows from whatever the previous round left at the same positions (that is another
// token's state, and the attention would read it silently).
static bool op_mtp_draft(conn & c, stage & s, const req_header & req) {
    // A mtp_req that is not the expected size leaves the body length unknown, so the stream cannot
    // be resynchronised -- answer and drop the connection, like the other malformed frames.
    if (req.n_bytes != sizeof(mtp_req) || req.n_embd == 0 || req.n_embd > MAX_EMBD) {
        fprintf(stderr, "[split-stage] OP_MTP_DRAFT carried n_bytes=%u n_embd=%u, expected %zu "
                        "and a width in (0, %u]\n",
                req.n_bytes, req.n_embd, sizeof(mtp_req), MAX_EMBD);
        reply(c.peer, ST_BAD_REQ, 0, 0, 0, nullptr, 0);
        return false;
    }
    // The hidden row is read at *this* stage's width, so a frame from a head built for another
    // model is refused before its bytes are consumed (same rule as OP_DECODE's n_embd check).
    if (req.n_embd != s.n_embd) {
        fprintf(stderr, "[split-stage] OP_MTP_DRAFT n_embd=%u but this stage is %u wide; the two "
                        "ends were not built from the same model\n", req.n_embd, s.n_embd);
        reply(c.peer, ST_BAD_REQ, 0, 0, 0, nullptr, 0);
        return false;
    }

    mtp_req mreq{};
    if (!net::recv_all(c.peer, &mreq, sizeof(mreq))) {
        return false;
    }
    s.mtp_h.resize(s.n_embd);
    if (!net::recv_all(c.peer, s.mtp_h.data(), s.mtp_h.size() * sizeof(float))) {
        return false;
    }

    // The frame is consumed from here on: every refusal below answers and keeps serving.
    if (mreq.k == 0 || mreq.k > MAX_MTP_DRAFT) {
        fprintf(stderr, "[split-stage] OP_MTP_DRAFT asked for k=%u drafts, out of range [1, %u]\n",
                mreq.k, MAX_MTP_DRAFT);
        reply(c.peer, ST_BAD_REQ, 0, 0, 0, nullptr, 0);
        return true;
    }
    if (!check_seq_id(c, s, req)) {
        return true;
    }
    // The capability, not the frame: this stage is a perfectly good tail that simply has no draft
    // head (no NextN block in the model, or its MTP context failed to build). The head turns this
    // into "this request does not speculate" and keeps serving.
    if (!s.has_nextn || s.ctx_mtp == nullptr) {
        fprintf(stderr, "[split-stage] OP_MTP_DRAFT on a stage without an MTP draft context; "
                        "answering UNSUPPORTED\n");
        reply(c.peer, ST_UNSUPPORTED, 0, 0, 0, nullptr, 0);
        return true;
    }

    // One round is one clear: the draft KV is k+1 rows, nothing in it outlives the frame (see the
    // note above). This is the same call the local chain makes per round.
    llama_memory_clear(llama_get_memory(s.ctx_mtp), true);

    // [QLH-SPLIT] v9: what one draft round costs, split into "the whole round" and "the decodes in
    // it". It matters because each step is not cheap: it is a NextN layer *plus the full 248k
    // output projection*, so k drafts cost k lm_heads -- a real share of a speculative round that
    // the non-speculative path never pays.
    const double t_op  = g_timing ? now_ms() : 0.0;
    double       t_dec = 0.0;

    s.mtp_out.resize(mreq.k);
    llama_token tok = mreq.token;

    for (uint32_t i = 0; i < mreq.k; ++i) {
        llama_pos pos[1] = { mreq.pos + (int32_t) i };

        llama_batch_ext * b = llama_batch_ext_init(s.ctx_mtp);
        const int32_t idx = llama_batch_ext_add_token(b, req.seq_id, tok);
        if (idx < 0 ||
            !llama_batch_ext_set_pos(b, idx, pos) ||
            !llama_batch_ext_set_embd_state(b, idx, llama_embd{ s.mtp_h.data(), 1, s.n_embd }) ||
            !llama_batch_ext_set_output_logits(b, idx, true)) {
            fprintf(stderr, "[split-stage] OP_MTP_DRAFT could not build draft step %u\n", i);
            llama_batch_ext_free(b);
            reply(c.peer, ST_INTERNAL, 0, 0, 0, nullptr, 0);
            return true;
        }

        const double t_step = g_timing ? now_ms() : 0.0;
        const int32_t rc = llama_process(s.ctx_mtp, LLAMA_PROCESS_TYPE_DECODE, b);
        if (g_timing) {
            t_dec += now_ms() - t_step;
        }
        llama_batch_ext_free(b);
        if (rc != 0) {
            // The draft is an optimization: a failed step answers a refusal, it does not kill the
            // stage or the chain (the head already handles ST_* as "no drafts this round").
            fprintf(stderr, "[split-stage] OP_MTP_DRAFT draft step %u failed rc=%d (pos=%d)\n",
                    i, (int) rc, (int) pos[0]);
            reply(c.peer, rc == 1 ? ST_NO_SPACE : ST_INTERNAL, 0, 0, 0, nullptr, 0);
            return true;
        }

        const float * row = llama_get_logits_ith(s.ctx_mtp, idx);
        if (row == nullptr) {
            fprintf(stderr, "[split-stage] OP_MTP_DRAFT draft step %u produced no logits; this "
                            "context does not build the DECODER_MTP graph\n", i);
            reply(c.peer, ST_INTERNAL, 0, 0, 0, nullptr, 0);
            return true;
        }
        s.mtp_out[i] = argmax_row(row, s.n_vocab);
        trace_row(s.frame_id, "draft", mreq.k, mreq.pos + (int32_t) i, i, row, s.n_vocab,
                  s.mtp_out[i]);
        tok = s.mtp_out[i];

        // The next step's input h: this step's own t_h_nextn. Not needed after the last draft, so
        // the read is skipped there (it is a host copy, but there is no reason to make it).
        if (i + 1 < mreq.k) {
            const float * h = llama_get_embeddings_nextn_ith(s.ctx_mtp, idx);
            if (h == nullptr) {
                fprintf(stderr, "[split-stage] OP_MTP_DRAFT draft step %u produced no h_nextn; "
                                "the draft cannot chain past it\n", i);
                reply(c.peer, ST_INTERNAL, 0, 0, 0, nullptr, 0);
                return true;
            }
            memcpy(s.mtp_h.data(), h, s.n_embd * sizeof(float));
        }
    }

    if (!s.logged_mtp) {
        s.logged_mtp = true;
        fprintf(stderr, "[split-stage] first OP_MTP_DRAFT: seq=%d token=%d pos=%d -> %u draft(s) "
                        "[%d, %d, ...]; each step is one decode of the NextN block against its own "
                        "%u-cell KV\n",
                req.seq_id, mreq.token, mreq.pos, mreq.k,
                s.mtp_out.empty() ? -1 : s.mtp_out.front(),
                s.mtp_out.size() > 1 ? s.mtp_out[1] : -1,
                llama_n_ctx(s.ctx_mtp));
    }

    reply(c.peer, ST_OK, mreq.k, 0, 0, s.mtp_out.data(),
          s.mtp_out.size() * sizeof(int32_t), -1,
          (uint32_t) (s.mtp_out.size() * sizeof(int32_t)));
    if (g_timing) {
        s.timing.n_draft++;
        s.timing.draft_total  += now_ms() - t_op;
        s.timing.draft_decode += t_dec;
        if ((s.timing.n_draft & 15u) == 0) {
            timing_report(s.timing);
        }
    }
    return true;
}

static bool dispatch_frame(conn & c, stage & s, const req_header & req) {
    switch (req.op) {
        case OP_HELLO:      return op_hello(c, s, req);
        case OP_BYE:        return false;
        case OP_CLEAR:      return op_clear(c, s, req);
        case OP_SEQ_RM:     return op_seq_rm(c, s, req);
        case OP_STATE_SAVE: return op_state_save(c, s, req);
        case OP_STATE_LOAD: return op_state_load(c, s, req);
        case OP_DECODE:     return op_decode(c, s, req);
        case OP_MTP_DRAFT:  return op_mtp_draft(c, s, req);
        default:
            reply(c.peer, ST_BAD_REQ, 0, 0, 0, nullptr, 0);
            return false;
    }
}

// Serve one peer until it disconnects, sends BYE, or a frame says the stream is unusable.
static void serve_connection(sock_t peer, stage & s) {
    // [QLH-SPLIT] Nagle off on the accepted socket, not just on the client's side: Nagle
    // delays the *sender*, and this process is the sender of every reply. Without it any
    // small write that follows unacknowledged data waits for the peer's delayed ACK --
    // measured at 48-52 ms per token on the candidate reply (see reply()). The merged write
    // there fixes the frame shape this protocol actually sends today; this makes the
    // property hold for the next frame shape someone adds.
    //
    // Cost: the kernel stops batching small writes, so the 32 B responses (ST_OK with no
    // body) and the header of a large reply now leave as their own segments instead of
    // being merged with what follows. That is one extra tiny packet per frame at worst, and
    // it is paid on the same direction that was already idle waiting for an ACK.
    if (!net::set_nodelay(peer)) {
        fprintf(stderr, "[split-stage] warning: TCP_NODELAY failed on the accepted socket; "
                        "small replies may wait for a delayed ACK\n");
    }
    fprintf(stderr, "[split-stage] peer connected\n");

    conn c{ peer };

    bool alive = true;
    while (alive) {
        req_header req{};
        if (!net::recv_all(peer, &req, sizeof(req))) {
            break;
        }
        // one line per frame: the wire-level trace, only with --verbose
        vlog("[split-stage] req: magic=%08x ver=%u op=%u flags=%u n_tok=%u n_embd=%u seq=%d\n",
                req.magic, req.version, (unsigned) req.op, (unsigned) req.flags,
                req.n_tokens, req.n_embd, req.seq_id);
        if (!check_frame(c, req)) {
            break;
        }
        alive = dispatch_frame(c, s, req);
    }

    net::close_sock(peer);
    fprintf(stderr, "[split-stage] peer disconnected\n");
}

int main(int argc, char ** argv) {
    stage_options opt;
    if (parse_args(argc, argv, opt) != 0) {
        return 1;
    }
    if (validate_options(opt) != 0) {
        return 1;
    }

    // [QLH-SPLIT] v9: the two diagnosis knobs. Read once, here, so every frame after this pays a
    // predictable branch and no getenv(). They print to stderr and change no behaviour: whatever
    // row a stage answers is the same row it would answer without them.
    g_trace_argmax = env_flag("QLH_STAGE_TRACE_ARGMAX");
    g_timing       = env_flag("QLH_STAGE_TIMING");
    if (g_trace_argmax || g_timing) {
        // A diagnosis log has to survive the way these stages are stopped: taskkill /F, which
        // discards whatever the C runtime still holds. The last frames are the interesting ones
        // -- they are where a request ended -- so the stream is made unbuffered while diagnosing.
        setvbuf(stderr, nullptr, _IONBF, 0);
        fprintf(stderr, "[split-stage] diagnosis: trace_argmax=%d timing=%d (environment; these "
                        "only add log lines and do not change any answer)\n",
                (int) g_trace_argmax, (int) g_timing);
    }

    // [QLH-SPLIT] v10: bind the port BEFORE the model is loaded. The order used to be the other
    // way round, and a port that was already taken (a stale stage from the last run, a second
    // copy of this command) was only discovered after a multi-GiB load: tens of seconds of
    // "starting up" that the control plane cannot tell from "loading slowly", then a bind error
    // and an exit. The bind is the cheap precondition that can be checked first, so it goes
    // first: a conflict now fails in the first second, before anything is read from disk.
    sock_t listener = net::invalid;
    if (!listen_open(opt, listener)) {
        return 1;
    }
    // From here on the port is ours, and every failure path has to give it back. A listener that
    // outlives a failed load is worse than a failed bind: the port accepts connections and then
    // no serve loop ever runs, so the peer blocks waiting for an answer instead of learning that
    // this stage is down. This line is the "port is ours" signal, not "ready to serve".
    fprintf(stderr, "[split-stage] listening on %s:%d (model not loaded yet)\n",
            opt.host.c_str(), opt.port);

    stage s;
    if (!stage_init(opt, s)) {
        net::close_sock(listener);
        stage_free(s);
        return 1;
    }

    for (;;) {
        sock_t peer = accept(listener, nullptr, nullptr);
        if (!net::valid(peer)) {
            break;
        }
        serve_connection(peer, s);
    }

    net::close_sock(listener);
    stage_free(s);
    return 0;
}
