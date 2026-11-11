#include "speculative.h"

#include "common.h"
#include "ggml.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "log.h"
#include "ngram-cache.h"
#include "ngram-map.h"
#include "ngram-mod.h"
#include "sampling.h"

#include "../src/llama-ext.h" // staging API: llama_set_embeddings_nextn / llama_get_embeddings_nextn_ith (used by MTP)

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <map>
#include <random>
#include <cinttypes>
#include <cstdlib>

#define SPC_DBG(fmt, ...) LOG_DBG("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_TRC(fmt, ...) LOG_TRC("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_INF(fmt, ...) LOG_INF("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_WRN(fmt, ...) LOG_WRN("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_ERR(fmt, ...) LOG_ERR("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_CNT(fmt, ...) LOG_CNT(""              fmt,               __VA_ARGS__)

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

// Rebuild seq_id's draft sampler at the target's temperature: rejection weighs q against p, so
// both have to sample alike. Only temp and seed carry over; the draft keeps its own top_k.
static void spec_retune(
        std::vector<common_sampler_ptr> & smpls,
        std::vector<common_params_sampling> & cfg,
        const llama_model * model,
        llama_seq_id seq_id,
        float temp,
        uint32_t seed) {
    if (cfg.size() != smpls.size()) {
        const size_t n_old = cfg.size();
        cfg.resize(smpls.size());

        // the initial sampler has no temperature, so no request may match the cache and skip a rebuild
        for (size_t i = n_old; i < cfg.size(); ++i) {
            cfg[i].temp = NAN;
        }
    }

    auto & cur = cfg[seq_id];

    if (cur.temp == temp && cur.seed == seed) {
        return;
    }

    cur.temp = temp;
    cur.seed = seed;

    common_params_sampling sparams;
    sparams.no_perf  = false;
    sparams.top_k    = 10;
    sparams.temp     = cur.temp;
    // must be explicit, the default reseeds at random; mixed so it differs from the target's
    sparams.seed     = cur.seed == LLAMA_DEFAULT_SEED ? cur.seed : cur.seed ^ 0x85ebca6bu;
    sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K, COMMON_SAMPLER_TYPE_TEMPERATURE };

    smpls[seq_id].reset(common_sampler_init(model, sparams));
}

const std::map<std::string, common_speculative_type> common_speculative_type_from_name_map = {
    {"none",          COMMON_SPECULATIVE_TYPE_NONE},
    {"draft-simple",  COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE},
    {"draft-eagle3",  COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3},
    {"draft-mtp",     COMMON_SPECULATIVE_TYPE_DRAFT_MTP},
    {"draft-dflash",  COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH},
    {"draft-dspark",  COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK},
    {"ngram-simple",  COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE},
    {"ngram-map-k",   COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K},
    {"ngram-map-k4v", COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V},
    {"ngram-mod",     COMMON_SPECULATIVE_TYPE_NGRAM_MOD},
    {"ngram-cache",   COMMON_SPECULATIVE_TYPE_NGRAM_CACHE}
};

static std::string common_speculative_get_devices_str(const std::vector<ggml_backend_dev_t> & devices) {
    std::string result;
    for (size_t i = 0; i < devices.size(); i++) {
        if (devices[i] == nullptr) {
            continue;
        }
        if (!result.empty()) result += ", ";
        result += ggml_backend_dev_name(devices[i]);
    }
    return result.empty() ? "default" : result;
}

struct common_speculative_config {
    common_speculative_type type;
    common_params_speculative params;

    common_speculative_config(common_speculative_type t,
            const common_params_speculative & p = common_params_speculative{}) : type(t), params(p) {}
};

bool common_speculative_are_compatible(
    const llama_model * model_tgt,
    const llama_model * model_dft) {
    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    const auto vocab_type_tgt = llama_vocab_type(vocab_tgt);
    SPC_DBG("vocab_type tgt: %d\n", vocab_type_tgt);

    const auto vocab_type_dft = llama_vocab_type(vocab_dft);
    SPC_DBG("vocab_type dft: %d\n", vocab_type_dft);

    if (vocab_type_tgt != vocab_type_dft) {
        SPC_WRN("draft model vocab type must match target model to use speculation but "
                "vocab_type_dft = %d while vocab_type_tgt = %d\n", vocab_type_dft, vocab_type_tgt);
        return false;
    }

    if (llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) ||
        (llama_vocab_get_add_bos(vocab_tgt) && llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft))) {
        SPC_WRN("draft model bos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                llama_vocab_get_add_bos(vocab_tgt), llama_vocab_get_add_bos(vocab_dft),
                llama_vocab_bos(vocab_tgt), llama_vocab_bos(vocab_dft));
        return false;
    }

    if (llama_vocab_get_add_eos(vocab_tgt) != llama_vocab_get_add_eos(vocab_dft) ||
        (llama_vocab_get_add_eos(vocab_tgt) && llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft))) {
        SPC_WRN("draft model eos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                llama_vocab_get_add_eos(vocab_tgt), llama_vocab_get_add_eos(vocab_dft),
                llama_vocab_eos(vocab_tgt), llama_vocab_eos(vocab_dft));
        return false;
    }

    {
        const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
        const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
        const int vocab_diff  = n_vocab_tgt > n_vocab_dft
            ? n_vocab_tgt - n_vocab_dft
            : n_vocab_dft - n_vocab_tgt;

        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            SPC_DBG("draft model vocab must closely match target model to use speculation but "
                    "target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_tgt, llama_vocab_n_tokens(vocab_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return false;
        }

        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            const char * token_text_tgt = llama_vocab_get_text(vocab_tgt, i);
            const char * token_text_dft = llama_vocab_get_text(vocab_dft, i);

            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                SPC_DBG("draft model vocab must match target model to use speculation but "
                        "token %d content differs - target '%s', draft '%s'\n", i,
                        common_token_to_piece(vocab_tgt, i).c_str(),
                        common_token_to_piece(vocab_dft, i).c_str());
                return false;
            }
        }
    }

    return true;
}

using common_speculative_draft_params_vec = std::vector<common_speculative_draft_params>;

// state of an implementation of speculative decoding
//
// each implementation has a unique type and a state that is implementation-specific
// in a subclass of common_speculative_impl
struct common_speculative_impl {
    const common_speculative_type type;

    uint32_t n_seq;
    int32_t n_max; // maximum draft length after implementation-specific limits

    size_t n_call_begin  = 0; // number of times this implementation was called for refresh.
    size_t n_call_draft  = 0; // number of times this implementation was called for generation.
    size_t n_call_accept = 0; // number of times this implementation was called for accumulation.

    size_t n_gen_drafts = 0; // number of times a draft or part was generated by this implementation.
    size_t n_acc_drafts = 0; // number of times a draft or part was accepted by the target model.
    size_t n_gen_tokens = 0; // number of tokens generated by this implementation.
    size_t n_acc_tokens = 0; // number of tokens accepted by the target model.

    std::vector<size_t> n_acc_tokens_per_pos; // number of tokens accepted per draft position.

    // TODO: track performance of most recent calls
    const bool gen_perf = true; // whether to generate performance stats.

    int64_t t_begin_us  = 0; // total time spent in refresh of this implementation in microseconds.
    int64_t t_draft_us  = 0; // total time spent in generating drafts in this implementation in microseconds.
    int64_t t_accept_us = 0; // total time spent in accumulation of this implementation in microseconds.

    common_speculative_impl(common_speculative_type type, uint32_t n_seq, int32_t n_max) : type(type), n_seq(n_seq), n_max(n_max) {}

    virtual ~common_speculative_impl() = default;

    virtual void begin(llama_seq_id seq_id, const llama_tokens & prompt) = 0;

    virtual bool process(const common_batch & batch) = 0;

    virtual void draft(common_speculative_draft_params_vec & dparams) = 0;

    virtual void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) = 0;

    // (optional) serialize/restore per-seq internal state (e.g. eagle3's deferred boundary).
    virtual bool get_state(llama_seq_id /*seq_id*/, std::vector<uint8_t> & /*data*/) const { return false; }
    virtual void set_state(llama_seq_id /*seq_id*/, const std::vector<uint8_t> & /*data*/) {}

    // [QLH-SPLIT] true when this implementation keeps its own draft KV across rounds and trims it
    // itself at the start of each draft(). The server trims the draft KV after every draft() to
    // drop the rows of the round that was just made; a keep-KV implementation must not be trimmed
    // there, or the history it exists to build is dropped on the very round that created it.
    virtual bool keeps_draft_kv() const { return false; }

    // [QLH-SPLIT] one-line report of how the keep-KV route actually ran (empty when unused)
    virtual std::string keep_draft_kv_stats() const { return {}; }
};

struct common_speculative_impl_draft_simple : public common_speculative_impl {
    common_params_speculative_draft params;

    common_batch batch;

    // zero row at the draft input width, stands in for target embeddings the draft cannot read
    std::vector<float> zeros;
    bool zeros_warned = false; // the substitution is reported once

    std::vector<common_sampler_ptr> smpls;

    std::vector<common_params_sampling> smpls_cfg;

    common_speculative_impl_draft_simple(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE, n_seq, params.draft.n_max)
        , params(params.draft)
    {
        auto * ctx_dft = this->params.ctx_dft;
        auto * ctx_tgt = this->params.ctx_tgt;

        if (!ctx_dft) {
            throw std::runtime_error("draft-simple requires a draft context");
        }

        zeros.assign(llama_model_n_embd_inp(llama_get_model(ctx_dft)), 0.0f);

        SPC_TRC("%s", "adding speculative implementation 'draft-simple'\n");
        SPC_TRC("- n_max=%d, n_min=%d, p_min=%f\n", this->params.n_max, this->params.n_min, this->params.p_min);
        SPC_TRC("- gpu_layers=%d, cache_k=%s, cache_v=%s, ctx_tgt=%s, ctx_dft=%s, devices=[%s]\n",
                this->params.n_gpu_layers,
                ggml_type_name(this->params.cache_type_k),
                ggml_type_name(this->params.cache_type_v),
                ctx_tgt ? "yes" : "no",
                ctx_dft ? "yes" : "no",
                common_speculative_get_devices_str(this->params.devices).c_str());

        batch = common_batch(ctx_dft);

        // TODO: optimize or pass from outside?
        // {
        //     common_params_sampling params;
        //     params.no_perf = false;
        //
        //     params.top_k = 40;
        //     params.top_p = 0.9;
        //
        //     params.samplers = {
        //         COMMON_SAMPLER_TYPE_TOP_K,
        //         COMMON_SAMPLER_TYPE_TOP_P,
        //         COMMON_SAMPLER_TYPE_INFILL,
        //     };
        //
        //     result->smpl = common_sampler_init(llama_get_model(ctx_dft), params);
        // }

        smpls.resize(n_seq);
        for (auto & smpl : smpls) {
            common_params_sampling params;
            params.no_perf = false;
            params.top_k = 10;
            params.samplers.assign(1, COMMON_SAMPLER_TYPE_TOP_K);

            smpl.reset(common_sampler_init(llama_get_model(ctx_dft), params));
        }

        const bool vocab_cmpt = common_speculative_are_compatible(llama_get_model(ctx_tgt), llama_get_model(ctx_dft));
        SPC_DBG("vocab_cmpt = %d\n", vocab_cmpt);

        if (!vocab_cmpt) {
            SPC_ERR("%s", "the target and draft vocabs are not compatible\n");

            throw std::runtime_error("draft model vocab type must match target model to use speculation");
        }

        if (n_seq != llama_n_seq_max(ctx_dft)) {
            SPC_ERR("n_seq mismatch: %d != %d\n", n_seq, llama_n_seq_max(ctx_dft));

            throw std::runtime_error("the draft model number of sequences is incompatible with the speculative n_seq");
        }
    }

    void begin(llama_seq_id seq_id, const llama_tokens & /*prompt*/) override {
        // reset here rather than per round, or two identical requests differ
        common_sampler_reset(smpls[seq_id].get());
    }

    bool process(const common_batch & batch_in) override {
        auto * ctx_dft = params.ctx_dft;

        // copy the entries to a batch owned by the draft context, only the last token is output
        batch.clear();
        const int32_t n_tokens = batch_in.size();
        for (int32_t k = 0; k < n_tokens; ++k) {
            const auto & t = batch_in.tokens[k];
            const bool output = k == n_tokens - 1;
            if (t.id != LLAMA_TOKEN_NULL) {
                const int32_t idx = batch.add(t.id, t.pos[0], t.seq_id, output);
                if (t.embd.data) {
                    batch.set_embd(idx, t.embd);
                }
            } else {
                // mtmd input is projected by the target encoder, a draft with a different width cannot read it
                // it gets zeros instead, keeping its positions contiguous
                // ref: https://github.com/ggml-org/llama.cpp/pull/29385#discussion_r4124743243
                const size_t n_embd = t.embd.n_rows * t.embd.n_embd;
                const bool same_width = n_embd == zeros.size();
                if (!same_width && !zeros_warned) {
                    SPC_WRN("target embeddings of size %zu do not fit the draft input width %zu, "
                            "the draft receives zero rows for them and drafts after multimodal input will be poor\n",
                            n_embd, zeros.size());
                    zeros_warned = true;
                }
                const llama_embd embd = same_width ? t.embd : llama_embd{ zeros.data(), 1, zeros.size() };
                batch.add_embd(embd, t.pos.data(), t.seq_id, output);
            }
        }

        if (batch.size() == 0) {
            return true;
        }

        const int ret = llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch.get());

        if (ret != 0) {
            SPC_ERR("failed to decode draft batch, ret = %d\n", ret);

            return false;
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        batch.clear();

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }

            n_drafting++;
            drafting[seq_id] = true;
            // greedy drafting leaves no candidates behind, so the verifier falls back to sample-and-match
            if (!params.probabilistic) {
                dp.result_q = nullptr;
            }

            // result_q is only set when the caller wants rejection, so it also gates the retune
            if (dp.result_q) {
                spec_retune(smpls, smpls_cfg, llama_get_model(ctx_dft), seq_id, dp.temp, dp.seed);
            }

            // a reset reseeds the chain, which breaks probabilistic drafting
            if (!dp.result_q) {
                common_sampler_reset(smpls[seq_id].get());
            }

            batch.add(dp.id_last, dp.pos0, seq_id, true);
        }

        int ret = llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch.get());
        if (ret != 0) {
            SPC_ERR("llama_process returned %d\n", ret);
            return;
        }

        int i = 0;

        while (n_drafting > 0) {
            int i_batch = 0;

            batch.clear();

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                const llama_token id_sampled = common_sampler_sample(smpl, ctx_dft, i_batch, true);
                ++i_batch;

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    SPC_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                // add drafted token for each sequence
                const llama_token id = dparams.at(seq_id).result_q ? id_sampled : cur_p->data[0].id;

                // only collect very high-confidence draft tokens
                if (cur_p->data[0].p < params.p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                result.push_back(id);

                if (dp.result_q) {
                    dp.result_q->emplace_back(cur_p->data, cur_p->data + cur_p->size);
                }

                if ((params.n_max <= (int) result.size()) ||
                    (dp.n_max > 0 && dp.n_max <= (int) result.size())) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                batch.add(id, dp.pos0 + i + 1, seq_id, true);
            }

            if (batch.size() == 0) {
                break;
            }

            // evaluate the drafted tokens on the draft model
            ret = llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch.get());
            if (ret != 0) {
                SPC_ERR("llama_process[%d] returned %d\n", i, ret);
                break;
            }

            ++i;
        }

        for (auto & dp : dparams) {
            if (!dp.drafting) {
                continue;
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }
};


// EAGLE3 speculative decoding state
//
// Input of draft decoder: (This is different compared to MTP)
//   At "pos P", the decoder takes input pair (t_{P+1}, g_P), with RoPE at P.
//     - t_{P+1} = token at sequence pos P+1 (the *next* token after P)
//     - g_P     = encoder output = projection of target's extracted hidden states at P
//
// Deferred boundary (MTP doesn't have this issue):
//   Within a single process() call with n_tokens, we can only write decoder KV for
//   training pos 0..n_tokens-2. The last training pos (n_tokens-1) needs t_{n_tokens}
//   which lies *outside* this batch — it is the token target will sample next or the first token from next ubatch.
//   So the last training pos of each process() call is *deferred* to whichever next call has
//   the missing token in hand:
//     - multi-ubatch prefill: the next process()'s first token completes the pair
//                              (handled by the per-seq "cross-ubatch bridge")
//     - single-ubatch prefill / after verify: draft()'s seed step uses "dp.id_last"
//                              (target's freshest sample) to complete the pair
//
// Per-seq carry-over state:
//   pending_g_last    [n_embd_dec]  ┐  the deferred boundary's (g, pos). Set by
//   pending_pos_last  llama_pos     ┘  process() at end of ubatch (= last row);
//                                       rebased by accept() to first-non-accepted pos.
//   verify_g          [N × n_embd_dec] snapshot of process()'s encoder output;
//   verify_pos_first  llama_pos         consumed by accept() to recover the right
//   verify_g_rows     int32_t           pending_g_last row for any n_accepted value.
//
// Performance is overall good but there is waste in verify cycle:
//   process() runs encoder + decoder on the *full* verify batch including rows for
//   rejected drafts. The KV at those positions is then dropped.
//
// TODO: Not sure if we need optimization for this waste?
// If so we may need hybrid stash:
//      in verify mode, have process() only stash features and let draft() seed run
//      encoder+decoder on n_accepted+1 rows).
struct common_speculative_impl_draft_eagle3 : public common_speculative_impl {
    common_params_speculative_draft params;
    common_batch batch;     // decoder input, (token, g_embd) pairs
    common_batch batch_enc; // encoder input, built from the extracted target features

    std::vector<common_sampler_ptr> smpls;

    // backend sampler chain per seq, attached to ctx_dft
    std::vector<llama_sampler *> backend_chains;

    int32_t n_embd_dec = 0;       // draft hidden size
    int32_t n_embd_enc = 0;       // target_layer_ids_n * target_hidden_size
    int32_t n_embd_tgt = 0;       // target model hidden size
    int32_t n_layer_tgt = 0;      // target model layer count

    const int32_t * target_layer_ids   = nullptr; // model_dft's extract layer indices
    uint32_t        target_layer_ids_n = 0;

    // [per-seq] deferred boundary state
    std::vector<std::vector<float>> pending_g_last;
    std::vector<llama_pos>          pending_pos_last;

    // [per-seq] snapshot of the most recent process()'s encoder output
    std::vector<std::vector<float>> verify_g;         // [n_seq][n_rows * n_embd_dec]
    std::vector<llama_pos>          verify_pos_first; // [n_seq] — pos of verify_g[seq][0]
    std::vector<int32_t>            verify_g_rows;    // [n_seq] — number of rows

    // scratch buffer for concatenated target features [n_tokens, n_embd_enc]
    std::vector<float> features_buf;
    std::vector<float> g_embd_buf;

    common_speculative_impl_draft_eagle3(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3, n_seq, params.draft.n_max)
        , params(params.draft)
    {
        SPC_TRC("%s", "adding speculative implementation 'draft-eagle3'\n");
        SPC_TRC("- n_max=%d, n_min=%d, p_min=%f, backend_sampling=%d\n", params.draft.n_max, params.draft.n_min, params.draft.p_min, (int) params.draft.backend_sampling);

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;
        GGML_ASSERT(ctx_tgt && ctx_dft && "EAGLE3 requires ctx_tgt and ctx_dft to be set");

        const llama_model * model_dft = llama_get_model(ctx_dft);
        const llama_model * model_tgt = llama_get_model(ctx_tgt);

        target_layer_ids   = llama_model_target_layer_ids  (model_dft);
        target_layer_ids_n = llama_model_target_layer_ids_n(model_dft);
        if (target_layer_ids_n != 3) {
            throw std::runtime_error("draft model is not eagle3 (expected 3 extract layers, got " +
                                     std::to_string(target_layer_ids_n) + ")");
        }

        n_embd_tgt = llama_model_n_embd(model_tgt);
        n_embd_dec = llama_model_n_embd(model_dft);
        n_embd_enc = (int32_t) target_layer_ids_n * n_embd_tgt;
        n_layer_tgt = llama_model_n_layer(model_tgt);

        batch     = common_batch(ctx_dft);
        batch_enc = common_batch(ctx_dft);

        smpls.resize(n_seq);
        for (auto & s : smpls) {
            common_params_sampling sparams;
            sparams.no_perf  = false;
            sparams.top_k    = 10;
            sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
            s.reset(common_sampler_init(llama_get_model(ctx_dft), sparams));
        }

        // offload draft sampling to the backend
        backend_chains.assign(n_seq, nullptr);
        if (this->params.backend_sampling) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
                llama_sampler_chain_add(chain, llama_sampler_init_top_k(10));

                if (!llama_set_sampler(ctx_dft, seq_id, chain)) {
                    SPC_WRN("backend offload failed for seq_id=%d; using CPU sampler\n", (int) seq_id);
                    llama_sampler_free(chain);
                    chain = nullptr;
                }
                backend_chains[seq_id] = chain;
            }
        }

        // turn on extraction of the target layers' hidden states
        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            if (target_layer_ids[k] < n_layer_tgt) {
                llama_set_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k], true);
            } else if (target_layer_ids[k] == n_layer_tgt) {
                llama_set_embeddings_nextn(ctx_tgt, true, /*masked*/ false);
            } else {
                GGML_ABORT("EAGLE3: target layer id %d exceeds target n_layer %d", target_layer_ids[k], n_layer_tgt);
            }
        }

        // turn on extraction of the draft model's pre-norm hidden state
        // (used both for the encoder output g_embd and the decoder pre-norm output).
        llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ true);

        pending_g_last.assign(n_seq, std::vector<float>(n_embd_dec, 0.0f));
        pending_pos_last.assign(n_seq, -1);

        verify_g.assign(n_seq, std::vector<float>());
        verify_pos_first.assign(n_seq, -1);
        verify_g_rows.assign(n_seq, 0);
    }

    ~common_speculative_impl_draft_eagle3() override {
        auto * ctx_dft = this->params.ctx_dft;
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) backend_chains.size(); ++seq_id) {
            if (backend_chains[seq_id] == nullptr) {
                continue;
            }
            if (ctx_dft) {
                llama_set_sampler(ctx_dft, seq_id, nullptr);
            }
            llama_sampler_free(backend_chains[seq_id]);
        }
        backend_chains.clear();
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }
        // expected state after prefill: ctx_dft has pos 0..N-2 (last position is deferred to
        // draft()'s seed step). Warn only if more than one position is missing.
        auto * ctx_dft = this->params.ctx_dft;
        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);
        if (pos_max < N - 2) {
            SPC_WRN("ctx_dft pos_max=%d < N-2=%d — process() did not run on every prefill ubatch. "
                    "Drafts may degrade.\n",
                    (int) pos_max, N - 2);
        }
    }

    bool process(const common_batch & batch_in) override {
        if (batch_in.size() <= 0) {
            return true;
        }

        if (!batch_in.has_token() || batch_in.has_embd()) {
            return true;
        }

        const int32_t n_tokens = batch_in.size();

        // i_batch_beg[seq] / i_batch_end[seq]: inclusive batch indices of this seq's
        // first/last token in batch_in. Assumes per-seq tokens are contiguous within
        // the ubatch (server's default ordering).
        std::vector<int32_t> i_batch_beg(n_seq, -1);
        std::vector<int32_t> i_batch_end(n_seq, -1);
        for (int k = 0; k < n_tokens; ++k) {
            const llama_seq_id seq_id = batch_in.tokens[k].seq_id;
            if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
                continue;
            }
            i_batch_end[seq_id] = k;
            if (i_batch_beg[seq_id] < 0) {
                i_batch_beg[seq_id] = k;
            }
        }

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;

        // Interleave each extract_layer's hidden state into a contiguous buffer of
        // shape [n_tokens, target_layer_ids_n * n_embd_tgt]. Then run EAGLE3 encoder
        // to get one g_embd row per token.
        features_buf.resize((size_t) n_tokens * n_embd_enc, 0.0f);

        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            const float * layer = target_layer_ids[k] < n_layer_tgt
                ? llama_get_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k])
                : llama_get_embeddings_nextn(ctx_tgt);
            if (!layer) {
                GGML_ABORT("EAGLE3: target layer %d input not extracted.", target_layer_ids[k]);
            }
            for (int32_t i = 0; i < n_tokens; ++i) {
                float * dst = features_buf.data() + (size_t) i * n_embd_enc + k * (size_t) n_embd_tgt;
                const float * src = layer + (size_t) i * n_embd_tgt;
                std::memcpy(dst, src, (size_t) n_embd_tgt * sizeof(float));
            }
        }

        g_embd_buf.resize((size_t) n_tokens * n_embd_dec);

        // llama_process() requires the full encoder batch to fit in n_ubatch.
        // Allow batch > ubatch: eagle3's per-token encoder can be chunked safely.
        const int32_t n_ubatch_dft = (int32_t) llama_n_ubatch(ctx_dft);
        for (int32_t i = 0; i < n_tokens; i += n_ubatch_dft) {
            const int32_t n_chunk = std::min(n_ubatch_dft, n_tokens - i);

            // the per-token encoder does not use positions, generate placeholder ones from the memory state
            batch_enc.clear();
            llama_pos pos = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), 0) + 1;
            for (int32_t j = 0; j < n_chunk; ++j) {
                batch_enc.add_embd({ features_buf.data() + (size_t) (i + j) * n_embd_enc, 1, (size_t) n_embd_enc }, &pos, 0, true);
                pos++;
            }

            const int32_t rc = llama_process(ctx_dft, LLAMA_PROCESS_TYPE_ENCODE, batch_enc.get());
            if (rc != 0) {
                SPC_ERR("llama_process(ctx_dft) failed rc=%d (n_tokens=%d, offset=%d)\n",
                        rc, (int) n_chunk, (int) i);
                return false;
            }

            // g_embd has shape [n_chunk, n_embd_dec] in ctx_dft's pre-norm embeddings buffer.
            const float * g_embd_chunk = llama_get_embeddings_nextn(ctx_dft);
            GGML_ASSERT(g_embd_chunk && "EAGLE3 encoder produced no output.");
            std::memcpy(g_embd_buf.data() + (size_t) i * n_embd_dec,
                        g_embd_chunk,
                        (size_t) n_chunk * n_embd_dec * sizeof(float));
        }

        const float * g_embd = g_embd_buf.data();

        const size_t row_bytes = (size_t) n_embd_dec * sizeof(float);

        // EAGLE3 decoder input convention: at memory pos P the input pair is
        // (token[P+1], g_embd[P]). This shifts the token index "left by one" relative to g_embd.
        //
        // Per seq, in order:
        //   (a) cross-ubatch bridge — when applicable, write the previously-deferred
        //       pos using this ubatch's first token + pending_g_last.
        //   (b) main write loop — for k in [beg, end-1], write (token[k+1], g_embd[k])
        //       at pos[k]. The last training pos (k=end) is left unwritten = new
        //       deferred boundary, completed by the next process() or draft() call.
        //   (c) refresh deferred state — stash this ubatch's full g_embd into verify_g,
        //       update pending_g_last / pending_pos_last to the last row.
        batch.clear();

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            const int32_t beg = i_batch_beg[seq_id];
            const int32_t end = i_batch_end[seq_id];
            if (beg < 0 || end < 0) {
                continue;
            }

            // cross-ubatch bridge — complete the prior ubatch's deferred boundary.
            // Fires iff all three preconditions hold:
            //   1) pending_pos_last >= 0
            //   2) pending_pos_last + 1 == pos[beg]
            //   3) pending_pos_last > dft_pos_max // TODO: is this check needed?
            const llama_pos pending_pos = pending_pos_last[seq_id];
            if (pending_pos >= 0 && pending_pos + 1 == batch_in.tokens[beg].pos[0]) {
                const llama_pos dft_pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);
                if (pending_pos > dft_pos_max) {
                    const int32_t idx = batch.add(batch_in.tokens[beg].id, pending_pos, seq_id, /*output=*/ false);
                    batch.set_embd(idx, { pending_g_last[seq_id].data(), 1, (size_t) n_embd_dec });
                }
            }

            for (int32_t k = beg; k < end; ++k) {
                const int32_t idx = batch.add(batch_in.tokens[k + 1].id, batch_in.tokens[k].pos[0], seq_id, /*output=*/ false);
                batch.set_embd(idx, { g_embd + (size_t) k * n_embd_dec, 1, (size_t) n_embd_dec });
            }

            // refresh deferred state
            const int32_t n_rows = end - beg + 1;
            verify_pos_first[seq_id] = batch_in.tokens[beg].pos[0];
            pending_pos_last[seq_id] = batch_in.tokens[end].pos[0];
            verify_g_rows[seq_id]    = n_rows;
            verify_g[seq_id].resize((size_t) n_rows * n_embd_dec, 0.0f);
            std::memcpy(verify_g[seq_id].data(),       g_embd + (size_t) beg * n_embd_dec, row_bytes * n_rows);
            std::memcpy(pending_g_last[seq_id].data(), g_embd + (size_t) end * n_embd_dec, row_bytes);
        }

        if (batch.size() > 0) {
            const int32_t rc = llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch.get());
            if (rc != 0) {
                SPC_ERR("llama_process(ctx_dft) failed rc=%d (n_tokens=%d, ubatch_pos[0]=%d)\n",
                        rc, (int) batch.size(), (int) batch_in.tokens[0].pos[0]);
                return false;
            }
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        batch.clear();

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        // Complete the deferred boundary pair (dp.id_last, pending_g_last) at memory
        // pos pending_pos_last. dp.id_last is target's freshest sample (= corrected
        // token after verify, or first generated token after prefill), matching the
        // EAGLE3 input convention (token[P+1], g_embd[P]) at pos P.
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }
            if (pending_pos_last[seq_id] < 0) {
                continue;
            }

            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());

            llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, pending_pos_last[seq_id], -1);

            const int32_t idx = batch.add(dp.id_last, pending_pos_last[seq_id], seq_id, true);
            batch.set_embd(idx, { pending_g_last[seq_id].data(), 1, (size_t) n_embd_dec });
        }

        if (batch.size() == 0) {
            return;
        }

        int ret = llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch.get());
        if (ret != 0) {
            SPC_ERR("llama_process returned %d\n", ret);
            return;
        }

        int i = 0;

        while (n_drafting > 0) {
            int i_batch = 0;

            batch.clear();

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                common_sampler_sample(smpl, ctx_dft, i_batch, true);
                // pre-norm hidden state of this position becomes g_embd for the next step
                const float * prenorm = llama_get_embeddings_nextn_ith(ctx_dft, i_batch);
                ++i_batch;

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    SPC_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                const llama_token id = cur_p->data[0].id;

                // only collect very high-confidence draft tokens
                // (configurable via --spec-draft-p-min, set to 0.0 to disable early-stop)
                if (cur_p->data[0].p < params.p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                result.push_back(id);

                if (params.n_max <= (int) result.size()) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                const int32_t idx = batch.add(id, pending_pos_last[seq_id] + (i + 1), seq_id, true);
                batch.set_embd(idx, { prenorm, 1, (size_t) n_embd_dec });
            }

            if (batch.size() == 0) {
                break;
            }

            ret = llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch.get());
            if (ret != 0) {
                SPC_ERR("llama_process[%d] returned %d\n", i, ret);
                break;
            }

            ++i;
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool /*is_other*/) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        const int32_t n_rows = verify_g_rows[seq_id];
        if (n_rows <= 0) {
            return;
        }

        const int32_t i_g = std::min<int32_t>(n_accepted, n_rows - 1);
        pending_pos_last[seq_id] = verify_pos_first[seq_id] + i_g;
        std::memcpy(pending_g_last[seq_id].data(),
                    verify_g[seq_id].data() + (size_t) i_g * n_embd_dec,
                    (size_t) n_embd_dec * sizeof(float));
    }

    // we only need to stash the deferred boundary's g_embd row for recurrent/hybrid targets:
    // their single-position checkpoints drop it on restore
    bool need_boundary_stash() const {
        const llama_model * model_tgt = llama_get_model(params.ctx_tgt);
        return llama_model_is_recurrent(model_tgt) || llama_model_is_hybrid(model_tgt);
    }

    bool get_state(llama_seq_id seq_id, std::vector<uint8_t> & data) const override {
        if (!need_boundary_stash()) {
            return false;
        }
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq || pending_pos_last[seq_id] < 0) {
            return false;
        }

        const llama_pos          pos = pending_pos_last[seq_id];
        const std::vector<float> & g = pending_g_last[seq_id];

        data.resize(sizeof(llama_pos) + g.size() * sizeof(float));
        std::memcpy(data.data(),                     &pos,     sizeof(llama_pos));
        std::memcpy(data.data() + sizeof(llama_pos), g.data(), g.size() * sizeof(float));
        return true;
    }

    void set_state(llama_seq_id seq_id, const std::vector<uint8_t> & data) override {
        if (!need_boundary_stash()) {
            return;
        }
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }
        if (data.size() != sizeof(llama_pos) + (size_t) n_embd_dec * sizeof(float)) {
            return;
        }

        llama_pos pos = -1;
        std::memcpy(&pos, data.data(), sizeof(llama_pos));

        pending_pos_last[seq_id] = pos;
        pending_g_last[seq_id].resize(n_embd_dec);
        std::memcpy(pending_g_last[seq_id].data(), data.data() + sizeof(llama_pos), (size_t) n_embd_dec * sizeof(float));
    }
};

// DFlash: block-diffusion drafting with a draft-side KV cache injection
struct common_speculative_impl_draft_dflash : public common_speculative_impl {
    common_params_speculative_draft params;

    common_batch batch;        // noise tokens
    common_batch batch_inject; // target features for KV cache injection

    std::vector<float> features_buf; // [n_chunk, n_embd_enc] gathered target features

    std::vector<common_sampler_ptr> smpls;

    // backend sampler chain per seq, attached to ctx_dft
    std::vector<llama_sampler *> backend_chains;

    int32_t n_embd_dec = 0;  // draft hidden size
    int32_t n_embd_enc = 0;  // target_layer_ids_n * target_hidden_size
    int32_t n_embd_tgt = 0;  // target model hidden size

    int32_t     block_size    = 0;
    llama_token mask_token_id = 0;

    bool    is_lilicorr    = false;
    int32_t lilicorr_top_k = 0;
    size_t lilicorr_rows = 0;
    std::vector<float> lilicorr_embd;
    std::vector<llama_pos> lilicorr_pos;

    bool    is_dflash2     = false;
    bool    is_mrope       = false;
    int32_t selector_top_k = 0;

    // DFlash2 probabilistic drafting: one generator per sequence and the seed it was last seeded with (-1: reseed)
    std::vector<std::mt19937> dflash2_rng;
    std::vector<int64_t>      dflash2_seed;

    // draft-dspark: the draft carries a Markov head and uses an anchor-first block layout
    const bool is_dspark;

    // dspark speculators
    bool sample_from_anchor = true;

    // block-internal attention
    bool causal_attn = false;

    const int32_t * target_layer_ids   = nullptr; // model_dft's extract layer indices
    uint32_t        target_layer_ids_n = 0;

    // inject target features from the target's device buffer instead of host copies (env LLAMA_DFLASH_DEV=0 disables)
    bool use_dev_features = false;
    // [QLH-SPLIT] target_layer_ids are *source-model* ids: DFlash's sidecar names trunk layers
    // 2,7,12,17,23,28,33,38 of the 40-block model. A chain splits those over two contexts, so
    // each id has to resolve to (context, local layer index): ids below the head's block count
    // live in ctx_tgt (the head), the rest in ctx_tail offset by that count. llama_split_attach
    // guarantees the ranges are contiguous, so the head's n_layer() *is* the tail's start and no
    // extra accessor is needed. Whole-model loads never fill ctx_tail: every id stays on ctx_tgt.
    std::vector<llama_context *> tap_ctx;
    std::vector<uint32_t>        tap_lid;

    // [QLH-SPLIT] v9: the taps that live on a *remote* stage. There is no context to read those out of,
    // so they are the one thing this drafter needs from the wire: `tap_remote[k]` says tap k is one of
    // them, `tap_remote_src` is the set of source-layer ids asked for (every remote stage is sent the
    // whole set and answers what it owns), and `tap_remote_ptr` is where each tap's rows were found on
    // the last decode -- null for a tap the stages did not answer.
    //
    // `tap_remote_ptr` is resolved once per frame, not per read: the answer is per frame and the
    // accessor below would otherwise re-scan the block list for every row of every tap.
    std::vector<char>          tap_remote;
    std::vector<uint32_t>      tap_remote_src;
    std::vector<const float *> tap_remote_ptr;
    uint32_t                   tap_remote_rows = 0;
    // One line per request when the tap channel came back empty, so a chain whose stages do not own
    // the head's tap layers is visible without a log line per decode.
    bool                       tap_remote_warned = false;
    // [QLH-SPLIT] v9: set when a remote tap did not arrive for a frame. The draft KV is positional and
    // the *next* successful injection overwrites the positions it covers, so a single missed round
    // heals itself; positions accepted while it was missed do not, because nothing injects them again.
    // Rather than keep drafting from a KV with holes in it -- every round then pays the round trip and
    // buys nothing -- this stops drafting for the rest of the request. The request is unaffected: it
    // produces exactly the tokens the chain would have produced without speculation.
    bool                       tap_remote_dead = false;

    common_speculative_impl_draft_dflash(const common_params_speculative & params, uint32_t n_seq,
            common_speculative_type type = COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH)
        : common_speculative_impl(type, n_seq, params.draft.n_max)
        , params(params.draft)
        , is_dspark(type == COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK)
    {
        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;
        GGML_ASSERT(ctx_tgt && ctx_dft && "DFlash requires ctx_tgt and ctx_dft to be set");

        const llama_model * model_dft = llama_get_model(ctx_dft);
        const llama_model * model_tgt = llama_get_model(ctx_tgt);

        target_layer_ids   = llama_model_target_layer_ids  (model_dft);
        target_layer_ids_n = llama_model_target_layer_ids_n(model_dft);
        GGML_ASSERT(target_layer_ids_n > 0 && "DFlash model has no target_layer_ids");

        // [QLH-SPLIT] resolve every tap to the stage that owns it (see tap_ctx/tap_lid above)
        {
            llama_context * ctx_tail = this->params.ctx_tail;
            // [QLH-SPLIT] v9: a remote chain has no tail context in this process. Its depth comes from
            // the source model instead -- and only for a bounds message, because the stage resolves the
            // ids against its own range and is the authority on what it owns.
            const bool    tail_remote   = this->params.ctx_tail_remote;
            const int32_t head_n_layer  = (int32_t) llama_model_n_layer(model_tgt);
            const int32_t tail_n_layer  = ctx_tail
                    ? (int32_t) llama_model_n_layer(llama_get_model(ctx_tail))
                    : (tail_remote ? (int32_t) llama_model_n_layer_src(model_tgt) - head_n_layer : 0);

            tap_ctx.resize(target_layer_ids_n);
            tap_lid.resize(target_layer_ids_n);
            tap_remote.assign(target_layer_ids_n, 0);
            for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
                const int32_t id = target_layer_ids[k];
                if (id < head_n_layer || (ctx_tail == nullptr && !tail_remote)) {
                    // A whole model keeps every id on one context -- there is no cut to resolve, and a
                    // whole-model load must behave exactly as it did before the chain existed.
                    tap_ctx[k] = ctx_tgt;
                    tap_lid[k] = (uint32_t) id;
                } else if (tail_remote) {
                    // [QLH-SPLIT] v9: this layer is computed by another process, so there is nothing
                    // to read here -- it has to come back on the decode. The id stays a *source* id and
                    // the stage subtracts its own range start, which is what lets the same set of ids
                    // be sent to every stage of a chain without this side knowing where the cuts are.
                    tap_remote[k] = 1;
                    tap_remote_src.push_back((uint32_t) id);
                    tap_ctx[k] = nullptr;
                    tap_lid[k] = (uint32_t) id;
                } else {
                    tap_ctx[k] = ctx_tail;
                    tap_lid[k] = (uint32_t) (id - head_n_layer);
                    if ((int32_t) tap_lid[k] >= tail_n_layer) {
                        GGML_ABORT("DFlash: target layer %d maps to tail layer %d, but the tail has %d",
                                id, (int32_t) tap_lid[k], tail_n_layer);
                    }
                }
            }
        }

        n_embd_tgt    = llama_model_n_embd(model_tgt);
        n_embd_dec    = llama_model_n_embd(model_dft);
        n_embd_enc    = (int32_t) target_layer_ids_n * n_embd_tgt;

        // read the trained block size from the dflash.block_size metadata key
        block_size = 16;
        {
            char buf[32] = {};
            if (llama_model_meta_val_str(model_dft, "dflash.block_size", buf, sizeof(buf)) >= 0) {
                block_size = std::atoi(buf);
            }
            if (llama_model_meta_val_str(model_dft, "dflash.sample_from_anchor", buf, sizeof(buf)) >= 0) {
                sample_from_anchor = std::strcmp(buf, "true") == 0;
            }
            if (llama_model_meta_val_str(model_dft, "dflash.attention.causal", buf, sizeof(buf)) >= 0) {
                causal_attn = std::strcmp(buf, "true") == 0;
            }
        }

        selector_top_k = llama_model_dflash_selector_top_k(model_dft);
        is_dflash2     = selector_top_k > 0;
        dflash2_rng.resize(n_seq);
        dflash2_seed.assign(n_seq, -1);
        char lilicorr_meta[32] = {};
        lilicorr_top_k = llama_model_meta_val_str(model_dft, "lilicorr.candidate_topk", lilicorr_meta, sizeof(lilicorr_meta)) >= 0
                ? std::atoi(lilicorr_meta) : 0;
        is_lilicorr = lilicorr_top_k > 0;
        if (is_lilicorr) {
            lilicorr_rows = (size_t) block_size + 1;
            const size_t lilicorr_slots = (size_t) n_seq * lilicorr_rows;
            lilicorr_embd.resize(lilicorr_slots * (size_t) n_embd_dec);
            lilicorr_pos.assign(lilicorr_slots, -1);
        }
        mask_token_id = llama_vocab_mask(llama_model_get_vocab(model_dft));

        if (is_dspark && this->params.p_min > 0.0f) {
            char buf[16] = {};
            const bool has_conf =
                llama_model_meta_val_str(model_dft, "dflash.has_confidence_head", buf, sizeof(buf)) < 0 ||
                std::strcmp(buf, "true") == 0;
            if (!has_conf) {
                throw std::runtime_error("DSpark draft has no confidence head: please set --spec-draft-p-min 0");
            }
        }

        LOG_INF("%s: adding speculative implementation '%s'\n", __func__, common_speculative_type_to_str(type).c_str());
        LOG_INF("%s: - n_max=%d, n_min=%d, p_min=%.2f\n", __func__, this->params.n_max, this->params.n_min, this->params.p_min);
        LOG_INF("%s: - block_size=%d, mask_token_id=%d, n_extract=%u, sample_from_anchor=%s\n", __func__,
                block_size, mask_token_id, target_layer_ids_n, sample_from_anchor ? "true" : "false");

        // DFlash input is [id_last, <mask> * (block_size-1)]: in-place denoising yields at most
        // block_size-1 draft tokens, anchor-first DSpark yields a full block_size draft tokens
        const int32_t n_draft_max = is_dspark && sample_from_anchor ? block_size : block_size - 1;
        if (this->params.n_max > n_draft_max || this->params.n_min > n_draft_max) {
            LOG_WRN("%s: requested draft size (n_max=%d, n_min=%d) exceeds the trained block size %d -- clamping to %d\n",
                    __func__, this->params.n_max, this->params.n_min, block_size, n_draft_max);
            this->params.n_max = std::min(this->params.n_max, n_draft_max);
            this->params.n_min = std::min(this->params.n_min, n_draft_max);
        }
        this->n_max = this->params.n_max;

        batch        = common_batch(ctx_dft);
        batch_inject = common_batch(ctx_dft);

        // embd batches on an M-RoPE draft carry 4 position rows per token
        is_mrope = llama_model_rope_type(model_dft) == LLAMA_ROPE_TYPE_MROPE;

        smpls.resize(n_seq);
        for (auto & s : smpls) {
            common_params_sampling sparams;
            sparams.no_perf  = false;
            sparams.top_k    = 10;
            sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
            s.reset(common_sampler_init(model_dft, sparams));
        }

        // offload draft sampling to the backend
        backend_chains.assign(n_seq, nullptr);
        if (this->params.backend_sampling && !is_dflash2 && !is_lilicorr) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
                llama_sampler_chain_add(chain, llama_sampler_init_top_k(10));

                if (!llama_set_sampler(ctx_dft, seq_id, chain)) {
                    SPC_WRN("backend offload failed for seq_id=%d; using CPU sampler\n", (int) seq_id);
                    llama_sampler_free(chain);
                    chain = nullptr;
                }
                backend_chains[seq_id] = chain;
            }
        }

        // turn on extraction of the target layers' input embeddings
        int n_head_taps = 0;
        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            n_head_taps += tap_ctx[k] == ctx_tgt ? 1 : 0;
            if (tap_ctx[k] != nullptr) {
                llama_set_embeddings_layer_inp(tap_ctx[k], tap_lid[k], true);
            }
        }
        if (!tap_remote_src.empty()) {
            // [QLH-SPLIT] v9: ask the remote stages for the layers this process cannot compute. The set
            // is sticky on the head context (every decode carries it), because which layers a draft
            // model reads is a property of the chain and not of a frame -- and because arming it from
            // here would put the request after the decode it belongs to.
            llama_split_set_layer_inp(ctx_tgt, tap_remote_src.data(), (uint32_t) tap_remote_src.size());
            tap_remote_ptr.assign(target_layer_ids_n, nullptr);
            LOG_INF("%s: remote split chain: %zu of %u DFlash taps cross the wire (head has %d blocks; "
                    "the rest are read back from the stages)\n", __func__,
                    tap_remote_src.size(), target_layer_ids_n, (int) llama_model_n_layer(model_tgt));
        }
        if (this->params.ctx_tail) {
            LOG_INF("%s: split chain: %u DFlash taps over 2 stages (head %d blocks, tail %d blocks; "
                    "%d on the head, %u on the tail; source layer %d -> tail local %u)\n", __func__,
                    target_layer_ids_n,
                    (int) llama_model_n_layer(model_tgt),
                    (int) llama_model_n_layer(llama_get_model(this->params.ctx_tail)),
                    n_head_taps, target_layer_ids_n - (uint32_t) n_head_taps,
                    target_layer_ids[target_layer_ids_n - 1], tap_lid[target_layer_ids_n - 1]);
        }

        // the device buffer keeps layers in ascending id order, the fc input expects target_layer_ids order
        {
            const char * env = getenv("LLAMA_DFLASH_DEV");
            const bool ascending = std::adjacent_find(target_layer_ids, target_layer_ids + target_layer_ids_n,
                    [](int32_t a, int32_t b) { return a >= b; }) == target_layer_ids + target_layer_ids_n;

            use_dev_features = !is_lilicorr && ascending && (env == nullptr || atoi(env) != 0);
            if (use_dev_features) {
                llama_set_embeddings_layer_inp_dev(ctx_tgt, true);
                use_dev_features = llama_supports_embd_row(ctx_dft);
                if (!use_dev_features) {
                    llama_set_embeddings_layer_inp_dev(ctx_tgt, false);
                }
            }
            LOG_INF("%s: DFlash target features via %s\n", __func__, use_dev_features ? "device buffer" : "host copies");
            if (!use_dev_features && n_head_taps < (int) target_layer_ids_n) {
                // [QLH-SPLIT] v9: the usual reason a chain falls back. The device buffer is ctx_tgt's,
                // so it holds only the extract layers this context computes; a chain keeps the rest in
                // the tail context or on a remote stage, and no row of that buffer can carry them.
                LOG_INF("%s:   device buffer unusable on this chain: it holds the head's %d of %u extract layers\n",
                        __func__, n_head_taps, target_layer_ids_n);
            }
        }

        // DFlash2 reads its selector lattice from h_nextn and never consumes raw logits.
        llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ !is_dflash2 && !is_lilicorr);
        llama_set_causal_attn(ctx_dft, causal_attn); // DFlash needs non-causal attention unless the model says otherwise
    }

    ~common_speculative_impl_draft_dflash() override {
        auto * ctx_dft = this->params.ctx_dft;
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) backend_chains.size(); ++seq_id) {
            if (backend_chains[seq_id] == nullptr) {
                continue;
            }
            if (ctx_dft) {
                llama_set_sampler(ctx_dft, seq_id, nullptr);
            }
            llama_sampler_free(backend_chains[seq_id]);
        }
        backend_chains.clear();
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        // reseed per request rather than per round, or two identical requests differ
        dflash2_seed[seq_id] = -1;

        // [QLH-SPLIT] v9: a request starts with the tap channel assumed good. A missed tap stops
        // drafting for the request (see tap_remote_dead), and the next request tries again -- the
        // frame that missed one is gone, so there is nothing to retry within this one.
        if (seq_id == 0) {
            tap_remote_dead   = false;
            tap_remote_warned = false;
            tap_remote_rows   = 0;
        }

        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }

        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(params.ctx_dft), seq_id);
        if (pos_max < N - 1) {
            LOG_WRN("%s: ctx_dft pos_max=%d < N-1=%d - process() did not run on every prefill ubatch. "
                    "Drafts may degrade.\n",
                    __func__, (int) pos_max, N - 1);
        }
    }

    // [QLH-SPLIT] v9: one accessor for both kinds of tap. A tap is either read out of the context that
    // computed the layer -- which for a whole model and a local chain is a buffer in this process, and
    // costs nothing -- or out of the rows the remote stages answered on this frame.
    //
    // Why an accessor and not two code paths in process(): the two differ only in *where* a row is, and
    // splitting the injection loop in two is how the remote half quietly drifts from what the local half
    // was measured to do. `k` is the tap index; the returned block is [row][n_embd] of this frame.
    const float * tap_base(uint32_t k) const {
        if (!tap_remote[k]) {
            return llama_get_embeddings_layer_inp(tap_ctx[k], tap_lid[k]);
        }
        return tap_remote_ptr[k];
    }

    // Does this frame's remote answer cover row `last` of the batch? It always does when the stage
    // answered the frame (the wire carries n_tok rows per tap, and the client validates that byte
    // count), so this only fires on a stage that answered a shorter frame than it was sent -- reading
    // past the block would inject whatever memory follows it into the draft KV.
    bool remote_taps_cover(int32_t last) const {
        return tap_remote_src.empty() || last < (int32_t) tap_remote_rows;
    }

    // Bind this frame's remote taps to the blocks the stages answered for it. False means at least one
    // tap is missing, i.e. this frame's features cannot be injected; the caller then skips the injection
    // instead of failing the request, because a draft is an optimization and the request without it is
    // exactly the one the chain would have served without speculation. One line per process: a chain
    // that cannot answer them would otherwise log this on every decode.
    bool resolve_remote_taps() {
        if (tap_remote_src.empty()) {
            return true;
        }
        const uint32_t n_blocks = llama_split_take_layer_inp_n(params.ctx_tgt);
        uint32_t       n_rows   = 0;
        bool           ok       = n_blocks > 0;
        for (uint32_t k = 0; k < (uint32_t) tap_remote.size(); ++k) {
            tap_remote_ptr[k] = nullptr;
            if (!tap_remote[k]) {
                continue;
            }
            for (uint32_t b = 0; b < n_blocks && tap_remote_ptr[k] == nullptr; ++b) {
                uint32_t      id     = 0;
                uint32_t      rows_b = 0;
                uint32_t      embd_b = 0;
                const float * p = llama_split_take_layer_inp(params.ctx_tgt, b, &id, &rows_b, &embd_b);
                if (p != nullptr && id == tap_lid[k] && embd_b == (uint32_t) n_embd_tgt) {
                    tap_remote_ptr[k] = p;
                    n_rows = rows_b;
                }
            }
            ok = ok && tap_remote_ptr[k] != nullptr;
        }
        tap_remote_rows = ok ? n_rows : 0;
        if (!ok) {
            tap_remote_dead = true;
            if (!tap_remote_warned) {
                tap_remote_warned = true;
                SPC_ERR("DFlash: the remote chain answered %u of %zu layer-input taps; this request "
                        "stops drafting rather than drafting from a KV the features never reached "
                        "(its tokens are unchanged)\n", n_blocks, tap_remote_src.size());
            }
        }
        return ok;
    }

    bool process(const common_batch & batch_in) override {
        if (batch_in.size() <= 0) {
            return true;
        }

        // Target prefill may contain token IDs or multimodal embeddings. Both
        // produce the target-layer features used to seed the draft KV cache, so
        // embeddings are injected too, except the pinned ones skipped below.
        // TODO: revisit after https://github.com/ggml-org/llama.cpp/pull/24669 is merged
        const bool has_tokens     = batch_in.has_token();
        const bool has_embeddings = batch_in.has_embd();
        if (has_tokens == has_embeddings) {
            return true;
        }

        const int32_t n_tokens = batch_in.size();

        // per-seq inclusive batch range (assumes each seq's tokens are contiguous in the batch)
        std::vector<int32_t> i_batch_beg(n_seq, -1);
        std::vector<int32_t> i_batch_end(n_seq, -1);
        for (int32_t k = 0; k < n_tokens; ++k) {
            const llama_seq_id seq_id = batch_in.tokens[k].seq_id;
            if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
                continue;
            }
            i_batch_end[seq_id] = k;
            if (i_batch_beg[seq_id] < 0) {
                i_batch_beg[seq_id] = k;
            }
        }

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;

        // [QLH-SPLIT] v9: bind this frame's remote taps, or skip the injection for it. Skipping is the
        // whole error policy of this channel: the draft model's KV then misses this frame's positions
        // (the next injection overwrites them, nothing is left half-written), which costs acceptance and
        // not a single token of the answer. The request that would have failed is the one the chain
        // serves without speculation.
        if (!resolve_remote_taps()) {
            return true;
        }

        const int32_t n_ubatch = (int32_t) llama_n_ubatch(ctx_dft);

        // [QLH-SPLIT] v9: the device-buffer path is only valid while the graph agrees that the batch may
        // use embd rows -- graph_params() resolves layer_inp_src from this same predicate. Asking it here,
        // on the frame the rows are built, is what keeps the two from drifting apart: a constructor-time
        // answer alone can go stale, and the batch walking ahead of the graph lands in the assert in
        // models/dflash.cpp build_inp_target_features() (a logic error, so that assert stays).
        const bool dev_features = use_dev_features && llama_supports_embd_row(ctx_dft);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_batch_beg[seq_id] < 0) {
                continue;
            }
            const int32_t n_rows = i_batch_end[seq_id] - i_batch_beg[seq_id] + 1;

            // [QLH-SPLIT] v9: a remote answer that does not cover this sequence's rows cannot be
            // injected (see remote_taps_cover). Same policy as a missing tap: skip, do not fail.
            if (!remote_taps_cover(i_batch_end[seq_id])) {
                tap_remote_dead = true;
                if (!tap_remote_warned) {
                    tap_remote_warned = true;
                    SPC_ERR("DFlash: the remote chain answered %u rows of layer-input taps for a "
                            "frame that needs row %d; this request stops drafting (its tokens are "
                            "unchanged)\n", tap_remote_rows, (int) i_batch_end[seq_id]);
                }
                continue;
            }

            // an M-RoPE image pins all its rows to one position, so a windowed draft
            // cache cannot free cells for it - skip it, the draft can jump over the gap
            const bool pos_pinned = batch_in.tokens[i_batch_beg[seq_id]].pos[0] == batch_in.tokens[i_batch_end[seq_id]].pos[0];
            if (has_embeddings && n_rows > 1 && pos_pinned) {
                continue;
            }

            for (int32_t offset = 0; offset < n_rows; offset += n_ubatch) {
                const int32_t n_chunk = std::min(n_ubatch, n_rows - offset);

                if (dev_features) {
                    // the features are the target batch rows in the target's device buffer
                    batch_inject.clear();
                    for (int32_t i = 0; i < n_chunk; ++i) {
                        const int32_t   row        = i_batch_beg[seq_id] + offset + i;
                        const llama_pos p          = batch_in.tokens[row].pos[0];
                        const llama_pos pos_arr[4] = { p, p, p, 0 };
                        batch_inject.add_embd_row(row, pos_arr, seq_id, false);
                    }

                    const int32_t rc = llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch_inject.get());
                    if (rc != 0) {
                        LOG_ERR("%s: llama_process(ctx_dft) failed rc=%d (n_tokens=%d, offset=%d)\n",
                                __func__, rc, (int) n_chunk, (int) offset);
                        return false;
                    }
                    continue;
                }

                // gather target features per extract layer; the fused decode encodes and
                // injects them into the K/V cache at the target positions
                features_buf.resize((size_t) n_chunk * n_embd_enc);
                for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
                    // [QLH-SPLIT] v9: the rows this tap has for this *frame*, from whichever stage owns
                    // the layer -- a local context's buffer, or the block the remote stages answered
                    // (see tap_base / resolve_remote_taps above). One lookup per tap, then the rows.
                    const float * base = tap_base(k);
                    if (base == nullptr) {
                        // Only a local tap can get here: a remote one was resolved before the loop and
                        // a missing one skipped the whole frame. A local context that does not export
                        // an enabled layer is a broken build, not a degraded chain.
                        GGML_ABORT("DFlash: target layer %d input not extracted.", target_layer_ids[k]);
                    }
                    for (int32_t i = 0; i < n_chunk; ++i) {
                        const float * src = base + (size_t) (i_batch_beg[seq_id] + offset + i) * n_embd_tgt;
                        float       * dst = features_buf.data() + (size_t) i * n_embd_enc + k * (size_t) n_embd_tgt;
                        std::memcpy(dst, src, (size_t) n_embd_tgt * sizeof(float));
                    }
                }

                batch_inject.clear();
                for (int32_t i = 0; i < n_chunk; ++i) {
                    const llama_pos p = batch_in.tokens[i_batch_beg[seq_id] + offset + i].pos[0];
                    const llama_pos pos_arr[4] = { p, p, p, 0 };
                    batch_inject.add_embd({ features_buf.data() + (size_t) i * n_embd_enc, 1, (size_t) n_embd_enc }, pos_arr, seq_id, false);
                }
                const int32_t rc = llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch_inject.get());
                if (rc != 0) {
                    LOG_ERR("%s: llama_process(ctx_dft) failed rc=%d (n_tokens=%d, offset=%d)\n",
                            __func__, rc, (int) n_chunk, (int) offset);
                    return false;
                }
                if (is_lilicorr) {
                    // Ordinary DFlash can write to the KV cache, but LiliCorr needs to store embeddings across sequences.
                    const float * projected = llama_get_embeddings_nextn(ctx_dft);
                    if (projected) {
                        const size_t seq_offset = (size_t) seq_id * lilicorr_rows;
                        const llama_pos pos_min = std::max(0, batch_inject.tokens.back().pos[0] - block_size);
                        for (int32_t i = 0; i < n_chunk; ++i) {
                            const llama_pos pos = batch_inject.tokens[i].pos[0];
                            if (pos < pos_min) {
                                continue;
                            }
                            const size_t slot = seq_offset + (size_t) pos % lilicorr_rows;
                            std::memcpy(lilicorr_embd.data() + slot * (size_t) n_embd_dec,
                                    projected + (size_t) i * n_embd_dec, (size_t) n_embd_dec * sizeof(float));
                            lilicorr_pos[slot] = pos;
                        }
                    }
                }
            }
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        // [QLH-SPLIT] v9: this request's tap channel failed, so the draft KV never received the target
        // features it drafts from. Drafting anyway would pay every round trip and produce drafts that
        // are wrong for a reason no verifier can fix -- the tokens are unaffected either way, so the
        // honest answer is "no drafts".
        if (tap_remote_dead) {
            for (auto & dp : dparams) {
                dp.drafting = false;
            }
            return;
        }

        batch.clear();

        // build one batch holding every drafting sequence's noise block into a single decode)
        // record where each block starts and its size
        std::vector<int32_t> i_block_beg(n_seq, -1);
        std::vector<int32_t> n_block    (n_seq,  0);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            common_sampler_reset(smpls[seq_id].get());

            const int32_t n = (int32_t) dp.pos0;

            const int32_t n_draft = params.n_max;
            const int32_t n_block_tokens = is_lilicorr ? block_size : n_draft + (is_dspark && sample_from_anchor ? 0 : 1);
            i_block_beg[seq_id] = batch.size();
            n_block    [seq_id] = n_block_tokens;
            for (int32_t i = 0; i < n_block_tokens; ++i) {
                const int32_t idx = batch.add(i == 0 ? dp.id_last : mask_token_id, n + i, seq_id, !is_dflash2 && !is_lilicorr);
                if (is_lilicorr) {
                    // For LiliCorr drafting batch, set state embeddings for use in the associated transformer head.
                    const llama_pos pos = dp.pos0 - 1;
                    const size_t slot = (size_t) seq_id * lilicorr_rows + (size_t) pos % lilicorr_rows;
                    batch.set_embd_state(idx, { lilicorr_embd.data() + slot * (size_t) n_embd_dec, 1, (size_t) n_embd_dec });
                }
            }
        }

        if (batch.size() == 0) {
            return;
        }

        // decode all sequence's noise block in a single batch
        int ret = llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch.get());
        if (ret != 0) {
            LOG_WRN("%s: llama_process returned %d\n", __func__, ret);
            return;
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_block_beg[seq_id] < 0) {
                continue;
            }
            auto & dp = dparams[seq_id];

            const int32_t beg            = i_block_beg[seq_id];
            const int32_t n_block_tokens = n_block[seq_id];

            auto * smpl = smpls[seq_id].get();

            auto & result = *dp.result;

            if (is_dflash2 || is_lilicorr) {
                // Both DFlash and LiliCorr greedily walk the candidate lattice.
                const float * lattice = llama_get_embeddings_nextn(ctx_dft);
                GGML_ASSERT(lattice && "DFlash2 selector produced no lattice");

                // DFlash2 can instead sample each step and hand q to the verifier for rejection sampling
                const bool sample = is_dflash2 && params.probabilistic && dp.result_q;

                // greedy keeps temperature 1, sampling uses the target's
                const float temp = sample ? dp.temp : 1.0f;

                if (sample && dflash2_seed[seq_id] != dp.seed) {
                    dflash2_seed[seq_id] = dp.seed;
                    // the default seed means random; mixed so it differs from the target's, as in spec_retune
                    dflash2_rng[seq_id].seed(dp.seed == LLAMA_DEFAULT_SEED ? std::random_device{}() : dp.seed ^ 0x85ebca6bu);
                }

                std::vector<float>   probs;
                std::vector<int32_t> sorted_idx;

                int32_t predecessor = 0;
                for (int32_t i = 1; i < (is_dflash2 ? n_block_tokens : params.n_max + 1); ++i) {
                    const float * row = lattice + (size_t) (beg + i) * n_embd_dec;
                    const int32_t top_k = is_dflash2 ? selector_top_k : lilicorr_top_k;
                    const float * scores = row + top_k + (size_t) predecessor * top_k;

                    predecessor = (int32_t) std::distance(scores,
                            std::max_element(scores, scores + top_k));
                    if (params.p_min > 0.0f || sample) {
                        // softmax(scores / temp) at the argmax, i.e. 1 / sum(exp((s_k - s_max) / temp))
                        probs.resize(top_k);
                        float sum = 0.0f;
                        for (int32_t k = 0; k < top_k; ++k) {
                            probs[k] = std::exp((scores[k] - scores[predecessor]) / temp);
                            sum += probs[k];
                        }
                        if (1.0f / sum < params.p_min) {
                            break;
                        }
                        for (int32_t k = 0; k < top_k; ++k) {
                            probs[k] /= sum;
                        }
                    }
                    if (sample) {
                        // ggml_top_k leaves candidates unordered, so walk them by token id to keep a seed's draft stable
                        sorted_idx.resize(top_k);
                        for (int32_t k = 0; k < top_k; ++k) {
                            sorted_idx[k] = k;
                        }
                        std::sort(sorted_idx.begin(), sorted_idx.end(), [row](int32_t a, int32_t b) { return row[a] < row[b]; });

                        // a round-off overshoot keeps the argmax, which has q > 0
                        double u = std::uniform_real_distribution<double>(0.0, 1.0)(dflash2_rng[seq_id]);
                        for (const int32_t k : sorted_idx) {
                            u -= probs[k];
                            if (u < 0.0) {
                                predecessor = k;
                                break;
                            }
                        }

                        auto & q = dp.result_q->emplace_back();
                        q.reserve(top_k);
                        for (int32_t k = 0; k < top_k; ++k) {
                            q.push_back({ (llama_token) row[k], scores[k], probs[k] });
                        }
                    }
                    result.push_back((llama_token) row[predecessor]);
                }

                if (result.size() < (size_t) params.n_min) {
                    result.clear();
                    if (sample) {
                        dp.result_q->clear();
                    }
                }
                continue;
            }

            if (is_dspark) {
                // DSpark: read from the first draft slot, truncate below the confidence threshold
                const float * conf = params.p_min > 0.0f ? llama_get_embeddings_nextn(ctx_dft) : nullptr;
                // bonus-anchor drafts read the mask positions only, like DFlash
                const int32_t i_draft_beg = sample_from_anchor ? 0 : 1;
                for (int32_t i = i_draft_beg; i < n_block_tokens; ++i) {
                    const int32_t idx = beg + i;

                    if (conf && conf[(size_t) idx * n_embd_dec] < params.p_min) {
                        break;
                    }

                    common_sampler_sample(smpl, ctx_dft, idx, true);

                    const auto * cur_p = common_sampler_get_candidates(smpl, true);

                    for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                        LOG_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                                seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                                common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                    }

                    const llama_token id = cur_p->data[0].id;

                    common_sampler_accept(smpl, id, true);

                    result.push_back(id);
                }
            } else {
                // greedily read the predicted block at this sequence's noise positions 1..n_block_tokens-1
                for (int32_t i = 1; i < n_block_tokens; ++i) {
                    common_sampler_sample(smpl, ctx_dft, beg + i, true);

                    const auto * cur_p = common_sampler_get_candidates(smpl, true);

                    for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                        LOG_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                                seq_id, k, i - 1, cur_p->data[k].id, cur_p->data[k].p,
                                common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                    }

                    const llama_token id = cur_p->data[0].id;

                    if (cur_p->data[0].p < params.p_min) {
                        break;
                    }

                    common_sampler_accept(smpl, id, true);

                    result.push_back(id);
                }
            }

            if (result.size() < (size_t) params.n_min) {
                result.clear();
            }
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }
};

// [QLH-SPLIT] chain catch-up mode named by `name`, or -1 when the variable is unset.
//
// -1 is the whole point: "unset" (adaptive) and "=0" (explicitly off) must not collapse into the
// same value, or the adaptive default could never be reached and an explicit 0 could not be told
// apart from it. An empty value counts as an explicit 0, matching what the variable meant before
// this round (and what QLH_CHAIN_DRAFT_KEEP_KV still means).
//
// One parser for both variables: they differ in nothing but the name. The explicit override and
// the server's per-request pick have to be read the same way, or "unset" would mean two different
// things depending on which one a caller happens to consult. That is why this is exported (see
// common/speculative.h) instead of staying file-local: the server asks it to decide whether to
// write the automatic value at all, and that answer has to be the same parse as the one that reads
// it back (tools/server/server-context.cpp, launch_slot_with_task).
int common_speculative_catchup_mode_read(const char * name) {
    const char * v = std::getenv(name);
    if (v == nullptr) {
        return -1;
    }
    return (v[0] == '\0' || v[0] == '0') ? 0 : (v[0] == '2' ? 2 : 1);
}

// the manual override -- read first, so an explicit 0/1/2 always wins over the adaptive default
static int catchup_mode_env() {
    return common_speculative_catchup_mode_read("QLH_CHAIN_DRAFT_CATCHUP");
}

// the mode the server picked for this request, or -1 when it wrote nothing (a chain that cannot
// catch up, or an older server). It is only written when the explicit variable is unset, so the
// two never fight.
static int catchup_mode_auto() {
    return common_speculative_catchup_mode_read("QLH_CHAIN_DRAFT_CATCHUP_AUTO");
}

struct common_speculative_impl_draft_mtp : public common_speculative_impl {
    common_params_speculative_draft params; // reuses the draft-model params slot (ctx_tgt/ctx_dft)

    common_batch batch;

    std::vector<common_sampler_ptr> smpls;

    std::vector<common_params_sampling> smpls_cfg;

    // backend sampler chain per seq, attached to ctx_dft
    std::vector<llama_sampler *> backend_chains;

    int32_t n_embd = 0;

    // One MTP draft driver, four modes (set once in the ctor):
    //   is_mem_shared (gemma4): shares the target KV, runs all heads in one graph.
    //   chain_heads (step35): n_mtp_layers trained heads, one per draft step.
    //   is_chain (split chain): the target is the tail segment of a local chain, so h_nextn
    //       comes from that context and the draft cannot replay the chain's token history.
    //   none of them (qwen35 / qwen35moe): a single trained MTP head over a whole model.
    //
    // [QLH-SPLIT] v8 adds a fifth, `is_remote`, which is not a variant of the four but their
    // absence: the MTP head is in another process. There is no ctx_dft at all -- no draft KV, no
    // batch, no sampler here -- and both the h_nextn rows and the drafts come over the chain's
    // connection (llama-ext.h: llama_context_split_mtp_draft / llama_split_take_nextn_rows). Every
    // branch below that reaches for ctx_dft is guarded by it.
    int32_t n_mtp_layers  = 1;
    bool    is_mem_shared = false;   // gemma4
    bool    chain_heads   = false;   // derived in the ctor: n_mtp_layers > 1 && !is_mem_shared && !is_chain
                                     // (why !is_chain: see the derivation in the ctor -- a chain never
                                     // gives the draft the prompt KV that the per-head rebuild needs)
    bool    is_chain      = false;   // [QLH-SPLIT] split chain: draft belongs to the tail segment
    bool    is_remote     = false;   // [QLH-SPLIT] v8: ... which is a stage process; no ctx_dft here

    // [QLH-SPLIT] keep-KV chain route (default on; QLH_CHAIN_DRAFT_KEEP_KV=0 restores the
    // per-round clear it replaced, bit for bit).
    //
    // Without it a chain clears the draft KV every round (see draft()), because the draft never
    // saw the prompt: the only thing a round may inherit is nothing. With it the draft keeps what
    // it built, and the point is that a round *does* inherit: the rows of every token this request
    // has already generated, at the tail's own positions. The prompt is still absent (it is never
    // replayed), so the gain is "the draft can read back the generated text", not "the draft has
    // the prompt". That is why the route costs nothing per round: no catch-up decode, no transfer.
    //
    // The invariant every path has to maintain:
    //     draft KV for a seq  =>  rows at positions < pos0 only
    // where pos0 is the position of id_last, i.e. the tail's next position. draft() trims to its
    // own pos0, so a rejected draft row (>= pos0) can never be read by a later round, and a tail
    // position that moved for any reason (whole-frame rollback, context shift, rebuilt chain) is
    // caught by comparing pos0 against the position this KV was left at.
    bool keep_kv = false;
    std::vector<llama_pos> keep_kv_expected; // pos0 the next draft() must see; < 0 = start empty
    std::vector<llama_pos> keep_kv_pos0;     // pos0 of the most recent draft() call
    size_t keep_kv_rounds = 0;               // rounds that continued from history
    size_t keep_kv_resets = 0;               // rounds that had to start from an empty draft KV
    bool   keep_kv_trim_warned = false;      // a refused trim is reported once, not per round

    // [QLH-SPLIT] chain catch-up route. Adaptive by default since the model-swap round; the
    // QLH_CHAIN_DRAFT_CATCHUP environment variable still overrides it completely.
    //
    // keep_kv gave the draft a *reader* for rows below pos0; the prompt's rows were still absent
    // because process() skips its catch-up decode on a chain. A local chain has everything that
    // decode needs -- the tail is a context in this process (ctx_tgt), its unmasked
    // embeddings_nextn buffer holds one h row per server batch row, and the batch still carries
    // the token ids -- so this route simply runs it, with the upstream pairing
    // (token at pos p, h of pos p-1) and the tail's own positions.
    //
    // Two modes, because they cost different things and answer different questions:
    //   1 = prefill only: the prompt rows are real, the generated ones stay the draft's own
    //       (exactly the keep_kv-only route for them).
    //   2 = every batch: same as the non-chain path, which runs this decode on every target
    //       batch, so accepted tokens also get the tail's real h back.
    //
    // [QLH-SPLIT] THE ONE BEHAVIOUR CHANGE of this round: "unset" used to mean 0 and now means
    // "pick one per request". QLH_CHAIN_DRAFT_CATCHUP unset + QLH_CHAIN_DRAFT_CATCHUP_AUTO (which
    // the server writes in launch_slot_with_task, where N_prompt and n_predict are both in hand)
    // decides. The thresholds and the measurements behind them are stated once, there, next to the
    // code that applies them -- not repeated here, where they would drift on the first retune
    // (tools/server/server-context.cpp, the QLH_CHAIN_DRAFT_CATCHUP_AUTO write).
    // An *explicit* QLH_CHAIN_DRAFT_CATCHUP=0/1/2 wins outright and the adaptive default is never
    // consulted -- "0" and "unset" must stay distinguishable, and that is what
    // common_speculative_catchup_mode_read() below encodes. Nothing here needs the server to know
    // this implementation's internals: the hand-off is the same environment channel the manual
    // override already used (common_set_env / common_get_env are the repo's cross-platform
    // wrappers for it).
    int chain_catchup = 0;                   // mode of the most recent request (logging / stats)
    bool catchup_enabled = false;            // this implementation can catch up at all
    std::vector<uint8_t>   catchup_mode;     // per-seq mode for the request in flight; 0 = off
    std::vector<uint8_t>   catchup_pending;  // 1 = this seq has not had its first draft() yet
    std::vector<llama_pos> catchup_pos;      // pos the next catch-up batch must start at; -2 = any
    size_t catchup_rows      = 0;            // rows written into the draft KV
    size_t catchup_fallbacks = 0;            // batches that degraded to "no history"

    // Per-sequence cross-batch carryover: pair (h_p, x_{p+1}) at MTP pos p+1.
    // The last h-row of one process() call needs the first token of the NEXT
    // call to pair with, so it's stashed here until that next call fires.
    std::vector<std::vector<float>> pending_h;   // [n_seq][n_embd]

    std::vector<int32_t> i_batch_beg;
    std::vector<int32_t> i_batch_end;

    // Hidden rows from the most recent target verification batch, grouped by seq.
    // Row 0 corresponds to the sampled token, row N to the Nth accepted draft token.
    std::vector<std::vector<float>> verify_h;
    std::vector<int32_t> verify_h_rows;

    std::vector<int>                i_last;
    std::vector<std::vector<float>> chain_h;

    // target h rows on the device (llama_set_embeddings_nextn_dev): the catch-up in process() reads row k - 1 of the last
    // target decode in place, and row n_batch_tgt + seq_id for the first token of seq_id, uploaded from pending_h
    // pending_h stays on the host: process() and accept() fetch it before the next target decode overwrites the rows
    bool    dev_rows    = false;
    int32_t n_batch_tgt = 0;
    std::vector<int32_t> verify_row0; // [n_seq] device row of the first verify row, see verify_h_rows

    common_speculative_impl_draft_mtp(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_MTP, n_seq, params.draft.n_max)
        , params(params.draft)
    {
        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;
        is_remote = this->params.ctx_tail_remote;
        GGML_ASSERT(ctx_tgt && (ctx_dft || is_remote) &&
                "MTP requires ctx_tgt and either ctx_dft or a remote tail to draft on");

        // [QLH-SPLIT] v8: in remote mode the width comes from the chain, not from a draft model --
        // every stage of a chain is validated to carry the same n_embd, and the h rows crossing the
        // wire are that width. n_mtp_layers is 1 by construction there: the head has no NextN block
        // to count, and the remote draft answers one trained head's tokens (chain_heads is false).
        n_embd = llama_model_n_embd_out(llama_get_model(ctx_dft != nullptr ? ctx_dft : ctx_tgt));
        if (ctx_dft != nullptr) {
            GGML_ASSERT(n_embd == llama_model_n_embd_out(llama_get_model(ctx_tgt)) &&
                    "MTP input row width must match the target h_nextn width");
            n_mtp_layers = std::max(1, (int) llama_model_n_layer_nextn(llama_get_model(ctx_dft)));
        }

        SPC_TRC("%s", "adding speculative implementation 'draft-mtp'\n");
        SPC_TRC("- n_max=%d, n_min=%d, p_min=%.2f, n_embd=%d, backend_sampling=%d, remote=%d\n", this->params.n_max, this->params.n_min, this->params.p_min, n_embd, (int) this->params.backend_sampling, (int) is_remote);
        SPC_TRC("- gpu_layers=%d, cache_k=%s, cache_v=%s, ctx_tgt=%s, ctx_dft=%s, devices=[%s]\n",
                this->params.n_gpu_layers,
                ggml_type_name(this->params.cache_type_k),
                ggml_type_name(this->params.cache_type_v),
                ctx_tgt ? "yes" : "no",
                ctx_dft ? "yes" : "no",
                common_speculative_get_devices_str(this->params.devices).c_str());

        // [QLH-SPLIT] v8: everything below is the *local* draft context's plumbing (a batch bound
        // to it, the top_k = 10 draft samplers, the optional backend sampler). In remote mode there
        // is no such context: the stage owns the draft KV, the loop and the argmax, and this side
        // only sends one h row and receives k token ids.
        if (ctx_dft != nullptr) {
            batch = common_batch(ctx_dft);

            smpls.resize(n_seq);
            for (auto & s : smpls) {
                common_params_sampling sparams;
                sparams.no_perf  = false;
                sparams.top_k    = 10;
                sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
                s.reset(common_sampler_init(llama_get_model(ctx_dft), sparams));
            }

            // offload draft sampling to the backend
            backend_chains.assign(n_seq, nullptr);
            if (this->params.backend_sampling) {
                for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                    llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
                    llama_sampler_chain_add(chain, llama_sampler_init_top_k(10));

                    if (!llama_set_sampler(ctx_dft, seq_id, chain)) {
                        SPC_WRN("backend offload failed for seq_id=%d; using CPU sampler\n", (int) seq_id);
                        llama_sampler_free(chain);
                        chain = nullptr;
                    }
                    backend_chains[seq_id] = chain;
                }
            }
        }

        // [QLH-SPLIT] In a chain ctx_tgt IS the tail segment's context (see
        // common_params_speculative_draft::ctx_tail). Only that model carries the NextN layers,
        // and only that context's decoder graph produces h_nextn, so the unmasked channel is
        // enabled there; each decode costs one dense h row per token (a memcpy), which is what
        // process() reads back below. ctx_dft keeps the masked channel: draft() indexes the
        // draft's own h rows by output row.
        //
        // [QLH-SPLIT] v8: neither applies to a remote chain. There ctx_tgt is the *head*, whose
        // range has no NextN block (enabling the channel on it makes the graph ask for a buffer
        // that was never planned), and the rows arrive over the wire instead.

        is_chain = this->params.ctx_tail != nullptr;

        // [QLH-SPLIT] "the tail is in another process" and "the tail is a context here" are
        // exclusive by construction on the server side (ctx_tail_remote is only set when ctx_tail
        // is null), but common/ is a library and this is the only place that can check it. The two
        // flags feed different subsets of the route predicates below -- keep_kv reads is_chain,
        // catchup_enabled reads both -- so a caller that set both would make them disagree about
        // whether a draft KV exists, and the catch-up arrays would be read unassigned.
        GGML_ASSERT(!(is_chain && is_remote) &&
                "a chain cannot have both a local tail (ctx_tail) and a remote tail (ctx_tail_remote)");
        // note: is_mem_shared keeps its old meaning on purpose. In chain mode ctx_other is the
        // tail segment's context, which the llama_context ctor drops for every arch that does
        // not actually share KV (qwen35/qwen35moe), so this is false there and stays true for
        // the assistants that do share (gemma4). What chain mode changes is only *where the h
        // rows come from*, never whether the KV is shared.
        // [QLH-SPLIT] v8: ctx_dft can be null (remote chain -- the draft context is in the stage),
        // and llama_get_ctx_other(nullptr) is a null dereference, not a query.
        is_mem_shared = ctx_dft != nullptr && llama_get_ctx_other(ctx_dft) == ctx_tgt;

        // [QLH-SPLIT] !is_chain is load-bearing, not symmetry with the branch above.
        //
        // chain_heads makes draft() rebuild the draft's cursor one trained head at a time, and
        // each head needs the KV of everything before pos0 *in its own decoder layer* -- that is
        // what process()'s catch-up decode fills. Chain mode skips that decode (the target batch
        // reaches the tail as embeddings, and replaying the prompt into the draft would be a full
        // extra draft pass) and draft() clears the draft KV every round, so those rows never
        // exist: the multi-head path would decode the prefix against an empty range and draft
        // over nothing, silently. With !is_chain a multi-head model falls back to the single-head
        // growth path (head 0, one row per step), which is correct and merely a weaker draft.
        //
        // Not reachable with the models on hand -- a split-chain head must be qwen35/qwen35moe
        // (split_arch_exports_pre_norm) and those ship one nextn layer here, so n_mtp_layers == 1
        // already makes this false. It is derived from the model file, though, not from the arch,
        // so the guard is what stops the next multi-head checkpoint from drafting blind.
        chain_heads   = n_mtp_layers > 1 && !is_mem_shared && !is_chain;
        // [QLH-SPLIT] v8: a remote chain enables neither channel -- ctx_dft is null there (the
        // draft runs in the stage) and the head's layer range has no NextN block, so asking the
        // head for the channel makes the graph request a buffer that was never planned. Those rows
        // arrive over the wire instead (see the is_remote branch in process()).
        if (!is_remote) {
            llama_set_embeddings_nextn(ctx_tgt, true, /*masked*/ is_mem_shared);
            llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ true);
        }

        if (chain_heads) {
            this->params.n_max = std::min(this->params.n_max, n_mtp_layers);

            chain_h.assign(n_seq, {});
            for (auto & c : chain_h) {
                c.reserve((size_t) (this->params.n_max + 1) * n_embd);
            }
        }
        // [QLH-SPLIT] One predicate, one place. `chain_local_draft` is the *capability*: this
        // process owns a draft KV that belongs to the chain and to nobody else. A shared KV would
        // be the target's (clearing or trimming it is not ours to do), and a remote chain has no
        // draft context in this process at all.
        //
        // keep_kv is that capability switched on by the environment; catchup_enabled is keep_kv
        // minus the remote case. Every other site in this file reads one of these two booleans
        // instead of re-deriving the conjunction -- draft()'s per-round clear, the catch-up decode,
        // the stats. That conjunction is exactly where this project has had two implementations
        // drift apart before, and the three-way version is the one nobody notices.
        const bool chain_local_draft = is_chain && !is_mem_shared && ctx_dft != nullptr;
        keep_kv = chain_local_draft && []() {
            // [QLH-SPLIT] Default ON. "0" (or an empty value) is the opt-out, and it must
            // reproduce the pre-route behaviour exactly: per-round clear, no retained rows,
            // no trim. An unset variable means "use the route".
            const char * v = std::getenv("QLH_CHAIN_DRAFT_KEEP_KV");
            return v == nullptr || (v[0] != '\0' && v[0] != '0');
        }();

        if (keep_kv) {
            keep_kv_expected.assign(n_seq, -1);
            keep_kv_pos0    .assign(n_seq, -1);
            LOG_INF("%s: [QLH-SPLIT] chain draft KV kept across rounds "
                    "(default; QLH_CHAIN_DRAFT_KEEP_KV=0 restores the per-round clear): "
                    "cleared per request, trimmed per round to pos0\n",
                    __func__);
        }

        // [QLH-SPLIT] chain catch-up: fills the draft KV that keep_kv then continues from. It is
        // only meaningful with keep_kv on (a per-round clear would throw the rows away again), so
        // deriving it from keep_kv is not a shortcut -- it is what guarantees the arrays below are
        // assigned whenever draft() reads them, and that a request for the route without it is
        // refused rather than silently wasted.
        //
        // The per-request mode is resolved in begin() (see there), because the adaptive default
        // needs N_prompt and n_predict and the server can only supply them per request. Here we
        // only decide whether this implementation is *capable* of catching up and seed the arrays
        // the per-request path writes into; a chain that cannot catch up leaves them empty and
        // begin() never touches them.
        catchup_enabled = keep_kv && !is_remote;
        if (catchup_enabled) {
            catchup_mode.assign(n_seq, 0);
            catchup_pending.assign(n_seq, 0);
            catchup_pos    .assign(n_seq, -2);
        }

        // [QLH-SPLIT] "catch-up asked for, keep-KV turned off" is a configuration error, so it is
        // reported once at startup, not once per request -- and this is now the *only* place that
        // can report it: catchup_enabled is derived from keep_kv, so begin() can never see the
        // conflict (it would be unreachable code there).
        if (chain_local_draft && !keep_kv) {
            const int forced = catchup_mode_env();
            if (forced > 0) {
                SPC_WRN("QLH_CHAIN_DRAFT_CATCHUP=%d needs the keep-KV route; it is off "
                        "(QLH_CHAIN_DRAFT_KEEP_KV=0), so the catch-up is disabled too\n", forced);
            }
        }

        this->n_max = this->params.n_max;

        pending_h.assign(n_seq, std::vector<float>(n_embd, 0.0f));

        i_last.assign(n_seq, -1);
        i_batch_beg.assign(n_seq, -1);
        i_batch_end.assign(n_seq, -1);

        verify_h.assign(n_seq, {});
        verify_h_rows.assign(n_seq, 0);

        n_batch_tgt = (int32_t) llama_n_batch(ctx_tgt);
        verify_row0.assign(n_seq, -1);

        // shared memory and chained heads keep the host rows
        if (!is_mem_shared && !chain_heads) {
            dev_rows = llama_set_embeddings_nextn_dev(ctx_tgt, true) && llama_set_embd_state_src(ctx_dft, ctx_tgt);
            if (!dev_rows) {
                llama_set_embd_state_src(ctx_dft, nullptr);
                llama_set_embeddings_nextn_dev(ctx_tgt, false);
            }
        }
        SPC_TRC("- target h rows on the device: %s\n", dev_rows ? "yes" : "no");
    }

    ~common_speculative_impl_draft_mtp() override {
        auto * ctx_dft = this->params.ctx_dft;
        if (dev_rows) {
            llama_set_embd_state_src(ctx_dft, nullptr);
            llama_set_embeddings_nextn_dev(this->params.ctx_tgt, false);
        }
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) backend_chains.size(); ++seq_id) {
            if (backend_chains[seq_id] == nullptr) {
                continue;
            }
            if (ctx_dft) {
                llama_set_sampler(ctx_dft, seq_id, nullptr);
            }
            llama_sampler_free(backend_chains[seq_id]);
        }
        backend_chains.clear();
    }

    // [QLH-SPLIT] The one way to start a sequence's draft KV from empty.
    //
    // Four things have to move together or the route breaks silently: the memory itself, the
    // position the next draft() must see, the pos0 the last draft() ran at, and the position the
    // next catch-up batch must start at. Clearing the memory without the bookkeeping makes the
    // next draft() believe it still has history; the bookkeeping without the clear leaves another
    // token's rows readable at these positions, which is the one thing the route forbids.
    //
    // Six sites need exactly this (the request boundary, three failure paths in the catch-up
    // decode, a refused trim, and the "pos0 moved" fallback). It is a function so that a seventh
    // site -- the next rollback someone adds -- cannot forget one of the four.
    void keep_kv_reset(llama_seq_id seq_id) {
        auto * ctx_dft = this->params.ctx_dft;
        if (ctx_dft != nullptr) {
            llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, 0, -1);
        }
        if (seq_id >= 0 && seq_id < (llama_seq_id) keep_kv_expected.size()) {
            keep_kv_expected[seq_id] = -1;
        }
        if (seq_id >= 0 && seq_id < (llama_seq_id) keep_kv_pos0.size()) {
            keep_kv_pos0[seq_id] = -1;
        }
        if (seq_id >= 0 && seq_id < (llama_seq_id) catchup_pos.size()) {
            catchup_pos[seq_id] = -2;
        }
    }

    bool keeps_draft_kv() const override {
        return keep_kv;
    }

    std::string keep_draft_kv_stats() const override {
        if (!keep_kv) {
            return {};
        }
        std::string s = string_format("%zu round(s) continued from history, %zu started empty",
                keep_kv_rounds, keep_kv_resets);
        if (catchup_enabled) {
            // chain_catchup is the mode of the most recent request; the counters are cumulative,
            // so a run that ended in mode 0 still reports the rows an earlier request wrote.
            s += string_format("; catch-up mode %d (last request) wrote %zu row(s), %zu fallback(s)",
                    chain_catchup, catchup_rows, catchup_fallbacks);
        }
        return s;
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        // [QLH-SPLIT] v8: a remote chain has no draft sampler here and no draft KV to inspect;
        // nothing in this hook applies. (The stage clears its own draft memory per round.)
        if (is_remote) {
            return;
        }

        auto * ctx_dft = this->params.ctx_dft;

        // [QLH-SPLIT] chain catch-up: pick this request's mode before the prompt goes through
        // process(), because that is the window the route fills. Precedence, highest first:
        //   1. an explicit QLH_CHAIN_DRAFT_CATCHUP (0/1/2) -- the manual override, untouched;
        //   2. QLH_CHAIN_DRAFT_CATCHUP_AUTO, written per request by the server's batch_building
        //      from N_prompt and n_predict (see the field comment for the thresholds);
        //   3. off, i.e. what an unset variable meant before this round.
        // A mode that needs keep_kv but has none cannot reach this point: catchup_enabled is
        // derived from keep_kv, and the conflict is reported once in the constructor. Keeping a
        // second check here would be unreachable code that still looks authoritative.
        if (catchup_enabled) {
            int mode = catchup_mode_env();
            if (mode < 0) {
                mode = catchup_mode_auto();
            }
            if (mode < 0) {
                mode = 0;
            }
            chain_catchup     = mode;
            catchup_mode[seq_id] = (uint8_t) mode;
            if (mode > 0) {
                // this request's first process() batch(es) are its prompt: that is the window
                // mode 1 replays, and draft() closes it by consuming this flag.
                catchup_pending[seq_id] = 1;
                catchup_pos    [seq_id] = -2;
            } else {
                catchup_pending[seq_id] = 0;
            }
        }

        // [QLH-SPLIT] keep-KV route: a request boundary is the one place that must clear. Rows
        // the *previous* request left at the same positions are another token's state and would
        // be read silently -- that is exactly why the per-round clear existed (see draft()), and
        // clearing here instead is what makes the per-round clear unnecessary. Same positions,
        // same seq: nothing else in the route keeps rows alive across requests.
        if (keep_kv) {
            keep_kv_reset(seq_id);
        }

        // reset here rather than per round, or two identical requests differ
        common_sampler_reset(smpls[seq_id].get());

        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }

        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);

        if (pos_max < N - 1 && !is_mem_shared && !is_chain) {
            SPC_WRN("ctx_dft pos_max=%d < N-1=%d - "
                    "process() hook may not have run on every prefill ubatch "
                    "(need_embd / output flag on every prompt position?). "
                    "Drafts may degrade.\n",
                    (int) pos_max, N - 1);
        }
    }

    // [QLH-SPLIT] Chain catch-up, step 1: which sequences this batch writes rows for.
    //
    // The field comment above has the route. Returns the number of sequences left in `want`; a
    // sequence is dropped here (and its draft KV reset) when its position continuity no longer
    // holds, because the rows below would then be another token's state at these positions --
    // exactly what the route forbids.
    int32_t catchup_select(const common_batch & batch_in, std::vector<uint8_t> & want) {
        int32_t n_want = 0;
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_batch_beg[seq_id] < 0) {
                continue;
            }
            if (catchup_mode[seq_id] == 0) {
                continue;
            }
            if (catchup_mode[seq_id] == 1 && !catchup_pending[seq_id]) {
                continue;
            }
            want[seq_id] = 1;
            n_want++;
        }
        if (n_want == 0) {
            return 0;
        }

        // the rows below must still be this request's: a whole-frame rollback, a prompt that
        // moved, or a rebuilt chain all break pos continuity, and inheriting "the token that
        // used to be here" is exactly what keep_kv's expected-pos check exists to prevent.
        //
        // Only the prefill window is checked. In mode 2 a generation batch starts at the previous
        // round's pos0, which is *below* where that round's batch ended (the rejected drafts are
        // rewritten at the same positions), so a strict "starts where the last one ended" test
        // would fire on every round. There the batch's own positions are the tail's positions --
        // rows are rewritten at pos_beg, never appended -- and keep_kv's expected-pos check still
        // guards the case that matters (a pos0 that moved for any reason clears the sequence).
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (!want[seq_id]) {
                continue;
            }
            if (catchup_mode[seq_id] == 2 && !catchup_pending[seq_id]) {
                continue;
            }
            const llama_pos pos_beg = batch_in.tokens[i_batch_beg[seq_id]].pos[0];
            if (catchup_pos[seq_id] != -2 && catchup_pos[seq_id] != pos_beg) {
                SPC_DBG("catch-up: seq %d starts at pos %d, expected %d; dropping its draft KV\n",
                        (int) seq_id, (int) pos_beg, (int) catchup_pos[seq_id]);
                // keep_kv_reset() puts catchup_pos back to -2 as well, so the next batch starts
                // from an empty KV instead of failing the same continuity check again.
                keep_kv_reset(seq_id);
                want[seq_id] = 0;
                n_want--;
                catchup_fallbacks++;
            }
        }
        return n_want;
    }

    // [QLH-SPLIT] Chain catch-up, step 2: write the rows the tail just decoded into the draft KV.
    //
    // Shaped like the non-chain branch in process() (same pairing, same positions) but fed from
    // the tail's own buffer. Returns false when the draft decode failed; the caller must not fail
    // the request on it -- a draft is an optimization, and the sequence is left with an empty
    // draft KV, i.e. exactly what the keep_kv route saw before its first accepted token.
    bool catchup_write(const common_batch & batch_in, const std::vector<uint8_t> & want) {
        auto * ctx_dft = this->params.ctx_dft;
        auto * mem_dft = llama_get_memory(ctx_dft);

        const float * h_tgt = llama_get_embeddings_nextn(this->params.ctx_tgt);
        const int32_t n_tokens = (int32_t) batch_in.size();
        if (h_tgt == nullptr || n_tokens <= 0) {
            SPC_WRN("catch-up: the tail produced no h nextn rows (n_tokens=%d); "
                    "the draft starts empty for this sequence\n", n_tokens);
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (want[seq_id]) {
                    keep_kv_reset(seq_id);
                }
            }
            catchup_fallbacks++;
            return false;
        }

        batch.clear();

        // same pairing as the non-chain branch: a row carries the token at its own position and
        // the h of the *previous* row; the first row of a sequence takes pending_h, the h the
        // previous batch ended on (that row is on the other side of the batch boundary).
        std::vector<int32_t> prev_row(n_seq, -1);
        for (int k = 0; k < n_tokens; ++k) {
            const auto & t = batch_in.tokens[k];
            if (t.seq_id < 0 || t.seq_id >= (llama_seq_id) n_seq || !want[t.seq_id]) {
                continue;
            }
            const int32_t idx = t.id != LLAMA_TOKEN_NULL
                ? batch.add(t.id, t.pos[0], t.seq_id, false)
                : batch.add_embd(t.embd, t.pos.data(), t.seq_id, false);
            const float * h_row = prev_row[t.seq_id] < 0
                ? pending_h[t.seq_id].data()
                : h_tgt + (size_t) prev_row[t.seq_id] * n_embd;
            batch.set_embd_state(idx, { h_row, 1, (size_t) n_embd });
            prev_row[t.seq_id] = k;
        }

        // everything from this batch's first position on is being rewritten: in mode 2 the
        // previous round's drafts sit exactly there (keep_kv keeps them, and the server skips its
        // trim so that history survives). Writing a position twice would make the draft attend to
        // it twice, because the KV allocates a fresh cell and the old row keeps its sequence.
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (!want[seq_id]) {
                continue;
            }
            llama_memory_seq_rm(mem_dft, seq_id, batch_in.tokens[i_batch_beg[seq_id]].pos[0], -1);
        }

        // A chain is single-head by construction: chain_heads is false whenever is_chain (the
        // ctor derives it that way), so the multi-head rebuild the non-chain branch does here
        // cannot happen -- it needs the prompt's draft KV, which is exactly what this route
        // exists to skip. One decode, at the default layer offset.
        const int32_t rc = llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch.get());
        if (rc != 0) {
            SPC_WRN("catch-up: llama_process(ctx_dft) failed rc=%d; "
                    "the draft starts empty for this sequence\n", (int) rc);
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (want[seq_id]) {
                    keep_kv_reset(seq_id);
                }
            }
            catchup_fallbacks++;
            return false;
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (!want[seq_id]) {
                continue;
            }
            // the next draft() starts one past the last row written. In the prefill window this
            // route is what established the history, so it states the expectation itself. In a
            // generation batch (mode 2) accept() owns keep_kv_expected: it is the only hook that
            // knows how many of the rows this batch carried the tail actually kept, and the batch
            // ends past that point whenever a draft was rejected.
            const llama_pos pos_end = batch_in.tokens[i_batch_end[seq_id]].pos[0];
            if (catchup_pending[seq_id]) {
                keep_kv_expected[seq_id] = pos_end + 1;
            }
            catchup_pos[seq_id] = pos_end + 1;
            catchup_rows += (size_t) (i_batch_end[seq_id] - i_batch_beg[seq_id] + 1);
        }
        return true;
    }

    // [QLH-SPLIT] The two steps as one call, which is all process() needs: pick the sequences,
    // then write them. Split because the two halves fail differently -- "nobody wants a row" is a
    // no-op and returns true, "the decode failed" returns false -- and a single function had them
    // interleaved with three separate copies of the same KV reset.
    bool chain_catchup_decode(const common_batch & batch_in) {
        std::vector<uint8_t> want(n_seq, 0);
        if (catchup_select(batch_in, want) == 0) {
            return true;
        }
        return catchup_write(batch_in, want);
    }

    bool process(const common_batch & batch_in) override {
        if (batch_in.size() <= 0) {
            return true;
        }

        if (!batch_in.has_token() && !batch_in.has_embd()) {
            return true;
        }

        const int32_t n_tokens = batch_in.size();

        // remember the first and last batch index for each sequence
        std::fill(i_batch_beg.begin(), i_batch_beg.end(), -1);
        std::fill(i_batch_end.begin(), i_batch_end.end(), -1);

        for (int k = 0; k < n_tokens; ++k) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (batch_in.tokens[k].seq_id == seq_id) {
                    i_batch_end[seq_id] = k;
                    if (i_batch_beg[seq_id] < 0) {
                        i_batch_beg[seq_id] = k;
                    }
                }
            }
        }

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;

        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        // if kv is shared with target (e.g Gemma4), then we can skip this catch-up decode.
        //
        // [QLH-SPLIT] A chain skips it too, for a different reason. There the target batch is
        // fed to the tail as *embeddings* (decode_split hands the head's hidden rows over), and
        // the draft would have to be replayed token by token with h shifted right by one -- a
        // full extra draft pass over the prompt -- only to build history that draft() never
        // reads: draft() starts from a cleared draft KV (see is_chain there) and attends to the
        // k+1 positions it built in that round alone. So only the h rows are needed here, and
        // those come from the tail's own embeddings_nextn buffer (non-masked: one row per token
        // of the batch it just decoded, i.e. rows align 1:1 with the server batch indices).
        //
        // [QLH-SPLIT] v8: a *remote* chain skips it for the same reason and has no ctx_dft to run
        // it on anyway. Its h rows are the ones the tail sent back on the decode this hook follows
        // (llama_split_take_nextn_rows below), not an in-process buffer.
        if (!is_mem_shared && !is_chain && !is_remote) {
            batch.clear();

            // pair each token with the tgt embedding shifted right by one position, and
            // the first token of each sequence with the pending embedding from a previous run
            // assumes that the tokens in the batch are sequential for each sequence
            // i.e. we cannot have seq_id like this: [0, 0, 0, 1, 1, 0, 1, 1]
            //                                                       ^--- this is a problem
            // TODO:this is generally true, but would be nice to assert it
            if (dev_rows) {
                GGML_ASSERT(n_tokens <= n_batch_tgt && "the target batch exceeds the h_nextn device buffer");
                for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                    if (i_batch_beg[seq_id] >= 0) {
                        llama_set_embeddings_nextn_rows(ctx_tgt, n_batch_tgt + seq_id, 1, pending_h[seq_id].data());
                    }
                }
            }

            const float * h_tgt = dev_rows ? nullptr : llama_get_embeddings_nextn(ctx_tgt);

            for (int k = 0; k < n_tokens; ++k) {
                const auto & t = batch_in.tokens[k];

                const llama_seq_id seq_id = t.seq_id;

                // vision tokens carry an embedding instead of an id
                const int32_t idx = t.id != LLAMA_TOKEN_NULL
                    ? batch.add(t.id, t.pos[0], seq_id, false)
                    : batch.add_embd(t.embd, t.pos.data(), seq_id, false);

                if (dev_rows) {
                    batch.set_embd_state_row(idx, k == i_batch_beg[seq_id] ? n_batch_tgt + seq_id : k - 1);
                    continue;
                }

                const float * h_row = k == i_batch_beg[seq_id]
                    ? pending_h[seq_id].data()
                    : h_tgt + (size_t) (k - 1) * n_embd;

                batch.set_embd_state(idx, { h_row, 1, (size_t) n_embd });
            }

            auto * mem_dft = llama_get_memory(ctx_dft);

            bool ok = true;
            for (int head = 0; head < n_mtp_layers; ++head) {
                if (chain_heads) {
                    // ref: https://github.com/ggml-org/llama.cpp/pull/24340/changes#r3413498544
                    for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                        if (i_batch_beg[seq_id] < 0) {
                            continue;
                        }
                        llama_memory_seq_rm(mem_dft, seq_id, batch_in.tokens[i_batch_beg[seq_id]].pos[0], -1);
                    }
                    llama_set_nextn_layer_offset(ctx_dft, head);
                }

                const int32_t rc = llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch.get());
                if (rc != 0) {
                    SPC_ERR("llama_process(ctx_dft) head=%d failed rc=%d (pos=%d)\n",
                            head, (int) rc, (int) batch_in.tokens[0].pos[0]);
                    ok = false;
                    break;
                }
            }

            if (chain_heads) {
                llama_set_nextn_layer_offset(ctx_dft, 0); // restore default for non-draft decodes
            }
            if (!ok) {
                return false;
            }
        }

        // [QLH-SPLIT] chain catch-up (see catchup_mode): the local chain has the same h rows the
        // branch above reads, just on the tail's context, so the draft can be given the same
        // history the non-chain path gives it. A failure here degrades the draft to "no history"
        // and never fails the request -- and it has already cleared the sequence's draft KV, which
        // is all draft() needs to know to start empty.
        //
        // No per-seq pre-test here on purpose: chain_catchup_decode() starts by building its
        // `want` mask from catchup_mode and returns immediately when no sequence wants a row,
        // so a request that opted out costs one loop over n_seq per batch, nothing else.
        if (catchup_enabled) {
            chain_catchup_decode(batch_in);
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_batch_end[seq_id] < 0) {
                continue;
            }

            const int32_t n_rows = i_batch_end[seq_id] - i_batch_beg[seq_id] + 1;

            // [QLH-SPLIT] v8: remote chain -- the rows come from the wire. The tail answers them
            // for the rows it answered *for*: all of a verify frame, the last one of a plain
            // decode. So there are exactly two shapes to handle, and both are anchored on the
            // batch row the first returned row belongs to:
            //   - n_rows_got == n_rows  -> a verify frame; the whole array is usable and accept()
            //     can pick the accepted row out of it.
            //   - n_rows_got == 1 at the last row -> a plain decode; only pending_h can be filled,
            //     which is all a round needs (draft() starts from it).
            // Anything else leaves the previous pending_h alone; drafts then get verified like any
            // other, and a wrong draft costs speed, never correctness.
            if (is_remote) {
                const float *  rows_got  = nullptr;
                int32_t        row0      = -1;
                uint32_t       rows_embd = 0;
                const uint32_t n_rows_got = llama_split_take_nextn_rows(ctx_tgt, &rows_got, &row0,
                                                                        &rows_embd);
                if (n_rows_got == 0 || rows_got == nullptr || rows_embd != (uint32_t) n_embd) {
                    verify_h_rows[seq_id] = 0;
                    continue;
                }
                const int32_t wanted_last = i_batch_end[seq_id];
                if (n_rows_got == (uint32_t) n_rows && row0 == i_batch_beg[seq_id]) {
                    verify_h_rows[seq_id] = n_rows;
                    verify_h[seq_id].assign(rows_got, rows_got + (size_t) n_rows * n_embd);
                    std::memcpy(pending_h[seq_id].data(),
                            verify_h[seq_id].data() + (size_t) (n_rows - 1) * n_embd, row_bytes);
                } else if (n_rows_got == 1 && row0 == wanted_last) {
                    verify_h_rows[seq_id] = 0;
                    std::memcpy(pending_h[seq_id].data(), rows_got, row_bytes);
                } else {
                    verify_h_rows[seq_id] = 0;
                }
                continue;
            }

            if (is_mem_shared && !batch_in.tokens[i_batch_end[seq_id]].output) {
                verify_h_rows[seq_id] = 0;
                continue;
            }

            const bool all_rows_output = std::all_of(
                    batch_in.tokens.begin() + i_batch_beg[seq_id],
                    batch_in.tokens.begin() + i_batch_end[seq_id] + 1,
                    [](const common_batch::token & token) { return token.output; });
            const int32_t h_rows = is_mem_shared && !all_rows_output ? 1 : n_rows;
            const int32_t h_offset = is_mem_shared && !all_rows_output ? i_batch_end[seq_id] : i_batch_beg[seq_id];

            verify_h_rows[seq_id] = h_rows;

            if (dev_rows) {
                // the rows stay on the device until the next target decode, only the pending row comes to the host
                verify_row0[seq_id] = h_offset;
                llama_get_embeddings_nextn_rows(ctx_tgt, h_offset + h_rows - 1, 1, pending_h[seq_id].data());
                continue;
            }

            verify_h[seq_id].resize((size_t) h_rows * n_embd);

            for (int32_t i = 0; i < h_rows; ++i) {
                const float * h = llama_get_embeddings_nextn_ith(ctx_tgt, h_offset + i);
                std::memcpy(verify_h[seq_id].data() + (size_t) i * n_embd, h, row_bytes);
            }

            std::memcpy(pending_h[seq_id].data(),
                    verify_h[seq_id].data() + (size_t) (h_rows - 1) * n_embd, row_bytes);
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        // [QLH-SPLIT] v8: remote chain. There is no draft context here at all, so none of the
        // machinery below applies -- no batch, no draft KV to clear, no sampler. One call per
        // sequence and the whole block comes back, because the k steps are recursive inside the
        // stage (draft i's h is draft i-1's output) and only the round trip is being saved.
        //
        // The drafts are greedy. That is not a simplification: the local implementation's draft
        // sampler is top_k = 10 and it takes cur_p->data[0], i.e. the argmax, and the verify
        // channel these drafts are checked against is only armed for a chain that samples the
        // argmax (common_sampler_verify_rows_check). A probabilistic drafter would need q over the
        // whole vocabulary on the wire, which is the cost this layer exists to avoid.
        if (is_remote) {
            auto * ctx_tgt = this->params.ctx_tgt;

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                auto & dp = dparams[seq_id];

                if (!dp.drafting) {
                    continue;
                }

                // greedy: there is no q to report, and leaving a stale result_q would make the
                // caller run rejection against a distribution nothing produced.
                dp.result_q = nullptr;

                // [QLH-SPLIT] The draft block is the *configured* n_max, exactly like the local
                // loop's `params.n_max <= result.size()` test -- not dp.n_max. The server sets
                // dp.n_max to the largest block that fits the context (16 on a remote chain, see
                // get_n_draft_max), which is a capacity, not a preference: drafting 16 with a
                // modest acceptance rate costs more than it saves. MAX_MTP_DRAFT is the hard
                // ceiling either way, because a round has to fit one verify frame.
                const uint32_t k = (uint32_t) std::min<int32_t>(std::min(dp.n_max, params.n_max),
                        (int32_t) qlh_split::MAX_MTP_DRAFT);
                if (k == 0 || seq_id >= (llama_seq_id) pending_h.size() || dp.result == nullptr) {
                    continue;
                }

                std::vector<llama_token> ids(k);
                uint32_t n_out = 0;
                const int rc = llama_context_split_mtp_draft(ctx_tgt, pending_h[seq_id].data(),
                        (int32_t) dp.id_last, dp.pos0, k, seq_id, ids.data(), &n_out);

                // -1 is a transport failure and 0 is "this chain cannot draft". Both mean the same
                // thing *here*: no drafts this round, and the request decodes without speculation.
                // Neither is allowed to fail the request -- that is the error policy of the whole
                // remote layer, and a draft is exactly the kind of thing that must not violate it.
                if (rc <= 0 || n_out == 0) {
                    dp.result->clear();
                    continue;
                }

                dp.result->assign(ids.begin(), ids.begin() + n_out);

                if (dp.result->size() < (size_t) params.n_min) {
                    dp.result->clear();
                }
            }

            return;
        }

        batch.clear();

        // [QLH-SPLIT] chain catch-up, mode 1: the first draft() of a request is where the prompt
        // is known to be complete (the server only drafts from SLOT_STATE_DECODING), so the
        // prefill window closes here and later batches are left to keep_kv. Mode 2 replays every
        // batch and does not care about the flag -- but it is cleared for mode 2 as well, because
        // chain_catchup_decode() reads "pending" as "this is the request's first batch" and uses it
        // both for the position-continuity check and for setting keep_kv_expected (which accept()
        // owns from the second round on). Only sequences that asked for the route are touched.
        if (catchup_enabled) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (catchup_mode[seq_id] > 0) {
                    catchup_pending[seq_id] = 0;
                }
            }
        }

        if (is_chain && !is_mem_shared && !keep_kv) {
            // [QLH-SPLIT] v1's qlh_kh_mtp_reset, for the same reason it existed there: the
            // draft never saw the prompt (process() skips the catch-up), so a round must not
            // inherit rows from whatever the previous round / previous request left at the
            // same positions -- that is another token's state, and it would be read silently.
            // One clear per round costs nothing (the draft KV is k+1 rows per sequence) and is
            // far cheaper than replaying the prompt into the draft. Measured acceptance on the
            // 2B model stays in the 70-93% range v1 reported.
            // A KV-sharing assistant must not clear here: the KV it holds is the target's.
            //
            // [QLH-SPLIT] keep_kv replaces this clear with two narrower rules: the request
            // boundary clears (begin()), and every round trims to its own pos0 (below). Same
            // guarantee -- a round can never read another token's state -- but the rows this
            // request generated survive, and those were being dropped together with the prompt's.
            llama_memory_clear(llama_get_memory(ctx_dft), true);
        }

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }

            n_drafting++;
            drafting[seq_id] = true;

            if (keep_kv) {
                // [QLH-SPLIT] one round, one seq: trim the draft KV to this round's pos0. Everything
                // at or past pos0 is a rejected / never-verified draft row from an earlier round and
                // must go before the batch is built, or the batch's own consistency check would
                // refuse a position that goes backwards (iMROPE allows jumps forward, not back).
                auto * mem_dft = llama_get_memory(ctx_dft);

                const bool start_empty = keep_kv_expected[seq_id] < 0;
                if (!start_empty && dp.pos0 == keep_kv_expected[seq_id]) {
                    if (!llama_memory_seq_rm(mem_dft, seq_id, dp.pos0, -1)) {
                        // not fatal -- the draft is merely weaker without its history -- but it must
                        // not be silent: a KV that refuses the trim would keep rows the tail
                        // rejected, which is the one thing this route forbids. Once per process:
                        // a memory type that cannot trim would otherwise log this every round.
                        if (!keep_kv_trim_warned) {
                            keep_kv_trim_warned = true;
                            SPC_WRN("draft KV: seq_rm(seq %d, pos %d) failed; clearing the sequence "
                                    "instead (rounds draft without history; this is logged once)\n",
                                    (int) seq_id, (int) dp.pos0);
                        }
                        // per seq, not llama_memory_clear(): another slot may be mid-generation and
                        // its history is not this round's to throw away.
                        keep_kv_reset(seq_id);
                        keep_kv_resets++;
                    } else {
                        keep_kv_rounds++;
                        if (keep_kv_rounds == 1 && keep_kv_resets == 0) {
                            // one line per process, so "the route is on" and "the route is doing
                            // something" are distinguishable at INFO level
                            const llama_pos pmax = llama_memory_seq_pos_max(mem_dft, seq_id);
                            SPC_INF("draft KV: continuing from %d row(s) below pos0=%d "
                                    "(seq %d)%s\n",
                                    (int) pmax, (int) dp.pos0, (int) seq_id,
                                    catchup_mode.empty() || catchup_mode[seq_id] == 0
                                            ? "; the prompt is still absent"
                                            : "; the prompt rows are the tail's own");
                        }
                    }
                } else {
                    // Either the request just started, or the tail's position is not where this
                    // KV's history says it should be: a whole-frame rollback (grammar / reasoning
                    // budget), a context shift that slid the prompt down, or a rebuilt chain. The
                    // rows below pos0 are then another token's state at these positions -- exactly
                    // what must not be read -- so start empty. This fallback is what keeps the
                    // route correct without having to be told about every way pos0 can move.
                    if (!start_empty) {
                        SPC_DBG("draft KV: seq %d pos0=%d but expected %d; starting empty\n",
                                (int) seq_id, (int) dp.pos0, (int) keep_kv_expected[seq_id]);
                        keep_kv_resets++;
                    }
                    keep_kv_reset(seq_id);
                }

                keep_kv_pos0[seq_id] = dp.pos0;
            }

            // greedy drafting leaves no candidates behind, so the verifier falls back to sample-and-match
            if (!params.probabilistic) {
                dp.result_q = nullptr;
            }

            // result_q is only set when the caller wants rejection, so it also gates the retune
            if (dp.result_q) {
                spec_retune(smpls, smpls_cfg, llama_get_model(ctx_dft), seq_id, dp.temp, dp.seed);
            }

            // a reset reseeds the chain, which breaks probabilistic drafting
            if (!dp.result_q) {
                common_sampler_reset(smpls[seq_id].get());
            }

            const int32_t idx = batch.add(dp.id_last, dp.pos0, seq_id, true);
            batch.set_embd_state(idx, { pending_h[seq_id].data(), 1, (size_t) n_embd });

            i_last[seq_id] = idx;

            if (chain_heads) {
                chain_h[seq_id].assign(pending_h[seq_id].begin(), pending_h[seq_id].end());
            }
        }

        int i = 0;

        while (n_drafting > 0) {
            // each step decodes under a different head, i.e. a different decoder layer, and
            // KV is per layer. process() filled this layer's KV only for positions < pos0
            // (prompt + accepted prefix) — nothing in the draft region yet. so reset the
            // draft region (the seq_rm lower bound is pos0, leaving the prompt KV intact)
            // and select head i so it rebuilds its own layer's KV there; decoding just the
            // latest token would leave its attention reading cells only another head wrote.
            if (chain_heads) {
                auto * mem_dft = llama_get_memory(ctx_dft);
                for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                    if (drafting[seq_id]) {
                        llama_memory_seq_rm(mem_dft, seq_id, dparams[seq_id].pos0, -1);
                    }
                }
                llama_set_nextn_layer_offset(ctx_dft, i);
            }

            int ret = llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch.get());
            if (ret != 0) {
                SPC_ERR("llama_process[%d] returned %d\n", i, ret);
                break;
            }

            // rebuild the batch for the next step: the growing-KV paths re-add only the
            // new token (the KV already holds the prefix), while chained heads re-add the
            // whole prefix at the next head. dropped sequences are simply not re-added.
            batch.clear();

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                const llama_token id_sampled = common_sampler_sample(smpl, ctx_dft, i_last[seq_id], true);
                const float * h_row = llama_get_embeddings_nextn_ith(ctx_dft, i_last[seq_id]);

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    SPC_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                // add drafted token for each sequence
                const llama_token id = dparams.at(seq_id).result_q ? id_sampled : cur_p->data[0].id;

                // only collect very high-confidence draft tokens
                if (cur_p->data[0].p < params.p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                result.push_back(id);

                if (dp.result_q) {
                    dp.result_q->emplace_back(cur_p->data, cur_p->data + cur_p->size);
                }

                if (params.n_max <= (int) result.size()) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                if (chain_heads) {
                    // ref: https://github.com/ggml-org/llama.cpp/pull/24340#discussion_r3448031546
                    chain_h[seq_id].insert(chain_h[seq_id].end(), h_row, h_row + n_embd);

                    const int n_rows = (int) result.size() + 1; // id_last + tokens drafted so far
                    for (int t = 0; t < n_rows; ++t) {
                        const llama_token tok = (t == 0) ? dp.id_last : result[t - 1];
                        const int32_t idx = batch.add(tok, dp.pos0 + t, seq_id, t == n_rows - 1);
                        batch.set_embd_state(idx, { chain_h[seq_id].data() + (size_t) t * n_embd, 1, (size_t) n_embd });
                        i_last[seq_id] = idx;
                    }
                } else if (is_mem_shared) {
                    // note: with shared memory (e.g. Gemma4 assistants) we use the same position for all draft tokens
                    // ref: https://github.com/huggingface/transformers/blob/effde20942e3f82a1b97449f60b3a48c5ff96145/docs/source/en/model_doc/gemma4_assistant.md?plain=1#L36-L37
                    const int32_t idx = batch.add(id, dp.pos0, seq_id, true);
                    batch.set_embd_state(idx, { h_row, 1, (size_t) n_embd });
                    i_last[seq_id] = idx;
                } else {
                    const int32_t idx = batch.add(id, dp.pos0 + i + 1, seq_id, true);
                    batch.set_embd_state(idx, { h_row, 1, (size_t) n_embd });
                    i_last[seq_id] = idx;
                }
            }

            if (batch.size() == 0) {
                break;
            }

            ++i;
        }

        if (chain_heads) {
            llama_set_nextn_layer_offset(ctx_dft, 0); // restore default for non-draft decodes
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool /*is_other*/) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        // [QLH-SPLIT] keep-KV route: where the next round starts. The tail advances by the rows it
        // sampled itself (one, the position id_last was at) plus the drafts it accepted, so the
        // position id_last will be at is pos0 + n_accepted + 1. Computed here rather than read
        // back from the tail's memory: accept() is the only hook that knows how many rows this
        // revision actually kept, and it runs before any of them can be trimmed.
        if (keep_kv) {
            const llama_pos pos0 = keep_kv_pos0[seq_id];
            keep_kv_expected[seq_id] = pos0 < 0 ? -1 : pos0 + (llama_pos) n_accepted + 1;
        }

        const int32_t n_rows = verify_h_rows[seq_id];
        if (n_rows <= 0) {
            return;
        }

        const int32_t i_h = std::min<int32_t>(n_accepted, n_rows - 1);
        if (dev_rows) {
            // process() already fetched the last row
            if (i_h != n_rows - 1) {
                llama_get_embeddings_nextn_rows(params.ctx_tgt, verify_row0[seq_id] + i_h, 1, pending_h[seq_id].data());
            }
            return;
        }
        const size_t row_bytes = (size_t) n_embd * sizeof(float);
        std::memcpy(pending_h[seq_id].data(), verify_h[seq_id].data() + (size_t) i_h * n_embd, row_bytes);
    }

    bool get_state(llama_seq_id seq_id, std::vector<uint8_t> & data) const override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return false;
        }

        const size_t row_bytes = (size_t) n_embd * sizeof(float);
        data.resize(row_bytes);
        std::memcpy(data.data(), pending_h[seq_id].data(), row_bytes);
        return true;
    }

    void set_state(llama_seq_id seq_id, const std::vector<uint8_t> & data) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        const size_t row_bytes = (size_t) n_embd * sizeof(float);
        if (data.size() != row_bytes) {
            return;
        }
        std::memcpy(pending_h[seq_id].data(), data.data(), row_bytes);
    }
};

// state of self-speculation (simple implementation, not ngram-map)
struct common_speculative_impl_ngram_simple : public common_speculative_impl {
    common_params_speculative_ngram_map params;

    // shared across all sequences
    common_ngram_simple_config config;

    common_speculative_impl_ngram_simple(
            const common_params_speculative & params, uint32_t n_seq,
            common_ngram_simple_config config)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, n_seq, params.ngram_simple.size_m)
        , params(params.ngram_simple)
        , config(config)
    {
        SPC_TRC("%s", "adding speculative implementation 'ngram-simple'\n");
        SPC_TRC("- size_n=%d, size_m=%d, min_hits=%d\n",
                this->params.size_n, this->params.size_m, this->params.min_hits);
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    bool process(const common_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            *dp.result = common_ngram_simple_draft(config, *dp.prompt, dp.id_last);
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }
};

struct common_speculative_impl_ngram_map_k : public common_speculative_impl {
    // n_seq configs
    std::vector<common_ngram_map> config;

    common_speculative_impl_ngram_map_k(
            const common_ngram_map & config,
            uint32_t n_seq)
        : common_speculative_impl(config.key_only ? COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K
            : COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V, n_seq, config.size_value)
    {
        for (uint32_t i = 0; i < n_seq; i++) {
            this->config.push_back(config);
        }

        SPC_TRC("adding speculative implementation '%s'\n", common_speculative_type_to_str(this->type).c_str());
        SPC_TRC("- size_key=%d, size_value=%d, key_only=%d, min_hits=%d\n",
                config.size_key, config.size_value, config.key_only, config.min_hits);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        GGML_ASSERT(seq_id < (llama_seq_id) n_seq);

        common_ngram_map_begin(config[seq_id], prompt);
    }

    bool process(const common_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            common_ngram_map_draft(config[seq_id], *dp.prompt, dp.id_last, *dp.result);
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) override {
        GGML_ASSERT((seq_id < (llama_seq_id) config.size()));

        if (is_other) {
            return;
        }

        common_ngram_map_accept(config[seq_id], n_accepted);
    }
};

struct common_speculative_impl_ngram_mod : public common_speculative_impl {
    common_params_speculative_ngram_mod params;

    // shared across all sequences
    common_ngram_mod mod;

    // enable trace logging if LLAMA_TRACE is set
    const bool verbose;

    struct seq_info {
        // the last position in the prompt that was added to the ngram container
        size_t i_last = 0;

        // length of the last drafted n-gram (number of tokens returned by draft)
        size_t n_draft_last = 0;

        // consecutive accept rounds with low acceptance fraction (< 0.5)
        int n_low = 0;
    };

    std::vector<seq_info> sinfos;

    common_speculative_impl_ngram_mod(
            const common_params_speculative & params,
            uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_MOD, n_seq, params.ngram_mod.n_max)
        , params(params.ngram_mod)
        , mod(params.ngram_mod.n_match, 4*1024*1024)
        , verbose(std::getenv("LLAMA_TRACE") != nullptr) {
        static_assert(sizeof(llama_token) == sizeof(common_ngram_mod::entry_t));

        SPC_TRC("%s", "adding speculative implementation 'ngram-mod'\n");
        SPC_TRC("- n_match=%d, n_max=%d, n_min=%d\n",
                this->params.n_match, this->params.n_max, this->params.n_min);
        SPC_TRC("- mod size=%zu (%.3f MB)\n",
                mod.size(), (float)(mod.size_bytes())/1024/1024);

        if (this->params.n_match < 16) {
            SPC_WRN("ngram_mod n_match=%d is too small - poor quality is possible, "
                    "see: https://github.com/ggml-org/llama.cpp/pull/19164\n", this->params.n_match);
        }

        sinfos.resize(n_seq);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        auto & sinfo = sinfos[seq_id];

        sinfo.i_last = 0;
        sinfo.n_draft_last = 0;

        const size_t n = mod.get_n();
        if (prompt.size() < n) {
            return;
        }

        for (size_t i = 0; i < prompt.size() - n; ++i) {
            mod.add(prompt.data() + i);
        }

        sinfo.i_last = prompt.size() - n;

        const double f = (double)mod.get_used() / (double)mod.size();
        SPC_TRC("ngram_mod occupancy = %zu/%zu (%.2f)\n", mod.get_used(), mod.size(), f);

        constexpr double f_thold = 0.25;
        if (f > f_thold) {
            SPC_WRN("ngram_mod occupancy %.2f exceeds threshold (%.2f) - resetting\n", f, f_thold);

            mod.reset();
        }
    }

    void draft_one(
            llama_seq_id seq_id,
            common_speculative_draft_params & dparams) {
        auto & sinfo = sinfos[seq_id];
        auto & result = *dparams.result;

        const auto & prompt = *dparams.prompt;

        sinfo.n_draft_last = 0;

        const size_t cur_len = prompt.size();
        if (cur_len < mod.get_n()) {
            return;
        }

        const size_t n = mod.get_n();

        // add new ngrams in chunks
        if (sinfo.i_last + 32 < cur_len) {
            for (size_t i = sinfo.i_last; i < cur_len - n; ++i) {
                mod.add(prompt.data() + i);
            }

            sinfo.i_last = cur_len - n;
        }

        result.resize(n + params.n_max);
        for (size_t i = 0; i < n - 1; ++i) {
            result[i] = prompt.at(cur_len - n + 1 + i);
        }
        result[n - 1] = dparams.id_last;

        for (int i = 0; i < params.n_max; ++i) {
            const llama_token token = mod.get(result.data() + i);
            if (token == common_ngram_mod::EMPTY) {
                if (i < params.n_min) {
                    result.clear();
                    return;
                }

                result.resize(n + i);
                break;
            }
            result[n + i] = token;
        }

        // only return the m tokens that were drafted
        for (size_t i = 0; n + i < result.size(); ++i) {
            result[i] = result[n + i];
        }
        result.resize(result.size() - n);

        // store length of drafted n-gram for later acceptance analysis
        sinfo.n_draft_last = result.size();
    }

    bool process(const common_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            draft_one(seq_id, dp);
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) override {
        if (is_other) {
            return;
        }

        auto & sinfo = sinfos[seq_id];

        // compute acceptance fraction if we have a recorded draft length
        if (sinfo.n_draft_last > 0) {
            const double f_acc = (double)n_accepted / (double)sinfo.n_draft_last;
            if (f_acc < 0.25) {
                sinfo.n_low++;
                if (sinfo.n_low >= 5) {
                    if (verbose) {
                        SPC_TRC("low acceptance streak (%d) - resetting ngram_mod\n", sinfo.n_low);
                    }

                    mod.reset();
                    sinfo.n_low = 0;
                    sinfo.i_last = 0;
                }
            } else {
                sinfo.n_low = 0;
            }
        }
    }
};

struct common_speculative_impl_ngram_cache : public common_speculative_impl {
    common_params_speculative_ngram_cache params;

    uint16_t n_draft;

    struct seq_info {
        size_t cache_size = 0; // number of tokens in n-gram cache

        common_ngram_cache ngram_cache_context;
        common_ngram_cache ngram_cache_dynamic;
        common_ngram_cache ngram_cache_static;
    };

    std::vector<seq_info> sinfos;

    common_speculative_impl_ngram_cache(
            const common_params_speculative & params,
            uint32_t n_seq,
            uint16_t n_draft)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE, n_seq, n_draft)
        , params(params.ngram_cache)
        , n_draft(n_draft)
    {
        const auto & ngram = params.ngram_cache;

        SPC_TRC("%s", "adding speculative implementation 'ngram-cache'\n");
        SPC_TRC("- n_draft=%d, cache_static=%s, cache_dynamic=%s\n",
                n_draft,
                ngram.lookup_cache_static.empty() ? "none" : ngram.lookup_cache_static.c_str(),
                ngram.lookup_cache_dynamic.empty() ? "none" : ngram.lookup_cache_dynamic.c_str());

        sinfos.resize(n_seq);

        if (!ngram.lookup_cache_static.empty()) {
            try {
                auto ngram_cache_static = common_ngram_cache_load(ngram.lookup_cache_static);

                for (auto & sinfo : sinfos) {
                    sinfo.ngram_cache_static = ngram_cache_static;
                }
            } catch (const std::ifstream::failure &) {
                SPC_ERR("failed to open static lookup cache: %s\n", ngram.lookup_cache_static.c_str());
                GGML_ABORT("Couldn't read static lookup cache");
            }
        }

        if (!ngram.lookup_cache_dynamic.empty()) {
            try {
                auto ngram_cache_dynamic = common_ngram_cache_load(ngram.lookup_cache_dynamic);

                for (auto & sinfo : sinfos) {
                    sinfo.ngram_cache_dynamic = ngram_cache_dynamic;
                }
            } catch (const std::ifstream::failure &) {
                std::error_code ec;
                const bool exists = std::filesystem::exists(ngram.lookup_cache_dynamic, ec);
                if (!exists && !ec) {
                    // missing file is created on destruction
                    SPC_WRN("dynamic lookup cache file does not exist: %s, starting with empty cache\n",
                            ngram.lookup_cache_dynamic.c_str());
                } else {
                    SPC_ERR("failed to open dynamic lookup cache: %s\n", ngram.lookup_cache_dynamic.c_str());
                    GGML_ABORT("Couldn't read dynamic lookup cache");
                }
            }
        }
    }

    ~common_speculative_impl_ngram_cache() override {
        if (params.lookup_cache_dynamic.empty() || sinfos.empty()) {
            return;
        }

        auto & dynamic_cache = sinfos.front().ngram_cache_dynamic;
        for (auto & sinfo : sinfos) {
            common_ngram_cache_merge(dynamic_cache, sinfo.ngram_cache_context);
        }

        SPC_INF("saving dynamic lookup cache at %s\n", params.lookup_cache_dynamic.c_str());
        common_ngram_cache_save(dynamic_cache, params.lookup_cache_dynamic);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & /*prompt*/) override {
        sinfos[seq_id].cache_size = 0;
    }

    void draft_one(
            llama_seq_id seq_id,
            common_speculative_draft_params & dparams) {
        auto & sinfo = sinfos[seq_id];
        auto & result = *dparams.result;

        const auto & prompt = *dparams.prompt;

        llama_tokens inp;
        inp.reserve(prompt.size() + 1);
        for (size_t j = 0; j < prompt.size(); ++j) {
            inp.push_back(prompt[j]);
        }
        inp.push_back(dparams.id_last);

        if (sinfo.cache_size < inp.size()) {
            // Update context ngram cache with the full history so n-grams
            // ending at the new tokens have their preceding context
            common_ngram_cache_update(
                    sinfo.ngram_cache_context,
                    LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                    inp, inp.size() - sinfo.cache_size, false);
            sinfo.cache_size = inp.size();
        }

        result.push_back(dparams.id_last);

        common_ngram_cache_draft(
                inp, result, n_draft, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                sinfo.ngram_cache_context,
                sinfo.ngram_cache_dynamic,
                sinfo.ngram_cache_static);

        if (result.size() > 0) {
            // delete first token in result (which is the id_last token)
            result.erase(result.begin());
        }
    }

    bool process(const common_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            draft_one(seq_id, dp);
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }
};

struct common_speculative {
    common_speculative_draft_params_vec dparams;

    // list of implementations to use and their states
    std::vector<std::unique_ptr<common_speculative_impl>> impls;

    // which implementaion was used for a given seq_id
    std::vector<common_speculative_impl *> impl_last;

    std::vector<double> synth_probs;
};

static common_ngram_map get_common_ngram_map(
        common_speculative_type type,
        const common_params_speculative_ngram_map & config) {
    uint16_t size_key   = config.size_n;
    uint16_t size_value = config.size_m;
    bool     key_only   = type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K;
    uint16_t min_hits   = config.min_hits;

    return common_ngram_map(size_key, size_value, key_only, min_hits);
}

static std::unique_ptr<common_speculative_impl_ngram_cache> create_state_ngram_cache(
        const common_speculative_config & config,
        uint32_t n_seq) {
    uint16_t n_draft = 8; // TODO get from config?

    return std::make_unique<common_speculative_impl_ngram_cache>(config.params, n_seq, n_draft);
}

std::string common_speculative_type_name_str(const std::vector<common_speculative_type> & types) {
    std::string result;

    for (size_t i = 0; i < types.size(); i++) {
        if (i > 0) {
            result += ",";
        }
        result += common_speculative_type_to_str(types[i]);
    }
    return result;
}

const char * common_speculative_all_types_str() {
    static std::string all_types_str = []() {
        std::vector<common_speculative_type> types;
        types.reserve(COMMON_SPECULATIVE_TYPE_COUNT);
        for (int i = 0; i < COMMON_SPECULATIVE_TYPE_COUNT; i++) {
            types.push_back((common_speculative_type) i);
        }
        return common_speculative_type_name_str(types);
    }();
    return all_types_str.c_str();
}

std::string common_speculative_type_to_str(common_speculative_type type) {
    switch (type) {
        case COMMON_SPECULATIVE_TYPE_NONE:          return "none";
        case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE:  return "draft-simple";
        case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3:  return "draft-eagle3";
        case COMMON_SPECULATIVE_TYPE_DRAFT_MTP:     return "draft-mtp";
        case COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH:  return "draft-dflash";
        case COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK:  return "draft-dspark";
        case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:  return "ngram-simple";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:   return "ngram-map-k";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: return "ngram-map-k4v";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:     return "ngram-mod";
        case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:   return "ngram-cache";
        default:                                    return "unknown";
    }
}

std::vector<common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names) {
    std::vector<common_speculative_type> types;
    types.reserve(names.size());

    for (const auto & name : names) {
        auto type = common_speculative_type_from_name_map.find(name);
        if (type != common_speculative_type_from_name_map.end()) {
            if (type->second == COMMON_SPECULATIVE_TYPE_NONE) {
                return std::vector<common_speculative_type> { COMMON_SPECULATIVE_TYPE_NONE };
            }
            types.push_back(type->second);
            continue;
        }
        throw std::invalid_argument("unknown speculative type: " + name);
    }

    return types;
}

common_speculative_type common_speculative_type_from_name(const std::string & name) {
    const auto it = common_speculative_type_from_name_map.find(name);
    if (it == common_speculative_type_from_name_map.end()) {
        return COMMON_SPECULATIVE_TYPE_COUNT;
    }
    return it->second;
}

std::vector<common_speculative_type> common_speculative_types_from_gguf(const std::string & path) {
    struct gguf_init_params gguf_params = {
        /* .no_alloc = */ true,
        /* .ctx      = */ nullptr,
    };

    gguf_context_ptr gguf_ctx(gguf_init_from_file(path.c_str(), gguf_params));
    if (!gguf_ctx) {
        return {};
    }

    const int64_t arch_id = gguf_find_key(gguf_ctx.get(), "general.architecture");
    if (arch_id < 0 || gguf_get_kv_type(gguf_ctx.get(), arch_id) != GGUF_TYPE_STRING) {
        return {};
    }

    const std::string arch = gguf_get_val_str(gguf_ctx.get(), arch_id);
    if (arch != "dflash") {
        const uint32_t block_count = gguf_get_val_u32(gguf_ctx.get(), gguf_find_key(gguf_ctx.get(), (arch + ".block_count").c_str()));

        if (gguf_find_tensor(gguf_ctx.get(), ("blk." + std::to_string(block_count - 1) + ".nextn.eh_proj.weight").c_str()) >= 0) {
            return { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
        }

        return {};
    }

    // the Markov head distinguishes draft-dspark from draft-dflash
    const auto type = gguf_find_tensor(gguf_ctx.get(), "markov_w1.weight") >= 0
                    ? COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK
                    : COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH;

    SPC_INF("auto-detected speculative type '%s' from the draft model metadata\n", common_speculative_type_to_str(type).c_str());

    return { type };
}

static uint32_t common_get_enabled_speculative_configs(const std::vector<common_speculative_type> & configs) {
    uint32_t result = 0;
    for (size_t i = 0; i < configs.size(); i++) {
        result |= (1u << configs[i]);
    }
    return result;
}

int32_t common_speculative_n_max(const common_params_speculative * spec) {
    int32_t n_max = 0;

    for (const auto type : spec->types) {
        switch (type) {
            case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE:
            case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3:
            case COMMON_SPECULATIVE_TYPE_DRAFT_MTP:
            case COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH:
            case COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK:
                n_max = std::max(n_max, std::max(0, spec->draft.n_max));
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:
                n_max = std::max(n_max, (int32_t) spec->ngram_simple.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:
                n_max = std::max(n_max, (int32_t) spec->ngram_map_k.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V:
                n_max = std::max(n_max, (int32_t) spec->ngram_map_k4v.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:
                n_max = std::max(n_max, std::max(0, spec->ngram_mod.n_max));
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:
                n_max = std::max(n_max, (int32_t) 8);
                break;
            case COMMON_SPECULATIVE_TYPE_NONE:
            case COMMON_SPECULATIVE_TYPE_COUNT:
                break;
        }
    }

    return n_max;
}

int32_t common_speculative_n_max(const common_speculative * spec) {
    int32_t n_max = 0;

    if (spec == nullptr) {
        return n_max;
    }

    for (const auto & impl : spec->impls) {
        n_max = std::max(n_max, std::max(0, impl->n_max));
    }

    return n_max;
}

std::vector<double> common_speculative_synth_rates_resolve(const common_params_speculative * spec, int32_t n_max) {
    const bool has_length = spec->synth_len != -1.0;
    const bool has_rates  = !spec->synth_rates.empty();

    if (!has_length && !has_rates) {
        return {};
    }
    if (has_length && has_rates) {
        throw std::invalid_argument("synthetic acceptance length and rates are mutually exclusive");
    }

    if (n_max <= 0) {
        throw std::invalid_argument("synthetic acceptance requires at least one speculative token");
    }

    if (has_rates) {
        const auto & rates = spec->synth_rates;
        if (rates.size() != (size_t) n_max) {
            throw std::invalid_argument(string_format(
                    "synthetic acceptance rates must contain %d values, got %zu", n_max, rates.size()));
        }

        for (size_t i = 0; i < rates.size(); ++i) {
            if (!std::isfinite(rates[i]) || rates[i] < 0.0 || rates[i] > 1.0) {
                throw std::invalid_argument("synthetic acceptance rates must be finite and within [0, 1]");
            }
            if (i > 0 && rates[i] > rates[i - 1]) {
                throw std::invalid_argument("synthetic acceptance rates must be monotonically non-increasing");
            }
        }

        return rates;
    }

    const double length = spec->synth_len;
    const double length_max = (double) n_max + 1.0;
    if (!std::isfinite(length) || length < 1.0 || length > length_max) {
        throw std::invalid_argument(string_format(
                "synthetic acceptance length must be finite and within [1, %.0f]", length_max));
    }

    double p = 0.0;
    if (length == length_max) {
        p = 1.0;
    } else if (length > 1.0) {
        double p_min = 0.0;
        double p_max = 1.0;
        for (int i = 0; i < 32; ++i) {
            const double p_mid = 0.5 * (p_min + p_max);
            double sum = 0.0;
            double term = p_mid;
            for (int32_t j = 0; j < n_max; ++j) {
                sum += term;
                term *= p_mid;
            }

            if (sum < length - 1.0) {
                p_min = p_mid;
            } else {
                p_max = p_mid;
            }
        }
        p = 0.5 * (p_min + p_max);
    }

    std::vector<double> rates;
    rates.reserve(n_max);
    double rate = p;
    for (int32_t i = 0; i < n_max; ++i) {
        rates.push_back(rate);
        rate *= p;
    }

    return rates;
}

const std::vector<double> & common_speculative_get_synth_probs(const common_speculative * spec) {
    GGML_ASSERT(spec);
    return spec->synth_probs;
}

common_params common_base_params_to_speculative(const common_params & params) {
    const bool has_draft = params.speculative.has_dft();

    const auto & params_spec = params.speculative.draft;
    common_params result = params;

    // [QLH-SPLIT] A draft sidecar is a whole model of its own, not a stage of the chain: its
    // block count is its own (DFlash: 6, the target: 40) and the head's --split-layer-range
    // means nothing to it. Inheriting the range makes the loader look for blk.6..blk.31 in a
    // 6-block sidecar: "check_tensor_dims: tensor 'blk.6.attn_norm.weight' not found".
    //
    // The tail range is cleared for the same reason on the context side: a non-negative
    // split_tail_lo makes common_context_params_to_llama force embeddings + embeddings_pre_norm
    // on (that is the *chain stage* contract, see common/common.cpp), and a draft that emits
    // pre-norm hidden instead of logits samples garbage. A standalone load of the sidecar has
    // neither range, so this makes the draft context identical to one.
    result.split_layer_lo = -1;
    result.split_layer_hi = -1;
    result.split_tail_lo  = -1;
    result.split_tail_hi  = -1;
    // [QLH-SPLIT] v8: --split-remote belongs to the same list, and forgetting it was a crash, not
    // a degradation. A remote chain also forces embeddings + pre_norm on (same contract, the head
    // hands rows to the stage), so a draft context built from these params asks the MTP graph for
    // a t_embd it never builds: GGML_ASSERT in llm_graph_context::build_pooling, during the -fit
    // memory measurement -- i.e. before a single token, which is how it looked like "the server
    // will not start with --spec-type draft-mtp on a remote chain". The draft here is a stage of
    // another process; it has no role in this chain at all.
    result.split_remote.clear();

    result.embedding    = false;
    result.pooling_type = LLAMA_POOLING_TYPE_UNSPECIFIED;

    if (has_draft) {
        // default to global devices value
        if (!params_spec.devices.empty()) {
            result.devices           = params_spec.devices;
        }
        result.model                 = params_spec.mparams;
        result.n_gpu_layers          = params_spec.n_gpu_layers;
        result.tensor_buft_overrides = params_spec.tensor_buft_overrides;

        // a draft pinned to a single device doesn't need the meta wrapper an inherited -sm tensor would give it
        // (the device list is null-terminated, so a single device means size 2)
        const size_t n_devs = std::count_if(params_spec.devices.begin(), params_spec.devices.end(),
                [](ggml_backend_dev_t d) { return d != nullptr; });
        if (n_devs == 1) {
            result.split_mode = LLAMA_SPLIT_MODE_LAYER;
        }

        if (params_spec.cpuparams.n_threads > 0) {
            result.cpuparams.n_threads       = params_spec.cpuparams.n_threads;
            result.cpuparams_batch.n_threads = params_spec.cpuparams_batch.n_threads;
        }
    }

    result.cache_type_k  = params_spec.cache_type_k;
    result.cache_type_v  = params_spec.cache_type_v;
    result.n_outputs_max = params.n_parallel;
    result.n_outputs_max_per_seq = 1;

    if (params_spec.n_ubatch > 0) {
        result.n_ubatch = params_spec.n_ubatch;
    }

    // the MoE cache is only used by the target context
    result.moe_cache_size = 0;

    // dflash/dspark decode the whole noise block in a single pass and sample every block position on the backend
    // TODO: refactor such properties to be announced by the speculative types
    //       something like `struct common_speculative_type_props common_speculative_type_get_props(...);`
    const bool has_block_draft = std::any_of(
        params.speculative.types.begin(), params.speculative.types.end(),
        [](common_speculative_type t) {
            return t == COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH || t == COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK;
        });
    if (has_block_draft) {
        // per-seq output positions: DFlash decodes anchor + n_max masks (n_max + 1); DSpark n_max -> +1 covers both
        const int32_t per_seq = std::max(1, params_spec.n_max + 1);
        result.n_outputs_max = params.n_parallel * per_seq;
        if (params_spec.backend_sampling) {
            result.n_outputs_max_per_seq = per_seq;
        }
    }

    return result;
}

struct common_speculative_init_result::impl {
    impl() = default;
    ~impl() = default;

    // note: the order in which model, context, etc. are declared matters because their destructors will be called bottom-to-top
    llama_model_ptr   model;
    llama_context_ptr context;
};

common_speculative_init_result::common_speculative_init_result(
    common_params & params,
      llama_model * model_tgt,
    llama_context * ctx_tgt) :
    pimpl(new impl{}) {
    const bool has_draft = params.speculative.has_dft();
    const bool spec_mtp = std::find(params.speculative.types.begin(),
                                    params.speculative.types.end(),
                                    COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params.speculative.types.end();

    auto mparams = common_model_params_to_llama(params);
    auto cparams = common_context_params_to_llama(params);

    if (spec_mtp) {
        cparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    }

    // the draft context holds as many tokens per sequence as the target context
    cparams.n_ctx = llama_n_ctx(ctx_tgt);

    // note: for small models maybe we can set this to the maximum possible draft from all speculative types
    //       the extra memory for small models is likely negligible?
    cparams.n_rs_seq  = 0;
    cparams.ctx_other = ctx_tgt;

    // [QLH-SPLIT] chain mode: the MTP head lives in the tail segment, so model_tgt/ctx_tgt here
    // are the tail's. The draft context is not a stage of the chain -- it has its own KV (that
    // is what makes the MTP draft work at all), and it is created from `params` which describe
    // the *head* range. Two of the head's settings are actively wrong for it:
    //
    // This is MTP-only: DFlash is a draft model with no tok_embd/output.weight, and its ctor
    // needs ctx_other (the target context) to borrow them -- clearing ctx_other here makes
    // llama_init_from_model throw "DFlash requires ctx_other to be set" / return nullptr.
    const bool spec_chain = params.speculative.draft.ctx_tail != nullptr && spec_mtp;
    // [QLH-SPLIT] v8: a remote tail owns the MTP head, so there is no draft context to build in
    // this process at all. Building one here would be impossible anyway -- model_tgt is the head
    // range, which loads no NextN layers -- and the previous behaviour (refusing to start) is what
    // kept a remote chain on ngram drafts. ctx_tgt stays the head context (the chain driver, and
    // the only thing this process has); the drafts themselves go through
    // llama_context_split_mtp_draft().
    const bool spec_remote = params.speculative.draft.ctx_tail_remote && spec_mtp;
    if (spec_chain) {
        // 1. the shared params force the head's hidden export on (embeddings + pre-norm) since
        //    the head hands rows to the next stage. Inheriting that replaces the draft's logits
        //    with normed hidden rows and the draft samples garbage.
        cparams.embeddings          = false;
        cparams.embeddings_pre_norm = false;
        // 2. the shared params derive the output limits from the head's chain role; the draft
        //    needs one output row per drafted position instead (k+1 rows, one sequence each).
        cparams.n_outputs_max         = 0; // 0 = derive from n_batch
        cparams.n_outputs_max_per_seq = 0;
        cparams.n_seq_max             = llama_n_seq_max(ctx_tgt);
        // the draft decodes one row per sequence per step, but a round is k+1 positions long:
        // keep both batch sizes above that so a round is never split into two ubatches (a split
        // would decode a different KV layout than the draft loop assumes).
        cparams.n_batch  = std::max<int32_t>(cparams.n_batch,  params.speculative.draft.n_max + 1);
        cparams.n_ubatch = std::max<int32_t>(cparams.n_ubatch, params.speculative.draft.n_max + 1);
        // no ctx_other: an independent KV is the whole point, and leaving a pointer here is what
        // makes the MTP impl believe it shares memory with gemma4-style assistants.
        cparams.ctx_other = nullptr;

        LOG_INF("%s: split chain: MTP draft context over the tail model (%d layers, n_nextn = %d)\n",
                __func__, (int) llama_model_n_layer(model_tgt), (int) llama_model_n_layer_nextn(model_tgt));
    }

    std::string model_path;
    if (has_draft) {
        model_path = params.speculative.draft.mparams.path;
        LOG_INF("%s: loading draft model '%s'\n", __func__, model_path.c_str());

        llama_model * model_dft = common_model_load(params.speculative.draft.mparams.get_paths(), mparams);
        if (model_dft == NULL) {
            LOG_ERR("%s: failed to load draft model, '%s'\n", __func__, model_path.c_str());
            return;
        }

        pimpl->model.reset(model_dft);

        // [QLH-SPLIT] State the loaded shape explicitly: the whole point of clearing the split
        // range above is that the sidecar is a *whole* model (DFlash: 6 blocks), not the head's
        // [0, layer_hi) range. Seeing "6 blocks" here is the direct evidence.
        LOG_INF("%s: draft model: %d blocks, n_embd = %d (whole model, no split range)\n",
                __func__, (int) llama_model_n_layer(model_dft), (int) llama_model_n_embd(model_dft));

        llama_context * ctx_dft = llama_init_from_model(model_dft, cparams);
        if (ctx_dft == nullptr) {
            LOG_ERR("%s: failed to create MTP context\n", __func__);
            return;
        }

        pimpl->context.reset(ctx_dft);
    } else if (spec_mtp && !spec_remote) {
        model_path = params.model.path;

        LOG_INF("%s: creating MTP draft context against the target model '%s'\n", __func__, model_path.c_str());

        llama_context * ctx_dft = llama_init_from_model(model_tgt, cparams);
        if (ctx_dft == nullptr) {
            LOG_ERR("%s: failed to create MTP context\n", __func__);
            return;
        }

        pimpl->context.reset(ctx_dft);
    } else if (spec_remote) {
        // [QLH-SPLIT] v8: nothing to build here. The head range has no NextN block and the draft
        // context lives in the stage process, so this result carries no model and no context on
        // purpose -- and the caller must not treat that as a failure (it checks ctx_dft for a
        // *draft model*, which this is not).
        LOG_INF("%s: remote split chain: the MTP draft context lives in the tail stage; drafts "
                "cross the wire (k is capped at %u by the verify frame)\n",
                __func__, (unsigned) qlh_split::MAX_MTP_DRAFT);
    }
}

common_speculative_init_result::~common_speculative_init_result() = default;

llama_model * common_speculative_init_result::model() {
    return pimpl->model.get();
}

llama_context * common_speculative_init_result::context() {
    return pimpl->context.get();
}

common_speculative_init_result_ptr common_speculative_init_from_params(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt) {
    return std::make_unique<common_speculative_init_result>(params, model_tgt, ctx_tgt);
}

common_speculative_output_limits common_speculative_get_output_limits(
        int32_t n_batch, int32_t n_parallel, int32_t n_draft) {
    const int64_t per_seq = 1 + (int64_t) std::max(0, n_draft);
    const int64_t total   = (int64_t) n_parallel * per_seq;

    return {
        /* .total   = */ (int32_t) std::min<int64_t>(n_batch, total),
        /* .per_seq = */ (int32_t) std::min<int64_t>(n_batch, per_seq),
    };
}

// initialization of the speculative decoding system
//
common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq) {
    // Compute the implementations to use based on the config and their order of preference
    std::vector<common_speculative_config> configs = {}; // list of speculative configs to try
    {
        uint32_t enabled_configs = common_get_enabled_speculative_configs(params.types);

        auto add_config_if_enabled = [&](common_speculative_type type, bool available = true) {
            if (available && (enabled_configs & (1u << type))) {
                configs.emplace_back(type, params);
            }
        };

        // when adding a new type - update here the logic above
        static_assert(COMMON_SPECULATIVE_TYPE_COUNT == 11);

        // this list here defines the priority of the speculators
        // the one with highest priority are listed first
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_MOD);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE);

        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE);
        // [QLH-SPLIT] v8: a remote chain's MTP head has no local context at all (it runs in the
        // stage), so "ctx_dft != nullptr" is not the test for "MTP is available" -- ctx_tail_remote
        // is. Requiring the pointer here made the server log "no implementations specified for
        // speculative decoding" and silently serve without drafts.
        const bool mtp_available = params.draft.ctx_dft != nullptr || params.draft.ctx_tail_remote;
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3, params.draft.ctx_dft != nullptr);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_MTP,    mtp_available);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH, params.draft.ctx_dft != nullptr);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK, params.draft.ctx_dft != nullptr);
    }

    std::vector<std::unique_ptr<common_speculative_impl>> impls = {};

    for (const common_speculative_config & config : configs) {
        switch (config.type) {
            case COMMON_SPECULATIVE_TYPE_NONE:
                break;
            case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_simple>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_eagle3>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_MTP: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_mtp>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_dflash>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_dflash>(
                        config.params, n_seq, COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE: {
                common_ngram_map ngram_map = get_common_ngram_map(config.type, config.params.ngram_simple);

                uint16_t ngram_size_key   = ngram_map.size_key;
                uint16_t mgram_size_value = ngram_map.size_value;

                auto config_simple = common_ngram_simple_config {
                    /* .size_ngram = */ ngram_size_key,
                    /* .size_mgram = */ mgram_size_value
                };
                auto state = std::make_unique<common_speculative_impl_ngram_simple>(
                    /* .params = */ config.params,
                    /* .n_seq  = */ n_seq,
                    /* .state  = */ config_simple
                );
                impls.push_back(std::move(state));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_map_k>(
                            get_common_ngram_map(config.type, config.params.ngram_map_k), n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_map_k>(
                            get_common_ngram_map(config.type, config.params.ngram_map_k4v), n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_mod>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE: {
                impls.push_back(create_state_ngram_cache(config, n_seq));
                break;
            }
            default:
                break;
        }
    }

    if (impls.empty()) {
        SPC_TRC("%s", "no implementations specified for speculative decoding\n");
        return nullptr;
    }

    common_speculative_ptr result(new common_speculative {
        /* .dparams     = */ common_speculative_draft_params_vec(n_seq),
        /* .impls       = */ std::move(impls),
        /* .impl_last   = */ std::vector<common_speculative_impl *>(n_seq, nullptr),
        /* .synth_probs = */ {},
    });

    const int32_t n_max_configured = common_speculative_n_max(&params);
    const int32_t n_max_effective  = common_speculative_n_max(result.get());
    const auto rates = common_speculative_synth_rates_resolve(&params, n_max_effective);

    std::vector<std::string> rates_str;
    rates_str.reserve(rates.size());
    result->synth_probs.reserve(rates.size());
    double rate_prev = 1.0;
    double acceptance_length = 1.0;
    for (const double rate : rates) {
        result->synth_probs.push_back(rate_prev > 0.0 ? rate / rate_prev : 0.0);
        rates_str.push_back(string_format("%.6g", rate));
        rate_prev = rate;
        acceptance_length += rate;
    }
    if (!result->synth_probs.empty()) {
        SPC_WRN("%s", "synthetic speculative acceptance is enabled for benchmarking; generated output is not valid\n");
        if (n_max_effective != n_max_configured) {
            SPC_WRN("synthetic acceptance draft limit was reduced from %d to %d by the initialized speculative implementations\n",
                    n_max_configured, n_max_effective);
        }
        SPC_INF("synthetic acceptance: n_max = %zu, mean length = %.6f, rates = [%s]\n",
                rates.size(), acceptance_length, string_join(rates_str, ", ").c_str());
    }

    return result.release();
}

void common_speculative_free(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    delete spec;
}

common_speculative_draft_params & common_speculative_get_draft_params(
        common_speculative * spec,
        llama_seq_id seq_id) {
    GGML_ASSERT(spec);
    GGML_ASSERT(seq_id < (llama_seq_id) spec->dparams.size());

    return spec->dparams[seq_id];
}

void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        common_time_meas tm(impl->t_begin_us, !impl->gen_perf);
        impl->begin(seq_id, prompt);
        impl->n_call_begin++;
    }
}

bool common_speculative_process(common_speculative * spec, const common_batch & batch) {
    bool result = true;

    if (spec == nullptr) {
        return result;
    }

    for (auto & impl : spec->impls) {
        result = result && impl->process(batch);
    }

    return result;
}

void common_speculative_draft(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    auto & dparams = spec->dparams;

    {
        int n_drafting = 0;

        for (auto & dp : dparams) {
            GGML_ASSERT(!dp.drafting || dp.result->empty());

            if (dp.drafting) {
                n_drafting++;
            }
        }

        if (n_drafting == 0) {
            return;
        }
    }

    for (auto & impl : spec->impls) {
        {
            common_time_meas tm(impl->t_draft_us, !impl->gen_perf);
            impl->draft(dparams);
            impl->n_call_draft++;
        }

        int n_drafting = 0;

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) dparams.size(); ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }

            auto & result = *dp.result;

            // a new draft has been sampled
            if (dp.drafting && !result.empty()) {
                dp.drafting = false;

                if (dp.n_max > 0) {
                    if (!result.empty() && (int) result.size() > dp.n_max) {
                        SPC_DBG("truncating draft to %d tokens\n", dp.n_max);
                        result.resize(dp.n_max);

                        // trim the candidates only if the drafter produced them (n-gram drafters do not)
                        if (dp.result_q && !dp.result_q->empty()) {
                            dp.result_q->resize(dp.n_max);
                        }
                    }
                }

                if (!result.empty()) {
                    SPC_DBG("called impl %s, hist size = %zu, call_count = %zu, gen = %zu\n",
                            common_speculative_type_to_str(impl.get()->type).c_str(), dp.prompt->size(),
                            impl.get()->n_call_draft, result.size());

                    // remember which implementation was used
                    spec->impl_last[seq_id] = impl.get();

                    impl->n_gen_drafts++;
                    impl->n_gen_tokens += result.size();
                }
            }

            if (dp.drafting) {
                n_drafting++;
            }
        }

        if (n_drafting == 0) {
            break;
        }
    }

    // these sequences failed to generate a draft
    for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) dparams.size(); ++seq_id) {
        auto & dp = dparams[seq_id];

        if (dp.drafting) {
            dp.drafting = false;
        }
    }
}

void common_speculative_accept(common_speculative * spec, llama_seq_id seq_id, uint16_t n_accepted) {
    common_speculative_impl * impl = spec->impl_last[seq_id];

    if (impl == nullptr) {
        GGML_ASSERT(n_accepted == 0);
        return;
    }

    {
        common_time_meas tm(impl->t_accept_us, !impl->gen_perf);

        if (impl->n_acc_tokens_per_pos.size() < n_accepted) {
            impl->n_acc_tokens_per_pos.resize(n_accepted, 0);
        }

        for (size_t i = 0; i < n_accepted; ++i) {
            impl->n_acc_tokens_per_pos[i]++;
        }

        if (n_accepted > 0) {
            impl->n_acc_drafts++;
            impl->n_acc_tokens += n_accepted;
        }

        impl->accept(seq_id, n_accepted, false);
        impl->n_call_accept++;
    }

    // accept with the rest of the implementations, using is_other == true
    for (auto & impl_other : spec->impls) {
        if (impl_other.get() != impl) {
            impl_other->accept(seq_id, n_accepted, true);
        }
    }
}

// TODO: support the case of more than one speculative implementations having a state
bool common_speculative_get_state(common_speculative * spec, llama_seq_id seq_id, std::vector<uint8_t> & data) {
    if (spec == nullptr) {
        return false;
    }

    for (auto & impl : spec->impls) {
        if (impl->get_state(seq_id, data)) {
            return true;
        }
    }

    return false;
}

void common_speculative_set_state(common_speculative * spec, llama_seq_id seq_id, const std::vector<uint8_t> & data) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        impl->set_state(seq_id, data);
    }
}

void common_speculative_print_stats(const common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    for (const auto & impl : spec->impls) {
        std::string str_perf;
        if (impl->gen_perf) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3) << impl->t_begin_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_draft_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_accept_us / 1000.0;
            str_perf = ", dur(b,g,a) = " + oss.str() + " ms";
        } else {
            str_perf = "";
        }

        std::string str_stats;
        if (impl->n_call_accept > 0) {
            const double mean =
                1.0 + (double) impl->n_acc_tokens / (double) impl->n_call_accept;
            std::ostringstream tmp;
            tmp << std::fixed << std::setprecision(3);
            for (size_t i = 0; i < impl->n_acc_tokens_per_pos.size(); ++i) {
                if (i > 0) {
                    tmp << ", ";
                }
                tmp << (double) impl->n_acc_tokens_per_pos[i] / (double) impl->n_call_accept;
            }
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << mean;
            str_stats = ", #mean acc len = " + oss.str() + ", #acc rate/pos = (" + tmp.str() + ")";
        }

        SPC_TRC("statistics %16s: #calls(b,g,a) = %4zu %6zu %6zu, #gen drafts = %6zu, #acc drafts = %5zu, #gen tokens = %6zu, #acc tokens = %5zu%s%s\n",
                common_speculative_type_to_str(impl->type).c_str(),
                impl->n_call_begin, impl->n_call_draft, impl->n_call_accept,
                impl->n_gen_drafts,
                impl->n_acc_drafts,
                impl->n_gen_tokens,
                impl->n_acc_tokens,
                str_stats.c_str(),
                str_perf.c_str());
    }

    // [QLH-SPLIT] keep-KV chain route: how many rounds continued from history versus how many had
    // to start empty. A request that resets every round gets no benefit from the route, so this
    // line is the difference between "enabled" and "working".
    for (const auto & impl : spec->impls) {
        if (impl->keeps_draft_kv()) {
            SPC_INF("chain draft KV: %s\n", impl->keep_draft_kv_stats().c_str());
            break;
        }
    }
}

bool common_speculative_keeps_draft_kv(const common_speculative * spec) {
    if (spec == nullptr) {
        return false;
    }
    for (const auto & impl : spec->impls) {
        if (impl->keeps_draft_kv()) {
            return true;
        }
    }
    return false;
}
