#include "server-context.h"
#include <atomic>
#include <thread>
#include "base64.hpp"
#include "server-chat.h"
#include "server-common.h"
#include "server-decision.h"
#include "server-http.h"
#include "server-task.h"
#include "server-queue.h"
#include "server-realtime.h"
#include "server-schema.h"
#include "server-stream.h"

#include "build-info.h"
#include "common.h"
#include "fit.h"
#include "llama.h"
// [QLH-SPLIT] llama_context_split_failure()/llama_context_split_failed(): the staging API that
// says whether a remote chain is still trustworthy, so a failed decode can name the reason
// instead of only "Compute error.".
#include "../../src/llama-ext.h"
// [QLH-SPLIT] v2.3: the VRAM budget check, shared with the split-stage tail so both ends of a
// chain judge a segment the same way (see the header for why the check exists).
#include "../../src/qlh-vram-budget.h"
// [QLH-SPLIT] v2.4: the op_offload default a split stage runs with, shared with tools/split-stage
// so the head and its stages cannot disagree about the switch (see the header for the measurement).
#include "../../src/qlh-split-defaults.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "base64.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>   // [QLH-SPLIT] v7: INT32_MAX for the "no draft cap" sentinel
#include <cinttypes>
#include <deque>
#include <exception>
#include <memory>
#include <filesystem>
#include <random>
#include <utility>
#include <fstream>

// fix problem with std::min and std::max
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

constexpr int HTTP_POLLING_SECONDS = 1;

static common_speculative_output_limits server_output_limits(const common_params & params) {
    if (!params.mmproj.path.empty()) {
        const auto mcaps = mtmd_get_cap_from_file(params.mmproj.path.c_str());
        if (mcaps.gen_audio) {
            // some TTS models decode with embeddings output for the whole batch
            return { params.n_batch, params.n_batch };
        }
    }

    if (params.embedding ||
            (params.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED && params.pooling_type != LLAMA_POOLING_TYPE_NONE)) {
        return { params.n_batch, 1 };
    }

    auto result = common_speculative_get_output_limits(
            params.n_batch, params.n_parallel, common_speculative_n_max(&params.speculative));

    result.total   = std::max<int32_t>(1, result.total);
    result.per_seq = std::max<int32_t>(1, result.per_seq);
    return result;
}

// a checkpoint restore dropped tokens the target had accepted - re-accept them rather than verify again
static std::vector<llama_token> server_accept_replay(
        common_sampler * smpl,
        llama_context * ctx,
        const std::vector<int32_t> & idxs,
        const llama_tokens & draft) {
    GGML_ASSERT(idxs.size() == draft.size() + 1);

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    for (size_t i = 0; i < draft.size(); ++i) {
        // the token is discarded - the call is what advances the sampler over this position
        common_sampler_sample(smpl, ctx, idxs[i]);
        common_sampler_accept(smpl, draft[i], true);
        result.push_back(draft[i]);
    }

    const llama_token id = common_sampler_sample(smpl, ctx, idxs[draft.size()]);
    common_sampler_accept(smpl, id, true);
    result.push_back(id);

    return result;
}

// synthetic draft verification for benchmarking - accept draft tokens at random instead of by match with the target
// on replay the draft was already accepted before a context checkpoint restore, so repeat the same decisions
static std::vector<llama_token> server_sample_and_accept_synth(
        common_sampler * smpl,
        llama_context * ctx,
        const std::vector<int32_t> & idxs,
        const llama_tokens & draft,
        const std::vector<double> & synth_probs,
        std::mt19937 & rng,
        bool is_replay) {
    GGML_ASSERT(idxs.size() == draft.size() + 1);
    GGML_ASSERT(synth_probs.size() >= draft.size());

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (size_t i = 0; i < draft.size(); ++i) {
        const llama_token id = common_sampler_sample(smpl, ctx, idxs[i]);
        const bool accept = is_replay || dist(rng) < synth_probs[i];
        // do not accept a drafted EOG token - it would end the generation early
        // on replay the last token is from the target and can be EOG, so skip this check
        if (accept && (is_replay || !llama_vocab_is_eog(vocab, draft[i]))) {
            // synthetic draft tokens do not advance grammar or reasoning state
            // the last replay token is from the target and must advance both
            const bool is_replay_target = is_replay && i + 1 == draft.size();
            common_sampler_accept(smpl, draft[i], is_replay_target);
            result.push_back(draft[i]);
            continue;
        }

        common_sampler_accept(smpl, id, true);
        result.push_back(id);
        return result;
    }

    const llama_token id = common_sampler_sample(smpl, ctx, idxs[draft.size()]);
    common_sampler_accept(smpl, id, true);
    result.push_back(id);

    return result;
}

// state diagram: https://github.com/ggml-org/llama.cpp/pull/9283
enum slot_state {
    SLOT_STATE_IDLE,
    SLOT_STATE_WAIT_OTHER, // after assigning a task, but waiting for parent slot to process prompt
    SLOT_STATE_STARTED,    // after assigning a task and about to process prompt
    SLOT_STATE_PROCESSING_PROMPT,
    SLOT_STATE_DONE_PROMPT,
    SLOT_STATE_GENERATING,
};

struct server_slot; // forward declaration

struct server_batch {
    common_batch view; // the rendered sub-batch [off, off + n_tokens), see render()

    struct token {
        int32_t id_slot;
        llama_token token;
        std::array<llama_pos, GGML_MROPE_SECTIONS> pos; // only pos[0] is used for text tokens
        bool output;
        bool is_prompt; // for stats tracking
        int32_t decision_order = 0;
        int32_t i_embd = -1; // row in embd, -1 if this is a token
    };
    std::vector<token> tokens;
    int32_t n_tokens_alloc = 0;
    int32_t n_embd = 0;

    // track if given slot can be batched with slots already in the batch
    server_slot * slot_batched = nullptr;

    // embedding entries, they can be mixed with tokens if the context supports it
    std::vector<float> embd; // n_embd per row

    float  alora_scale       = -1.0f;
    size_t alora_disabled_id = 0;

    void init(llama_context * ctx, int32_t n_tokens_alloc, int32_t n_embd) {
        this->n_tokens_alloc = n_tokens_alloc;
        this->n_embd = n_embd;
        view = common_batch(ctx);
        tokens.reserve(n_tokens_alloc);
    }

    bool add(int32_t id_slot, llama_token token, llama_pos pos, bool output, bool is_prompt) {
        if ((int32_t)tokens.size() >= n_tokens_alloc) {
            return false;
        }
        tokens.push_back({ id_slot, token, { pos, 0, 0, 0 }, output, is_prompt });
        return true;
    }

    // embd_in has n_embd values, pos has GGML_MROPE_SECTIONS values
    bool add_embd(int32_t id_slot, const float * embd_in, const llama_pos * pos, bool output, bool is_prompt) {
        if ((int32_t)tokens.size() >= n_tokens_alloc) {
            return false;
        }
        tokens.push_back({ id_slot, LLAMA_TOKEN_NULL, { pos[0], pos[1], pos[2], pos[3] }, output, is_prompt });
        tokens.back().i_embd = (int32_t) (embd.size() / n_embd);
        embd.insert(embd.end(), embd_in, embd_in + n_embd);
        return true;
    }

    bool has_embd() const {
        return !embd.empty();
    }

    void clear() {
        tokens.clear();
        embd.clear();
        view.clear();
        slot_batched      = nullptr;
        alora_scale       = -1.0f;
        alora_disabled_id = 0;
    }

    int32_t size() const {
        return (int32_t)tokens.size();
    }

    // remove the entries after the first n
    void truncate(int32_t n) {
        GGML_ASSERT(n >= 0 && n <= size());
        for (int32_t i = n; i < size(); i++) {
            if (tokens[i].i_embd >= 0) {
                embd.resize((size_t) tokens[i].i_embd * n_embd);
                break;
            }
        }
        tokens.resize(n);
    }

    void set_output(int32_t idx, bool output) {
        GGML_ASSERT(idx >= 0 && idx < (int32_t)tokens.size());
        tokens[idx].output = output;
    }

    void set_decision_order(int32_t idx, int32_t order) {
        GGML_ASSERT(idx >= 0 && idx < (int32_t)tokens.size());
        tokens[idx].decision_order = order;
    }

    // render the sub-batch [off, off + n_tokens) into view, index i in view is index off + i here
    void render(int32_t off, int32_t n_tokens) {
        GGML_ASSERT(off >= 0 && off < size());
        GGML_ASSERT(n_tokens > 0 && off + n_tokens <= size());

        view.clear();
        for (int32_t i = off; i < off + n_tokens; i++) {
            const auto & t = tokens[i];
            if (t.i_embd >= 0) {
                view.add_embd({ embd.data() + (size_t) t.i_embd * n_embd, 1, (size_t) n_embd }, t.pos.data(), t.id_slot, t.output);
            } else {
                view.add(t.token, t.pos[0], t.id_slot, t.output);
            }
            view.tokens.back().decision_order = t.decision_order;
        }
    }
};

struct server_slot {
    int id;

    llama_context * ctx_tgt = nullptr;
    llama_context * ctx_dft = nullptr;

    common_memory mem;

    // multimodal
    mtmd_context * mctx = nullptr;
    mtmd::batch_ptr mbatch = nullptr;

    struct tts_ctx {
        mtmd_helper::gen_audio ctx;
        std::vector<float> h_state_prompt; // the first h_state after step_prompt done
        const float * h_state;
        llama_token   sampled;
        int32_t       n_decoded;
        bool          stop;
        bool is_supported() const {
            return ctx.valid();
        }
        void reset() {
            ctx.reset();
            h_state_prompt.clear();
            h_state   = nullptr;
            sampled   = LLAMA_TOKEN_NULL;
            n_decoded = 0;
            stop      = false;
        }
    };
    tts_ctx tts;

    // speculative decoding
    common_speculative * spec;

    llama_tokens spec_draft;

    // draft candidates per token in spec_draft; only draft-simple and draft-mtp fill it
    std::vector<std::vector<llama_token_data>> spec_draft_q;
    llama_tokens spec_prompt;
    std::vector<int32_t> spec_i_batch;
    common_prompt_checkpoint spec_ckpt;
    bool spec_is_replay = false;

    // [QLH-SPLIT] v8: exactly one round of plain decoding after a verify frame that produced no
    // token (the grammar rejected one of its rows).
    //
    // Without it the round repeats itself forever: the first row of the next frame samples the
    // same position again -- same KV, same token, same grammar state -- so it would be the same
    // grammar-rejected argmax, and no token would ever reach the client. One plain decode
    // advances the position, which is all that is needed: the next frame's first row is a
    // different position, and the row path's own grammar resampling picked its token.
    bool skip_draft_once = false;

    std::mt19937 spec_synth_rng;

    // TODO: move members that belong to the task (such as `generated`, `has_new_line`) to task_results_state
    //       see https://github.com/ggml-org/llama.cpp/pull/18283#issuecomment-3710175837
    std::unique_ptr<const server_task> task;
    std::unique_ptr<const server_task> task_prev; // used for debugging

    // used to determine the slot that has been used the longest
    int64_t t_last_used = -1;

    // generation props
    int32_t n_ctx   = 0;  // context size per slot
    int32_t n_keep  = 0;
    int32_t i_batch = -1;

    // effective generation limit for the current task, -1 means unlimited
    int32_t n_predict_max = -1;

    size_t last_nl_pos = 0;

    common_chat_input generated;
    std::string  debug_generated_text;
    llama_tokens generated_tokens;
    size_t n_sent_text = 0; // number of sent text character (i.e. handle partial UTF-8 on streaming)

    std::vector<completion_token_output> generated_token_probs;

    bool has_next_token = true;
    bool has_new_line   = false;
    bool truncated      = false;

    stop_type stop;

    std::string stopping_word;

    // state
    slot_state state = SLOT_STATE_IDLE;

    server_prompt prompt;

    bool prompt_save(server_prompt_cache & prompt_cache) const {
        if (prompt.tokens.size() == 0) {
            return false;
        }

        const size_t cur_size_tgt =           llama_state_seq_get_size_ext(ctx_tgt, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        const size_t cur_size_dft = ctx_dft ? llama_state_seq_get_size_ext(ctx_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE) : 0;

        const size_t cur_size = cur_size_tgt + cur_size_dft;

        // [QLH-SPLIT] Zero means this context cannot export its sequence state at all (a chain
        // with remote stages; see llama_context::state_seq_get_size), not that the state is
        // empty. An entry with no bytes would be restored "successfully" later and hand the slot
        // a prompt whose tokens the chain does not actually hold; the following truncating
        // seq_rm would then abort or, worse, decode from positions that never existed here.
        // Skipping the save keeps the prompt cache out of a chain it cannot describe.
        if (cur_size == 0) {
            return false;
        }

        SRV_TRC(" - saving prompt with length %d, total state size = %.3f MiB (draft: %.3f MiB)\n",
                (int) prompt.tokens.size(), cur_size / (1024.0 * 1024.0), cur_size_dft / (1024.0 * 1024.0));

        auto * cur = prompt_cache.alloc(prompt, cur_size_tgt, cur_size_dft);
        if (cur == nullptr) {
            return false;
        }

        const size_t n_tgt = llama_state_seq_get_data_ext(ctx_tgt, cur->data.main.data(), cur_size_tgt, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        const size_t n_dft = ctx_dft ? llama_state_seq_get_data_ext(ctx_dft, cur->data.drft.data(), cur_size_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE) : 0;
        if (n_tgt != cur_size_tgt || n_dft != cur_size_dft) {
            SLT_ERR(*this, "%s", "failed to save complete prompt state\n");
            prompt_cache.states.pop_back();
            return false;
        }

        return true;
    }

    bool prompt_load(server_prompt_cache & prompt_cache, const server_tokens & tokens) {
        bool res = prompt_cache.load(prompt, tokens, ctx_tgt, ctx_dft, id);
        if (!res) {
            SLT_WRN(*this, "%s", "failed to load prompt from cache\n");
        }

        return res;
    }

    void prompt_clear() {
        SLT_TRC(*this, "clearing prompt with %zu tokens\n", prompt.tokens.size());

        mem.seq_rm(id, -1, -1);

        prompt.clear();
    }

    std::vector<common_adapter_lora_info> lora;
    int32_t alora_invocation_start = -1;

    // sampling
    json json_schema;

    common_sampler_ptr smpl;

    // [QLH-SPLIT] v5: the remote candidate channel for *this* request (see common/sampling.h).
    //
    // Decided once, when the slot's sampler is created -- i.e. before the first decode of the
    // request -- and never re-decided per token: grammar, pre-sampling logprobs and the shape of
    // the sampler chain are all fixed for the request, and a per-token re-check could only turn a
    // request into a mix of the two paths. `remote_cand_why` is what was logged when the plan was
    // rejected, kept for the debug line.
    bool                      remote_cand_ok = false;
    common_sampler_candidates remote_cand;
    std::string               remote_cand_why;

    // The logit-bias list the tail must apply, copied out of the sampler at plan time. A copy and
    // not the sampler's pointer: this struct outlives the sampler -- a checkpoint restore copies
    // sampler state into it -- and a pointer into a vector that can be reassigned is exactly the
    // kind of stale state that would silently change which tokens are suppressed.
    std::vector<llama_logit_bias> remote_cand_bias;

    // [QLH-SPLIT] v7: the per-row verify channel for *this* request (see llama-ext.h and
    // common/sampling.h). Also decided when the sampler is created, from the same request-level
    // inputs, for the same reason: a decision per token could only mix the two paths.
    //
    // `remote_verify_ok` is true when this request's speculation can verify drafts through the
    // channel. `spec_blocked` is the other half, and it is the one that changes behaviour: a
    // remote chain can only speculate *through* the channel (there are no logits for the extra
    // rows), so a request that cannot use it must not speculate at all. Before this existed, that
    // case aborted the server inside common_sampler_sample (`GGML_ASSERT(logits != nullptr)`).
    //
    // `spec_draft_cap` bounds the draft block so that a verify frame fits MAX_VERIFY_ROWS rows.
    // INT32_MAX for a whole model and for a local chain, whose tail surfaces every row it is
    // asked for.
    bool    remote_verify_ok  = false;
    bool    spec_blocked      = false;
    int32_t spec_draft_cap    = INT32_MAX;
    std::string remote_verify_why;

    llama_token sampled; // in speculative mode, this is the last accepted token

    // for TTS models, this is the embd generated from prev step, decode this to generate next hidden state
    // corresponding to one token position (size = n_embd)
    // for realtime transcription, these are the rows from the last audio stream step (size = n_rows * n_embd)
    std::vector<float> inp_embd;

    // realtime transcription, see SERVER_TASK_TYPE_TRANSCRIBE_RT
    mtmd::audio_stream_ptr audio_stream;
    bool audio_wait = false; // the stream needs more audio before it can produce rows
    std::deque<int64_t> audio_commit_rows; // end row of each non-final commit not yet completed

    // realtime conversation, see SERVER_TASK_TYPE_REALTIME
    std::unique_ptr<server_realtime_slot> rt;

    server_slot_stats stats;

    // accepted tokens per draft position
    // not in server_slot_stats to avoid copying to every task result
    std::vector<uint64_t> n_accepted_per_pos;

    std::function<void(int /* id_slot */)>   callback_on_release;
    std::function<void(const server_slot &)> callback_on_reset; // called before reset()

    // this is for printing timings with slot progress, not part of metrics
    int64_t t_print_last = 0;
    int32_t n_gen_last = 0;

    void reset() {
        SLT_DBG(*this, "%s", "\n");

        spec_is_replay = false;

        last_nl_pos    = 0;
        generated      = {};
        has_new_line   = false;
        truncated      = false;
        stop           = STOP_TYPE_NONE;
        stopping_word  = "";
        n_sent_text    = 0;

        // [QLH-SPLIT] v7: test the module, not can_speculate(). The latter is now also false for a
        // request whose speculation was blocked, and "this request may not draft" is not "there is
        // no speculative state to clear".
        if (spec != nullptr) {
            spec_draft.clear();
            spec_i_batch.clear();
            spec_ckpt.clear();
            skip_draft_once = false;
        }
        generated_tokens.clear();
        generated_token_probs.clear();
        json_schema = json();

        task_prev = std::move(task);
        task.reset();

        // note: callback_on_reset() must have run before this, see release()
        stats = {};
        n_accepted_per_pos.clear();

        n_predict_max = -1;

        llama_set_sampler(ctx_tgt, id, nullptr);

        // clear alora start
        alora_invocation_start = -1;

        // clear multimodal state
        mbatch.reset();

        tts.reset();

        audio_stream.reset();
        audio_wait = false;
        audio_commit_rows.clear();
        inp_embd.clear();
        rt.reset();
    }

    void init_sampler() const {
        common_sampler_reset(smpl.get());

        if (!task->need_sampling()) {
            return;
        }

        const int64_t t_start = ggml_time_us();

        int n_text = 0;

        for (int i = 0; i < (int) prompt.tokens.size(); i++) {
            const llama_token id = prompt.tokens[i];

            if (id != LLAMA_TOKEN_NULL) {
                common_sampler_accept(smpl.get(), id, false);
                n_text++;
            }
        }

        SLT_TRC(*this, "init sampler, took %0.2f ms, tokens: text = %d, total = %d\n",
                (ggml_time_us() - t_start) / 1000.0, n_text, (int) prompt.tokens.size());
    }

    bool need_embd() const {
        GGML_ASSERT(task);
        return task->need_embd();
    }

    bool can_split() const {
        GGML_ASSERT(task);
        // a decision over a non-causal context reads the whole prompt in one batch
        if (task->type == SERVER_TASK_TYPE_DECISION && !llama_get_causal_attn(ctx_tgt)) {
            return false;
        }
        // MTP supports splitting - uses task->need_embd() not need_embd()
        if (!task->need_embd()) {
            return true;
        }
        // if the context does not have a memory module then all embeddings have to be computed within a single ubatch
        if (!llama_get_memory(ctx_tgt)) {
            return false;
        }
        // context can be chunked/split if the pooling type is LAST
        const auto pooling = llama_pooling_type(ctx_tgt);
        if (pooling == LLAMA_POOLING_TYPE_LAST) {
            return true;
        }
        // causal rerankers read the last token and have a KV cache, so they can also be chunked/split.
        if (pooling == LLAMA_POOLING_TYPE_RANK && llama_get_causal_attn(ctx_tgt)) {
            return true;
        }
        // a decision task reads its outputs from the last batch
        if (task->type == SERVER_TASK_TYPE_DECISION) {
            return true;
        }
        return false;
    }

    bool can_batch_with(server_slot & other_slot) const {
        GGML_ASSERT(task);

        // a joint decision head reads the whole batch
        if (!task->decision.order.empty() || !other_slot.task->decision.order.empty()) {
            return false;
        }

        // lfm3 d1 has no memory, its prompt is read as one flat ubatch and the short convs would mix the sequences
        // TODO: batch them along the sequence dimension, this needs ubatches of sequences with unequal lengths
        if (task->decision.no_batching || other_slot.task->decision.no_batching) {
            return false;
        }

        return task->type == other_slot.task->type
            && inp_embd.size() == other_slot.inp_embd.size()
            && are_lora_equal(lora, other_slot.lora);
    }

    // returns -1 if the generation is limitless
    int32_t n_remaining() const {
        return n_predict_max == -1 ? -1 : n_predict_max - (int32_t) stats.n_gen;
    }

    bool has_budget() const {
        return n_predict_max == -1 || n_remaining() > 0;
    }

    bool is_processing() const {
        return state != SLOT_STATE_IDLE;
    }

    bool can_speculate() const {
        // [QLH-SPLIT] v7: a remote chain can only verify drafts through the per-row channel, so a
        // request whose sampler cannot use it must not draft at all (see spec_blocked).
        return !!spec && !spec_blocked;
    }

    // at temp 0 both p and q are point masses, so rejection is the same as sample-and-match
    bool use_spec_rejection() const {
        return task && task->params.sampling.temp > 0.0f;
    }

    void add_token(const completion_token_output & token) {
        if (!is_processing()) {
            SLT_WRN(*this, "%s", "slot is not processing\n");
            return;
        }

        generated_token_probs.push_back(token);
    }

    int get_n_draft_max() const {
        GGML_ASSERT(task);

        if (!can_speculate()) {
            return 0;
        }

        // determine the max draft that fits the current slot state
        // note: slot.prompt is not yet expanded with the `id` token sampled above
        //       also, need to leave space for 1 extra token to allow context shifts
        int n_draft_max = n_ctx - prompt.n_tokens() - 2;

        if (n_remaining() > 0) {
            n_draft_max = std::min(n_draft_max, n_remaining() - 1);
        }

        // [QLH-SPLIT] v7: a verify frame is draft.size() + 1 rows and the tail answers all of
        // them through the per-row channel, whose ceiling is MAX_VERIFY_ROWS (a shared constant,
        // see src/llama-split-proto.h). So the draft block itself is capped here, once, rather
        // than truncated later: the drafter then proposes a block that fits, instead of
        // proposing more and having it dropped.
        n_draft_max = std::min(n_draft_max, spec_draft_cap);

        SLT_DBG(*this, "max possible draft: %d\n", n_draft_max);

        return n_draft_max;
    }

    // add sampled token of this slot to the batch, optionally add the speculative draft tokens if any
    void handle_last_sampled_token(server_batch & batch) {
        bool add_ok = true;
        // the audio stream has no token before the first step, track the position with a dummy token
        const llama_token tok_track = sampled == LLAMA_TOKEN_NULL ? 0 : sampled;
        if (spec_draft.empty()) {
            // no speculative decoding
            i_batch = batch.size();

            if (!inp_embd.empty()) {
                const int32_t n_rows = (int32_t) inp_embd.size() / batch.n_embd;
                GGML_ASSERT((int32_t) inp_embd.size() == n_rows * batch.n_embd);
                i_batch = batch.size() + n_rows - 1;
                for (int32_t i = 0; i < n_rows; i++) {
                    // text embeddings broadcast the same position across the M-RoPE sections
                    const llama_pos p = prompt.tokens.pos_next();
                    const llama_pos pos[GGML_MROPE_SECTIONS] = { p, p, p, 0 };
                    add_ok &= batch.add_embd(id, inp_embd.data() + (size_t) i * batch.n_embd, pos, i == n_rows - 1, false);
                    if (i < n_rows - 1) {
                        prompt.tokens.push_back(tok_track);
                    }
                }
                if (audio_stream) {
                    inp_embd.clear(); // rows are consumed, the next step produces new ones
                }
            } else {
                add_ok &= batch.add(id, sampled, prompt.tokens.pos_next(), true, false);
            }

            SLT_DBG(*this, "slot decode token, id=%d, n_ctx = %d, n_tokens = %d, truncated = %d\n",
                    sampled, n_ctx, prompt.n_tokens(), truncated);
        } else {
            SLT_DBG(*this, "generate_draft: id=%d, #tokens=%zu, #draft=%zu, pos_next=%d\n",
                    sampled, prompt.tokens.size(), spec_draft.size(), prompt.tokens.pos_next());

            GGML_ASSERT(spec_i_batch.empty());

            spec_i_batch.push_back(batch.size());
            for (size_t i = 0; i < spec_draft.size(); i++) {
                spec_i_batch.push_back(batch.size() + i + 1);
            }

            auto pos0 = prompt.tokens.pos_next();

            add_ok &= batch.add(id, sampled, pos0++, true, false);
            for (auto token : spec_draft) {
                add_ok &= batch.add(this->id, token, pos0++, true, false);
            }
        }

        GGML_ASSERT(add_ok && "batch must be large enough to hold the sampled and draft tokens");

        prompt.tokens.push_back(tok_track);
        prompt.tokens.insert(spec_draft);
    }

    void release() {
        if (is_processing()) {
            GGML_ASSERT(task);

            SLT_INF(*this, "stop processing: n_tokens = %d, truncated = %d\n", prompt.n_tokens(), truncated);

            t_last_used = ggml_time_us();

            state = SLOT_STATE_IDLE;

            // do not keep context of the child slots - the parent's context is enough
            // nor of the stateless slots - no task ever reuses it
            if (task->is_child() || task->is_stateless() || audio_stream || task->type == SERVER_TASK_TYPE_REALTIME) {
                prompt_clear();
            }

            // mtmd_helper always re-eval the whole prompt
            if (task->type == SERVER_TASK_TYPE_TTS) {
                prompt_clear();
            }

            callback_on_reset(*this);

            reset();

            callback_on_release(id);
        }
    }

    size_t find_stopping_strings(const std::string & text, const size_t last_token_size, bool is_full_stop) {
        GGML_ASSERT(task);

        size_t stop_pos = std::string::npos;

        for (const std::string & word : task->params.antiprompt) {
            size_t pos;

            if (is_full_stop) {
                const size_t tmp      = word.size() + last_token_size;
                const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;

                pos = text.find(word, from_pos);
            } else {
                // otherwise, partial stop
                pos = string_find_partial_stop(text, word);
            }

            if (pos != std::string::npos && (stop_pos == std::string::npos || pos < stop_pos)) {
                if (is_full_stop) {
                    stop           = STOP_TYPE_WORD;
                    stopping_word  = word;
                    has_next_token = false;
                }
                stop_pos = pos;
            }
        }

        return stop_pos;
    }

    void print_timings_tg() {
        if (stats.n_gen < 100) {
            return;
        }

        const int64_t t_now = ggml_time_us();

        if (t_now - t_print_last < 3*1000*1000) {
            return;
        }

        const double n_gen_second     = stats.n_gen_tps();
        const double n_gen_second_win = 1e6 / (t_now - t_print_last) * (stats.n_gen - n_gen_last);

        t_print_last = t_now;
        n_gen_last = stats.n_gen;

        SLT_INF(*this, "n_gen = %6d, tg = %6.2f t/s, tg_3s = %6.2f t/s\n", (int) stats.n_gen, n_gen_second, n_gen_second_win);
    }

    void print_timings_pp() const {
        const double t_prompt_total = stats.t_prompt_ms();

        if (t_prompt_total < 3000.0) {
            return;
        }

        const double n_prompt_second = stats.n_prompt_tps();
        const double f_progress = task->n_tokens() > 0 ? (double) prompt.n_tokens() / task->n_tokens() : 0.0;

        SLT_INF(*this, "prompt processing, n_tokens = %6d, progress = %.2f, t = %6.2f s / %.2f tokens per second\n",
                (int) stats.n_prompt_processed, f_progress, t_prompt_total / 1e3, n_prompt_second);
    }

    void print_timings() const {
        const double t_prompt_total = stats.t_prompt_ms();
        const double t_gen_total    = stats.t_gen_ms();

        const double t_prompt        = stats.t_prompt_per_token_ms();
        const double n_prompt_second = stats.n_prompt_tps();

        const double t_gen        = stats.t_gen_per_token_ms();
        const double n_gen_second = stats.n_gen_tps();

        SLT_INF(*this,
                "prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                t_prompt_total, (int) stats.n_prompt_processed, t_prompt, n_prompt_second);

        SLT_INF(*this,
                "       eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                t_gen_total, (int) stats.n_gen, t_gen, n_gen_second);

        SLT_INF(*this,
                "      total time = %10.2f ms / %5d tokens\n",
                t_prompt_total + t_gen_total, (int) (stats.n_prompt_processed + stats.n_gen));

        SLT_INF(*this,
                "   graphs reused = %10d\n",
                llama_perf_context(ctx_tgt).n_reused);

        SLT_INF(*this,
                "  graph reallocs = %10d    / %5d        (same size) / (changed size)\n",
                llama_perf_context(ctx_tgt).n_reallocs_same, llama_perf_context(ctx_tgt).n_reallocs_diff);

        const int32_t n_draft_total       = stats.n_draft_tokens;
        const int32_t n_draft_accepted    = stats.n_draft_accepted;
        const int32_t n_draft_verif_steps = stats.n_draft_verif_steps;

        if (n_draft_total > 0) {
            const float  draft_ratio  = (float) n_draft_accepted / n_draft_total;
            const double mean_acc_len = n_draft_verif_steps > 0 ? 1.0 + (double) n_draft_accepted / (double) n_draft_verif_steps : 1.0;

            std::string acceptance_rates_per_pos;
            if (n_draft_verif_steps > 0) {
                for (size_t i = 0; i < n_accepted_per_pos.size(); ++i) {
                    if (i > 0) {
                        acceptance_rates_per_pos += ", ";
                    }
                    acceptance_rates_per_pos += string_format("%.3f", (double) n_accepted_per_pos[i] / (double) n_draft_verif_steps);
                }
            }

            SLT_INF(*this,
                    "draft acceptance = %0.5f (%5d accepted / %5d generated), mean len = %5.2f\n",
                    draft_ratio, n_draft_accepted, n_draft_total, mean_acc_len);
            SLT_TRC(*this,
                    "     acc per pos = (%s)\n", acceptance_rates_per_pos.c_str());
        }

        common_speculative_print_stats(spec);
    }

    json to_json(bool only_metrics = false) const {
        json res;

        res = {
            {"id",            id},
            {"n_ctx",         n_ctx},
            {"speculative",   can_speculate()},
            {"is_processing", is_processing()},
        };

        const auto & ptask = task ? task : task_prev;

        if (ptask) {
            res["id_task"] = ptask->id;
            res["n_prompt_tokens"]           = (int32_t) prompt.tokens.size();
            res["n_prompt_tokens_processed"] = stats.n_prompt_processed;
            res["n_prompt_tokens_cache"]     = stats.n_prompt_cached;
            res["params"] = ptask->params.to_json(only_metrics);
            res["next_token"] = json::array({
                {
                    {"has_next_token", has_next_token},
                    {"has_new_line",   has_new_line},
                    {"n_remain",       n_remaining()},
                    {"n_decoded",      stats.n_gen},
                }
            });

            if (!only_metrics) {
                res["prompt"] = ptask->tokens.detokenize(ctx_tgt, true);
                res["generated"] = generated.empty() ? debug_generated_text : generated.text;
            }
        }

        return res;
    }

    // the other slot continues from the tokens processed so far
    void copy_prompt_to(server_slot & other) const {
        mem.seq_rm(other.id,     -1, -1);
        mem.seq_cp(id, other.id, -1, -1);

        other.prompt = prompt.clone();
    }

    void copy_state_to(server_slot & other) const {
        GGML_ASSERT(state == SLOT_STATE_DONE_PROMPT);

        mem.seq_rm(other.id,     -1, -1);
        mem.seq_cp(id, other.id, -1, -1);

        other.i_batch = i_batch;

        other.stats = stats;

        other.prompt = prompt.clone();
        other.init_sampler();
    }
};

// encode the mtmd chunk at idx, batched with as many of the next chunks as possible
// returns 0 on success
static int encode_mtmd_chunk(const server_slot & slot, mtmd::batch_ptr & mbatch, size_t idx) {
    const auto & mctx = slot.mctx;
    const auto & input_tokens = slot.task->tokens;
    const auto & chunk = input_tokens.find_chunk(idx);

    mbatch.reset(mtmd_batch_init(mctx));
    int32_t res = mtmd_batch_add_chunk(mbatch.get(), chunk.get());
    GGML_ASSERT(res == 0); // we should never have an empty batch

    // try batching as much as possible
    int n_added = 1;
    size_t idx_cur = idx;
    while (res == 0) {
        auto [next_chunk, next_idx] = input_tokens.find_next_media_chunk(idx_cur);
        if (next_chunk == nullptr) {
            break;
        }
        res = mtmd_batch_add_chunk(mbatch.get(), next_chunk->get());
        n_added += (res == 0 ? 1 : 0);
        idx_cur = next_idx;
        SLT_DBG(slot, "try adding media chunk idx = %zu to batch, res = %d\n", next_idx, res);
        // if res != 0, batch is full or chunk is not compatible -> this loop breaks
    }

    // TODO @ngxson : move this log line to debug when it become more stable
    SLT_TRC(slot, "encoding mtmd batch from idx = %zu, n_chunks = %d\n", idx, n_added);

    res = mtmd_batch_encode(mbatch.get());
    if (res != 0) {
        SLT_ERR(slot, "failed to encode mtmd batch for chunk idx = %zu, res = %d\n", idx, res);
        return -1;
    }

    return 0;
}

// returns 0 on success
// caller need to update prompt.tokens after a successful call to keep track of the processing progress
// note: this is not a member of server_slot because we want to run it inside yield_to_queue
//       slot is passed as const to avoid accidental modification of the slot state
//       some pointers are allowed to be used, they are not used by to_json()
static int process_mtmd_chunk(const server_slot & slot, mtmd::batch_ptr & mbatch, size_t idx, size_t & n_tokens_out) {
    GGML_ASSERT(slot.mctx);
    const auto & mctx = slot.mctx;
    const auto & input_tokens = slot.task->tokens;
    const auto & chunk = input_tokens.find_chunk(idx);
    int32_t res = 0;

    auto try_decode = [&]() -> int32_t {
        if (mbatch) {
            float * embd = mtmd_batch_get_output_embd(mbatch.get(), chunk.get());
            if (embd) {
                struct cb_data_t {
                    common_speculative * spec;
                    llama_context * ctx;
                } cb_data = { slot.spec, slot.ctx_tgt };

                static auto cb = [](const mtmd_helper_embd_batch * b, void * user_data) {
                    const auto * data = static_cast<cb_data_t *>(user_data);

                    common_batch batch(data->ctx);
                    for (int32_t i = 0; i < b->n_tokens; ++i) {
                        llama_pos pos[GGML_MROPE_SECTIONS] = { 0, 0, 0, 0 };
                        for (int32_t j = 0; j < b->n_pos; ++j) {
                            pos[j] = b->pos[j * b->n_tokens + i];
                        }
                        batch.add_embd({ b->embd + (size_t) i * b->n_embd, 1, (size_t) b->n_embd }, pos, b->seq_id, false);
                    }

                    return common_speculative_process(data->spec, batch) ? 0 : 1;
                };

                llama_pos new_n_past; // unused for now
                res = mtmd_helper_decode_image_chunk(
                    mctx,
                    slot.ctx_tgt,
                    chunk.get(),
                    embd,
                    slot.prompt.tokens.pos_next(),
                    slot.id,
                    llama_n_batch(slot.ctx_tgt),
                    &new_n_past,
                    cb,
                    &cb_data
                );
                if (res != 0) {
                    SLT_ERR(slot, "failed to decode mtmd chunk, idx = %zu, res = %d\n", idx, res);
                    return -1;
                }
                n_tokens_out = mtmd_input_chunk_get_n_tokens(chunk.get());
                return 0; // success
            }
        }
        return 1; // (non-error) need to create & encode batch
    };

    // if the batch is already exist, try searching & encode
    res = try_decode();
    if (res == 0) {
        return 0;
    }
    if (res < 0) {
        // fatal error
        return res;
    }

    // otherwise, the batch is either uninitialized or is used up
    // we need to create & encode a new batch
    if (encode_mtmd_chunk(slot, mbatch, idx) != 0) {
        return -1;
    }

    return try_decode();
}

// add an encoded mtmd chunk to the batch of the text tokens, instead of decoding it on its own like process_mtmd_chunk()
// embd is the output of the encoder for this chunk
// returns false if the batch is full
static bool add_mtmd_chunk(const server_slot & slot, const mtmd_input_chunk * chunk, const float * embd, server_batch & batch) {
    // positions are the ones of mtmd_helper_decode_image_chunk()
    const auto * image    = mtmd_input_chunk_get_tokens_image(chunk);
    const bool   is_mrope = mtmd_decode_use_mrope(slot.mctx);
    const size_t n_tokens = mtmd_input_chunk_get_n_tokens(chunk);
    const size_t n_embd   = batch.n_embd;
    const llama_pos pos_0 = slot.prompt.tokens.pos_next();

    for (size_t i = 0; i < n_tokens; i++) {
        const llama_pos p = pos_0 + (llama_pos) i;
        llama_pos pos[GGML_MROPE_SECTIONS] = { p, p, p, p };
        if (is_mrope && image) {
            const mtmd_decoder_pos rel = mtmd_image_tokens_get_decoder_pos(image, pos_0, i);
            pos[0] = rel.t;
            pos[1] = rel.y;
            pos[2] = rel.x;
            pos[3] = rel.z;
        }
        if (!batch.add_embd(slot.id, embd + i * n_embd, pos, slot.need_embd(), /* is_prompt */ true)) {
            return false;
        }
    }
    return true;
}

//
// server_context_impl (private implementation)
//

struct server_context_impl {
    friend struct server_context;

public:
    // only use these pointers outside of this class:
    //  - when not in sleeping state
    //  - and, with thread-safe APIs (e.g., tokenizer calls)
    llama_model * model_tgt = nullptr;

    mtmd_context * mctx = nullptr;
    mtmd_helper_cap mtmd_cap = {};
    // note: video_params.ffmpeg_bin_dir points into params_base, which outlives this struct
    mtmd_helper_init_opt init_opt = mtmd_helper_init_opt_default();
    const llama_vocab * vocab = nullptr;

    // prompt of the "asr" chat template, for live ASR
    llama_tokens asr_prompt;

    bool has_cap_tts() const {
        return mtmd_cap.audio_gen;
    }

    bool has_cap_chat() const {
        return mtmd_cap.chat;
    }

    server_decision_context decision;

    server_queue    queue_tasks;
    server_response queue_results;

    // note: chat_params must not be refreshed upon existing sleeping state
    server_chat_params chat_params;

    server_state_callback_t callback_state = [](server_state, json) -> void {};

    server_context_impl() {
        mtmd_helper_log_set(common_log_default_callback, nullptr);
    }

    ~server_context_impl() {
        if (!sleeping) {
            // destroy() is already called when entering sleeping state
            // we don't call it again here to avoid double free
            destroy();
        }
    }

    server_metrics get_metrics() const {
        return metrics;
    }

    void reset_metrics_bucket() {
        metrics.reset_bucket();
    }

private:
    // note: accessing these fields outside of this class is not thread-safe
    // use server_context methods instead

    common_params params_base;

    // note: keep these alive - they determine the lifetime of the model, context, etc.
    common_init_result_ptr llama_init;

    // [QLH-SPLIT] Owns the tail stage's model and context. Declared after llama_init so it
    // is destroyed before it.
    common_init_result_ptr llama_init_tail;
    llama_context * ctx_tail = nullptr;

    llama_context * ctx_tgt = nullptr;

    server_batch batch;

    llama_model   * model_dft = nullptr;
    llama_context * ctx_dft   = nullptr;

    common_speculative_init_result_ptr spec_init;

    common_context_seq_rm_type ctx_tgt_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
    common_context_seq_rm_type ctx_dft_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;

    common_speculative_ptr spec;

    bool add_bos_token = true;

    int32_t n_ctx; // total context for all clients / slots

    // set to llama_model_n_swa(model)
    // if swa_full is enabled, this is set to 0 to simulate a non-SWA model
    int32_t n_swa;

    // slots / clients
    std::vector<server_slot> slots;

    int trace = 0;        // env: LLAMA_TRACE
    int slots_debug = 0;  // env: LLAMA_SERVER_SLOTS_DEBUG
    int slots_n_diff = 0; // env: LLAMA_SERVER_SLOTS_N_DIFF

    int n_empty_consecutive = 0;

    std::unique_ptr<server_prompt_cache> prompt_cache;

    server_metrics metrics;

    // queued prompt stats - llama_decode() is async, so the timing is only valid after a sync
    // note: kept out of server_metrics, which is copied as-is into the task result
    int64_t  t_decode_start  = 0; // start of the last submitted decode
    int64_t  t_prompt_start  = 0; // start of the oldest queued prompt decode
    uint64_t n_prompt_queued = 0;

    json json_ui_settings = json::object();

    // Necessary similarity of prompt for slot selection
    float slot_prompt_similarity = 0.0f;

    std::string model_name; // name of the loaded model, to be used by API
    std::set<std::string> model_aliases; // additional names for the model
    std::set<std::string> model_tags;    // informational tags

    bool sleeping = false;

    int64_t t_last_load_progress_ms = 0;

    void destroy() {
        spec.reset();
        spec_init.reset();

        ctx_dft   = nullptr;
        model_dft = nullptr;

        llama_init.reset();

        ctx_tgt = nullptr;
        model_tgt = nullptr;

        mtmd_free(mctx);
        mctx = nullptr;
    }

    void handle_sleeping_state(bool new_state) {
        GGML_ASSERT(sleeping != new_state);
        if (new_state) {
            if (callback_state) {
                callback_state(SERVER_STATE_SLEEPING, {});
                // note: for sleeping == false, event is emitted by load_model()
            }
            SRV_INF("%s", "server is entering sleeping state\n");
            destroy();
        } else {
            SRV_INF("%s", "server is exiting sleeping state\n");
            if (!load_model(params_base)) {
                GGML_ABORT("failed to reload model after sleeping");
            }
        }
        sleeping = new_state;
    }

    struct load_progress_data {
        server_context_impl * ctx;
        std::string stage;
        std::vector<std::string> stages;
        int64_t t_last_load_progress_ms = 0;
        load_progress_data(server_context_impl * ctx, const std::string & stage) : ctx(ctx), stage(stage) {}
    };
    static bool load_progress_callback(float progress, void * user_data) {
        auto * d = static_cast<load_progress_data *>(user_data);
        GGML_ASSERT(d);
        // always emit the first and final sample; throttle the rest to one per 200ms
        {
            auto & t_last = d->t_last_load_progress_ms;
            const int64_t t_now = ggml_time_ms();
            const bool first = t_last == 0;
            const bool done  = progress >= 1.0f;
            const bool throttled = !first && !done && (t_now - t_last) < 200;
            if (throttled) {
                return true;
            }
            t_last = t_now;
        }
        if (d->ctx->callback_state) {
            d->ctx->callback_state(SERVER_STATE_LOADING, {
                {"stages", d->stages},
                {"current", d->stage},
                {"value", progress},
            });
        }
        return true;
    }

    // load the model and initialize llama_context
    // this may also be called to resume from sleeping state
    bool load_model(common_params & params) {
        load_progress_data load_progress_text  (this, "text_model");
        load_progress_data load_progress_mmproj(this, "mmproj_model");
        load_progress_data load_progress_spec  (this, "spec_model");

        const bool is_resume = sleeping;

        params_base = params;

        // [QLH-SPLIT] v2.4: a split head is a *stage*, and a stage defaults op_offload off.
        //
        // Upstream defaults it on, and on a range load that is a negative head: the CPU-resident
        // blocks' weights move into the pinned CUDA host buffer and the GPU runs their matmuls,
        // costing ~31% of decode speed (segment probe 2026-11-09: head 0:32 -ngl 8 tg 14.60 with
        // it on vs 22.25 with it off, same build). tools/split-stage has defaulted it off since
        // v2.2; this server, which is the head of every chain, kept upstream's default, so the
        // same head measured +36% on the probe and -10% through the server (E 8.44 vs A 9.34).
        //
        // The condition is the same one common.cpp uses to turn on the relay output
        // (embeddings + embeddings_pre_norm): "this process is a stage of a chain". An explicit
        // --op-offload / --no-op-offload still wins, which is what no_op_offload_explicit is for
        // -- without it this default would overwrite an operator's flag. The value itself lives
        // in src/qlh-split-defaults.h so the head and its stages read one constant.
        const bool split_role = !params_base.split_remote.empty() || params_base.split_tail_lo >= 0;
        if (split_role) {
            if (!params_base.no_op_offload_explicit) {
                params_base.no_op_offload = QLH_SPLIT_DEFAULT_NO_OP_OFFLOAD;
            }
            SRV_INF("[split] op_offload = %s (%s)\n",
                    params_base.no_op_offload ? "off" : "on",
                    params_base.no_op_offload_explicit
                            ? "explicit --op-offload/--no-op-offload"
                            : "default for a split stage, see src/qlh-split-defaults.h");
        }

        const auto output_limits = server_output_limits(params_base);
        params_base.n_outputs_max = output_limits.total;
        params_base.n_outputs_max_per_seq = output_limits.per_seq;

        // [QLH-SPLIT] A chain is either all-local or all-remote (llama_context refuses to mix
        // them too), and saying so before loading two models is the difference between an
        // immediate usage error and a few seconds of startup followed by the same error.
        if (!params_base.split_remote.empty() && params_base.split_tail_lo >= 0) {
            SRV_ERR("%s\n", "--split-remote and --split-tail-range are mutually exclusive: a "
                    "chain is either all local stages or a local head with remote stages");
            return false;
        }

        const bool has_mmproj = !params.mmproj.path.empty();
        const bool has_draft = params.speculative.has_dft();
        const bool spec_mtp = std::find(params_base.speculative.types.begin(),
                                        params_base.speculative.types.end(),
                                        COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params_base.speculative.types.end();
        const bool has_spec = has_draft || spec_mtp;

        if (callback_state) {
            std::vector<std::string> stages = {"text_model"};
            if (has_spec) {
                stages.push_back("spec_model");
            }
            if (has_mmproj) {
                stages.push_back("mmproj_model");
            }
            load_progress_text.stages   = stages;
            load_progress_mmproj.stages = stages;
            load_progress_spec.stages   = stages;

            // trigger 0% progress
            load_progress_callback(0.0f, &load_progress_text);
        }


        SRV_INF("loading model '%s'\n", params.model.get_name().c_str());
        SRV_TRC("local path '%s'\n", params.model.path.c_str());

        std::string & mmproj_path = params_base.mmproj.path;
        mtmd_context_params mparams = mtmd_context_params_default();
        if (has_mmproj) {
            mparams.use_gpu          = params_base.mmproj_use_gpu;
            mparams.device           = params_base.mmproj_device;
            mparams.print_timings    = false;
            mparams.n_threads        = params_base.cpuparams.n_threads;
            mparams.flash_attn_type  = params_base.flash_attn_type;
            mparams.warmup           = params_base.warmup;
            mparams.image_min_tokens = params_base.image_min_tokens;
            mparams.image_max_tokens = params_base.image_max_tokens;
            mparams.batch_max_tokens = params_base.mtmd_batch_max_tokens;
            mparams.media_marker     = get_media_marker();
            // progress callback
            mparams.progress_callback           = load_progress_callback;
            mparams.progress_callback_user_data = &load_progress_mmproj;
        }

        // get the memory usage of mmproj, also used to check image_max_tokens against n_ubatch
        mtmd_memory_usage mmproj_usage = {{}, -1, false};
        int64_t mmproj_usage_t_us = 0;
        if (has_mmproj) {
            int64_t t_start = ggml_time_us();
            mmproj_usage = mtmd_get_memory_usage(mmproj_path.c_str(), mparams);
            mmproj_usage_t_us = ggml_time_us() - t_start;
        }

        // optionally fit mmproj memory usage
        // on resume params_base already holds the margin from the first load
        if (has_mmproj && params_base.fit_params && !is_resume) {
            const auto & mmproj_mem = mmproj_usage.backend_mem_usage;
            const int64_t t_elapsed = mmproj_usage_t_us;
            if (!mmproj_mem.empty()) {
                size_t total = 0;
                for (auto & [dev, size] : mmproj_mem) {
                    total += size;
                }
                SRV_TRC("[mtmd] estimated worst-case memory usage of mmproj is %.2f MiB (took %.2f ms)\n", total / (1024.0 * 1024.0), t_elapsed / 1000.0);
                GGML_ASSERT(!params_base.fit_params_target.empty());
                for (auto & [dev, size] : mmproj_mem) {
                    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                        if (ggml_backend_dev_get(i) == dev) {
                            if (i < params_base.fit_params_target.size()) {
                                SRV_DBG("[mtmd] adding %.2f MiB to fit_params_target for device %s\n", size / (1024.0 * 1024.0), ggml_backend_dev_name(dev));
                                params_base.fit_params_target[i] += size;
                            }
                            break;
                        }
                    }
                }
            } else {
                SRV_ERR("%s", "[mtmd] failed to get memory usage of mmproj\n");
            }
        }

        // note: the draft / MTP context is fitted together with the target model, see common_fit_extra_model

        // attach a progress callback
        {
            params_base.load_progress_callback = load_progress_callback;
            params_base.load_progress_callback_user_data = &load_progress_text;
        }

        // [QLH-SPLIT] v2.3: VRAM budget pre-check for the segment this server loads.
        //
        // The head is the second process on the card in the configuration that spilled silently
        // on 2026-11-09 (`head -ngl 8` + a fully offloaded tail: 8.4 GiB needed against 7.9 GiB
        // free, tg 10.76 -> 0.343). By the time the head loads, the tail has usually taken its
        // share already, so the free number read here is the one that decides -- which is why
        // this check is per-process and not a chain-wide sum: each process reports what it can
        // see, and the one that starts last sees the truth.
        //
        // -ngl 0 says nothing; a CPU-only build says one line and skips (see the header).
        qlh_vram::estimate_input vram_in;
        vram_in.model_path   = params_base.model.path.c_str();
        vram_in.lo           = params_base.split_layer_lo;
        vram_in.hi           = params_base.split_layer_hi;
        vram_in.n_gpu_layers = params_base.n_gpu_layers;
        vram_in.overrides    = params_base.tensor_buft_overrides.empty()
                             ? nullptr : params_base.tensor_buft_overrides.data();
        vram_in.moe_cache    = params_base.moe_cache_size;
        const qlh_vram::precheck_result vram_pre = qlh_vram::precheck(vram_in, "llama-server");
        if (vram_pre.v == qlh_vram::VERDICT_REFUSED) {
            return false;
        }

        llama_init = common_init_from_params(params_base);

        model_tgt = llama_init->model();
        ctx_tgt   = llama_init->context();

        // [QLH-SPLIT] Optional tail stage. It is a full second model over a different layer
        // range, loaded through the same path so every common_params setting (n_rs_seq
        // included) reaches it identically. llama_split_attach then makes ctx_tgt drive it.
        if (params_base.split_tail_lo >= 0 && ctx_tgt != nullptr) {
            common_params tail_params = params_base;
            tail_params.split_layer_lo = params_base.split_tail_lo;
            tail_params.split_layer_hi = params_base.split_tail_hi;
            tail_params.split_tail_lo  = -1; // no recursion
            tail_params.split_tail_hi  = -1;
            // [QLH-SPLIT] This context is the far half of a chain this server is assembling, so
            // a range that does not start at layer 0 is the correct shape here, not a truncated
            // model. The fields below cannot say that on their own: a user typing
            // `--split-layer-range 20:24` produces exactly this common_params and must still be
            // refused (nothing computes layers 0..19 in that process). The flag is the only
            // difference, and common.cpp reads it to allow this load while rejecting that one.
            tail_params.split_range_is_stage = true;

            SRV_INF("[split] loading tail stage for layers [%d, %d)\n",
                    tail_params.split_layer_lo, tail_params.split_layer_hi);

            llama_init_tail = common_init_from_params(tail_params);
            ctx_tail = llama_init_tail->context();

            if (ctx_tail == nullptr) {
                SRV_ERR("%s", "[split] failed to create the tail context\n");
                return false;
            }

            // A rejected chain is a startup failure, not a warning: continuing as a single-stage
            // context would serve plausible logits from a model that is missing layers, and every
            // request would be silently wrong.
            if (llama_split_attach(ctx_tgt, &ctx_tail, 1) < 0) {
                SRV_ERR("%s", "[split] failed to attach the tail stage; check that it starts where "
                        "the head ends (--split-layer-range / --split-tail-range) and ends at the "
                        "model's last trunk layer\n");
                return false;
            }
            SRV_INF("[split] attached %d tail stage(s); n_rs_seq=%u\n",
                    llama_split_n_stages(ctx_tgt), llama_n_rs_seq(ctx_tgt));
        }

        // [QLH-SPLIT] Remote chain: the head is this process, everything after it is a
        // split-stage process reached over TCP. A rejected attach is a startup failure for the
        // same reason as the local one above: serving from a head whose layers are not actually
        // followed by the rest of the model would be silently wrong, not "degraded".
        if (!params_base.split_remote.empty() && ctx_tgt != nullptr) {
            std::vector<const char *> endpoints;
            endpoints.reserve(params_base.split_remote.size());
            std::string joined;
            for (const auto & ep : params_base.split_remote) {
                endpoints.push_back(ep.c_str());
                joined += joined.empty() ? ep : ", " + ep;
            }

            char err[512] = {0};
            const int n = llama_split_attach_remote(ctx_tgt, endpoints.data(),
                                                    (int32_t) endpoints.size(), err, sizeof(err));
            if (n < 0) {
                SRV_ERR("[split] failed to attach remote stages [%s]: %s\n", joined.c_str(), err);
                return false;
            }

            SRV_INF("[split] attached %d remote stage(s): %s\n", n, joined.c_str());
        }

        if (model_tgt == nullptr) {
            SRV_ERR("failed to load model, '%s'\n", params_base.model.path.c_str());
            return false;
        }

        if (ctx_tgt == nullptr) {
            SRV_ERR("failed to create_context with model '%s'\n", params_base.model.path.c_str());
            return false;
        }

        // [QLH-SPLIT] v2.3: post-load VRAM check, once every segment this process owns (the head
        // and, with --split-tail-range, a local tail) has been loaded and attached.
        //
        // Warning only, never fatal: a spill has already happened by this point, and a spilled
        // tier is sometimes the fastest one measured -- the criterion is throughput, not this
        // line. What it buys is that a 27x slowdown stops looking like "the model is slow".
        qlh_vram::after_load(vram_pre.est, vram_pre.dev, "llama-server");

        vocab = llama_model_get_vocab(model_tgt);

        try {
            decision.init(model_tgt);
        } catch (const std::exception & e) {
            SRV_ERR("failed to init decision model: %s\n", e.what());
            return false;
        }

        n_ctx = llama_n_ctx(ctx_tgt);

        add_bos_token = llama_vocab_get_add_bos(vocab);

        if (has_spec) {
            // spec_mtp doesn't use load a model internally, so we report 0.0 and 1.0 manually
            load_progress_callback(0.0f, &load_progress_spec);
            load_progress_spec.t_last_load_progress_ms = 0;  // reset so internal cbs aren't delayed

            // [QLH-SPLIT] v8: decide whether this chain can draft with MTP *before* any params are
            // derived from it, so both the context construction below (params_dft) and the
            // speculative implementation (built later from params_base) see the same answer.
            //
            // A remote tail owns the MTP head, and this process cannot build a draft context at
            // all -- its model is the head range, which loads no NextN layers. What it can do is
            // ask: the tail reported has_nextn in OP_HELLO at attach time, so spec_init builds no
            // context and the drafts come over the chain (common/speculative.cpp, is_remote).
            // When the tail has no NextN block, MTP is dropped from the type list with a warning
            // instead of failing startup: an ngram drafter in the same request set still works,
            // and a chain without drafts is still a correct chain.
            bool spec_mtp_ok = spec_mtp;
            // [QLH-SPLIT] v9: DFlash/DSpark read the trunk's layer-input taps, so a remote chain
            // has to tell them the tail is remote too. Without this flag every tap resolves onto
            // the head context (common/speculative.cpp, tail_remote == false), which then aborts
            // in set_embeddings_layer_inp on a source layer the head does not have. The tap and
            // verify channels themselves are already there; this is only their entry switch.
            if (has_draft && ctx_tail == nullptr && llama_context_split_has_remote(ctx_tgt)) {
                params_base.speculative.draft.ctx_tail_remote = true;
                SRV_INF("%s", "[split] remote chain + draft model: layer-input taps cross the wire "
                              "(the head keeps the draft context, the stages answer the taps)\n");
            }
            if (spec_mtp && ctx_tail == nullptr && llama_context_split_has_remote(ctx_tgt)) {
                if (llama_context_split_mtp_ready(ctx_tgt)) {
                    params_base.speculative.draft.ctx_tail_remote = true;
                    SRV_INF("%s", "[split] the remote tail reports a NextN/MTP block: MTP drafts "
                                  "will run in the stage and cross the chain (one h row out, k "
                                  "token ids back per round)\n");
                } else {
                    SRV_WRN("%s", "[split] --spec-type draft-mtp is not available: the remote tail "
                                  "reports no NextN/MTP block (its model has none, or its range "
                                  "does not end at the model's last layer, or its MTP context "
                                  "failed to build). Dropping it; any other speculative type "
                                  "stays enabled\n");
                    auto & types = params_base.speculative.types;
                    types.erase(std::remove(types.begin(), types.end(),
                                            COMMON_SPECULATIVE_TYPE_DRAFT_MTP), types.end());
                    if (types.empty()) {
                        types.push_back(COMMON_SPECULATIVE_TYPE_NONE);
                    }
                    spec_mtp_ok = false;
                }
            }

            // This block builds the draft context. With MTP dropped above and no draft model, there
            // is nothing to build -- the remaining types (ngram, ngram-map, ...) need no context,
            // and common_speculative_init() below builds them from params_base.
            if (has_draft || spec_mtp_ok) {
                common_params params_dft = common_base_params_to_speculative(params_base);

                // progress callback
                params_dft.load_progress_callback           = load_progress_callback;
                params_dft.load_progress_callback_user_data = &load_progress_spec;

                // [QLH-SPLIT] A split chain's MTP head does not live in the head segment: the
                // head range loads [0, layer_hi) with nextn_keep = 0, so its model has no NextN
                // layers and llama_init_from_model(ctx_type=MTP) refuses it. The tail range is
                // the only one that keeps blk.<n_layer>.nextn.*, so the draft context is built
                // over the tail's model and the target h_nextn rows are read from the tail's
                // context. Without the chain, both stay the server's own model/context.
                //
                // DFlash is the opposite case and needs the head: it consumes layer-input taps
                // of the trunk, and its token embedding / vocabulary are in the head's range,
                // so ctx_other (borrowed through llama_init_from_model) has to be the head.
                // Taps that fall past the head's range are read from the tail's context directly
                // inside common/speculative.cpp; that is a memory read in this process, not a
                // transfer, so the chain stays a single round trip.
                //
                // This only covers a *local* tail. A remote tail is still not supported for
                // DFlash, and the blocker is the verification *channel*, not bandwidth:
                // tools/split-stage asks the tail for the last token's logits only
                // (main.cpp, `batch.logits[i] = relay ? 0 : (i + 1 == n_tok)`) and replies
                // with exactly one row, while speculative verification needs k+1 rows.
                //
                // [QLH-SPLIT] An earlier version of this comment claimed "k=3 is ~4 MB per
                // frame, 3-4 s on 1 Gbps". That arithmetic was wrong twice over: 4 MB on
                // 1 Gbps is 32 ms (250 ms at this repo's measured 16 MB/s), and greedy
                // verification does not need the full row at all -- argmax computed on the
                // tail side costs one token id per row (4 B). v2 did exactly that and got
                // 1.11-1.13x across hosts. What is actually missing here is a "one result
                // per row" reply, which the candidate channel could carry once it stops
                // rejecting speculative requests. Full-row logits are only needed for
                // *sampling* rejection (p and q), and that path is genuinely too expensive
                // on a slow link -- v2 measured it at 1.9x slower than no speculation.
                // Note: the name says what the flag *means* for this choice, not which type set it.
                // DSpark is built by the same implementation as DFlash (common/speculative.cpp
                // distinguishes them by markov_w1) and reads the same layer-input taps, so it needs
                // the head here exactly like DFlash does. Leaving it out sent a DSpark chain to the
                // tail context, whose n_layer() made target layer 20 resolve past the tail (8 layers)
                // and abort in the impl ctor.
                const auto & spec_types = params_base.speculative.types;
                const bool spec_needs_layer_inp =
                        std::find(spec_types.begin(), spec_types.end(), COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH) != spec_types.end() ||
                        std::find(spec_types.begin(), spec_types.end(), COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK) != spec_types.end();
                llama_context * ctx_spec_tgt = (ctx_tail != nullptr && !spec_needs_layer_inp) ? ctx_tail : ctx_tgt;
                llama_model   * mdl_spec_tgt = const_cast<llama_model *>(llama_get_model(ctx_spec_tgt));

                params_dft.speculative.draft.ctx_tail = ctx_tail;
                // params_dft is what actually reaches common_speculative_init_from_params(): the
                // remote-MTP flag has to be on it, or the MTP branch there would try to build a
                // draft context over the head range (and fail, which is exactly the behaviour v8
                // replaces).
                params_dft.speculative.draft.ctx_tail_remote = params_base.speculative.draft.ctx_tail_remote;

                spec_init = common_speculative_init_from_params(params_dft, mdl_spec_tgt, ctx_spec_tgt);
                model_dft = spec_init->model();
                ctx_dft   = spec_init->context();

                if (has_draft && model_dft == nullptr) {
                    SRV_ERR("failed to load draft model, '%s'\n", params_dft.model.path.c_str());
                    return false;
                }

                // [QLH-SPLIT] v8: a remote-MTP chain has no draft context in this process *by
                // design* -- the head lives in the stage. Only a local MTP chain (or a draft
                // model) can end up here without one, and that is still a startup failure.
                if (ctx_dft == nullptr && !params_base.speculative.draft.ctx_tail_remote) {
                    SRV_ERR("%s", "failed to create MTP context\n");
                    return false;
                }

                params_base.speculative.draft.ctx_tgt = ctx_spec_tgt;
                params_base.speculative.draft.ctx_dft = ctx_dft;
                // [QLH-SPLIT] The speculative implementation is built later from
                // params_base.speculative, not from the local params_dft above, so the tail
                // context has to be carried here too: DFlash's impl resolves the taps that
                // fall past the head's range through this pointer.
                params_base.speculative.draft.ctx_tail = ctx_tail;
            }

            load_progress_callback(1.0f, &load_progress_spec);
        }

        if (has_mmproj) {
            if (callback_state) {
                callback_state(SERVER_STATE_LOADING, {{"stage", "mmproj_model"}});
            }

            if (!is_resume) {
                mtmd_helper_log_set(common_log_default_callback, nullptr);
            }

            // the image must fit in one ubatch if the model needs non-causal attention on it
            // a non-causal model also has the text of the prompt in that ubatch, leave half of it to the text
            {
                const int n_ubatch = llama_n_ubatch(ctx_tgt);
                const bool is_mixed = use_mixed_batch();
                if (is_mixed || mmproj_usage.use_non_causal) {
                    const int n_max = is_mixed ? n_ubatch / 2 : n_ubatch;
                    if (mmproj_usage.image_max_tokens > n_max) {
                        SRV_WRN("cap image_max_tokens (original=%d) to %d (n_ubatch = %d) because %s\n", mmproj_usage.image_max_tokens, n_max, n_ubatch,
                                is_mixed ? "the model processes the prompt in one ubatch" : "model needs non-causal attention on image");
                        SRV_WRN("%s\n", "increase n_ubatch (-ub) to increase vision token budget");
                        mparams.image_max_tokens = n_max;
                        mparams.image_min_tokens = std::min(mparams.image_min_tokens, n_max);
                    }
                }
            }

            mctx = mtmd_init_from_file(mmproj_path.c_str(), model_tgt, mparams);
            if (mctx == nullptr) {
                SRV_ERR("failed to load multimodal model, '%s'\n", mmproj_path.c_str());
                return false;
            }
            SRV_INF("loaded multimodal model, '%s'\n", mmproj_path.c_str());

            init_opt.video_params.fps_target = params_base.video_fps;
            init_opt.video_params.timestamp_interval_ms = params_base.video_timestamp_interval_ms;
            init_opt.video_params.ffmpeg_bin_dir = params_base.video_ffmpeg_bin_dir.empty()
                                ? nullptr : params_base.video_ffmpeg_bin_dir.c_str();

            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by multimodal, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by multimodal, it will be disabled");
            }
        }

        mtmd_cap = mtmd_helper_get_capabilities(model_tgt, mctx);

        if (!llama_memory_can_shift(llama_get_memory(ctx_tgt))) {
            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by this context, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by this context, it will be disabled");
            }
        }

        if (llama_model_n_swa(model_tgt) == 0) {
            if (params_base.swa_full) {
                params_base.swa_full = false;
                SRV_WRN("%s\n", "swa_full is not supported by this model, it will be disabled");
            }
        }

        n_swa = params_base.swa_full ? 0 : llama_model_n_swa(model_tgt);

        // Necessary similarity of prompt for slot selection
        slot_prompt_similarity = params_base.slot_prompt_similarity;

        const int n_ctx_train = llama_model_n_ctx_train(model_tgt);

        {
            // note: the capping itself is done in n_ctx_slot(), here we only report it
            const int n_ctx_seq = llama_n_ctx_seq(ctx_tgt);

            if (params_base.kv_unified_per_slot > 0) {
                if (n_ctx_seq > params_base.kv_unified_per_slot) {
                    SRV_INF("capping per-slot context (%d) to --kv-unified-per-slot (%d)\n",
                            n_ctx_seq, params_base.kv_unified_per_slot);
                } else if (params_base.kv_unified_per_slot > n_ctx_seq) {
                    // cap is above the per-slot pool capacity, so it can never bind
                    SRV_WRN(
                        "--kv-unified-per-slot (%d) exceeds the per-slot pool capacity (%d) - cap has no effect, "
                        "slots are limited to %d (raise the KV pool with -c, or unset -c to size it to "
                        "n_parallel * kv_unified_per_slot)\n",
                        params_base.kv_unified_per_slot, n_ctx_seq, n_ctx_seq);
                }
            }

            const int n_ctx_capped = params_base.kv_unified_per_slot > 0 ?
                std::min(n_ctx_seq, params_base.kv_unified_per_slot) : n_ctx_seq;

            if (n_ctx_capped > n_ctx_train) {
                SRV_WRN("the slot context (%d) exceeds the training context of the model (%d) - capping\n",
                        n_ctx_capped, n_ctx_train);
            }
        }

        slots.clear();

        ctx_tgt_seq_rm_type = common_context_can_seq_rm(ctx_tgt);
        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            SRV_WRN("%s", "speculative decoding not supported by this context\n");
        }

        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
            SRV_TRC("%s", "speculative decoding will use checkpoints\n");
        }

        // setup slots
        SRV_INF("initializing, n_slots = %d, n_ctx_slot = %d, kv_unified = '%s'\n",
                params_base.n_parallel, n_ctx_slot(), params_base.kv_unified ? "true" : "false");

        // initialize slots
        for (int i = 0; i < params_base.n_parallel; i++) {
            slots.emplace_back();
        }

        // try speculative decoding
        if (ctx_tgt_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            try {
                spec.reset(common_speculative_init(params_base.speculative, params_base.n_parallel));
            } catch (const std::exception & e) {
                SRV_ERR("failed to initialize speculative decoding context: %s\n", e.what());
                if (params_base.speculative.has_synth()) {
                    return false;
                }
            }
        }

        if (ctx_dft) {
            ctx_dft_seq_rm_type = common_context_can_seq_rm(ctx_dft);
        }

        if (spec) {
            SRV_TRC("%s", "speculative decoding context initialized\n");
        } else {
            spec_init.reset();
            ctx_dft   = nullptr;
            model_dft = nullptr;
        }

        if (!spec && params_base.speculative.has_synth()) {
            SRV_ERR("%s", "synthetic acceptance requires an initialized speculative decoding context\n");
            return false;
        }

        for (int i = 0; i < params_base.n_parallel; i++) {
            server_slot & slot = slots[i];

            slot.id      = i;
            slot.ctx_tgt = ctx_tgt;
            slot.ctx_dft = ctx_dft;
            slot.mem.init(ctx_tgt, ctx_dft);
            slot.spec    = spec.get();
            slot.n_ctx   = n_ctx_slot();

            slot.mctx                   = mctx;
            slot.prompt.tokens.has_mtmd = mctx != nullptr;

            if (has_cap_tts()) {
                slot.tts.ctx.init(ctx_tgt, mctx);
            }

            SLT_TRC(slot, "new slot, n_ctx = %d\n", slot.n_ctx);

            slot.callback_on_release = [this](int id_slot) {
                queue_tasks.pop_deferred_task(id_slot);
            };

            slot.callback_on_reset = [this](const server_slot & slot) {
                // flush the generated token stats before reset()
                if (slot.stats.n_gen > 0) {
                    metrics_on_prediction(slot);
                }
            };

            slot.reset();
        }

        {
            const char * LLAMA_TRACE = getenv("LLAMA_TRACE");
            trace = LLAMA_TRACE ? atoi(LLAMA_TRACE) : 0;

            if (trace) {
                SRV_WRN("LLAMA_TRACE = %d\n", trace);
            }
        }

        {
            const char * LLAMA_SERVER_SLOTS_DEBUG = getenv("LLAMA_SERVER_SLOTS_DEBUG");
            slots_debug = LLAMA_SERVER_SLOTS_DEBUG ? atoi(LLAMA_SERVER_SLOTS_DEBUG) : 0;

            if (slots_debug) {
                SRV_WRN("LLAMA_SERVER_SLOTS_DEBUG = %d\n", slots_debug);
            }
        }

        {
            const char * LLAMA_SERVER_SLOTS_N_DIFF = getenv("LLAMA_SERVER_SLOTS_N_DIFF");
            slots_n_diff = LLAMA_SERVER_SLOTS_N_DIFF ? atoi(LLAMA_SERVER_SLOTS_N_DIFF) : 0;

            if (slots_n_diff) {
                SRV_WRN("LLAMA_SERVER_SLOTS_N_DIFF = %d\n", slots_n_diff);
            }
        }

        // the update_slots() logic will always submit a maximum of n_batch or n_parallel tokens
        // note that n_batch can be > n_ctx (e.g. for non-causal attention models such as BERT where the KV cache is not used)
        {
            const int32_t n_batch = llama_n_batch(ctx_tgt);
            const int32_t n_embd  = llama_model_n_embd_inp(model_tgt);
            batch.init(ctx_tgt, std::max(n_batch, params_base.n_parallel), n_embd);
        }

        // decision requests rarely repeat a state, the prompt cache is opt-in for decision models
        if (decision.type != COMMON_DECISION_TYPE_NONE && !params_base.cache_ram_explicit) {
            SRV_INF("%s", "decision model: prompt cache disabled by default, use `--cache-ram N` to enable it\n");
            params_base.cache_ram_mib    = 0;
            params_base.cache_idle_slots = false;
        }

        if (params_base.cache_ram_mib != 0) {
            if (params_base.cache_ram_mib < 0) {
                SRV_TRC("prompt cache is enabled, size limit: %s\n", "no limit");
            } else {
                SRV_TRC("prompt cache is enabled, size limit: %d MiB\n", params_base.cache_ram_mib);
            }
            SRV_TRC("%s", "use `--cache-ram 0` to disable the prompt cache\n");

            prompt_cache = std::make_unique<server_prompt_cache>(params_base.cache_ram_mib, n_ctx);
        } else {
            SRV_TRC("%s", "prompt cache is disabled - use `--cache-ram N` to enable it\n");
        }
        SRV_TRC("%s", "for more info see https://github.com/ggml-org/llama.cpp/pull/16391\n");

        if (params_base.n_ctx_checkpoints > 0) {
            SRV_TRC("context checkpoints enabled, max = %d, min spacing = %d\n",
                    params_base.n_ctx_checkpoints, params_base.checkpoint_min_step);
        } else {
            SRV_TRC("%s", "context checkpoints disabled\n");
        }

        if (!params_base.model_alias.empty()) {
            // backward compat: use first alias as model name
            model_name = *params_base.model_alias.begin();
        } else if (!params_base.model.get_name().empty()) {
            model_name = params_base.model.get_name();
        } else {
            // fallback: derive model name from file name
            auto model_path = std::filesystem::path(params_base.model.path);
            model_name = model_path.filename().string();
        }

        model_aliases = params_base.model_alias;
        model_tags    = params_base.model_tags;

        // propagate new defaults back to caller
        params = params_base;

        if (!is_resume) {
            return init();
        }

        if (callback_state) {
            callback_state(SERVER_STATE_READY, {});
        }

        return true;
    }

    // unlike load_model(), this is only called once during initialization
    bool init() {
        GGML_ASSERT(ctx_tgt   != nullptr);
        GGML_ASSERT(model_tgt != nullptr);

        GGML_ASSERT(!sleeping);

        // wiring up server queues
        queue_tasks.on_new_task([this](server_task && task, bool is_yielding) {
            return process_single_task(std::move(task), is_yielding);
        });
        queue_tasks.on_update_slots([this]() {
            update_slots();
        });
        queue_tasks.on_sleeping_state([this](bool sleeping) {
            handle_sleeping_state(sleeping);
        });

        metrics.init();

        if (params_base.cache_idle_slots) {
            if (params_base.cache_ram_mib == 0) {
                SRV_WRN("%s", "--cache-idle-slots requires --cache-ram, disabling\n");
                params_base.cache_idle_slots = false;
            } else {
                if (params_base.kv_unified) {
                    SRV_TRC("%s", "idle slots will be saved to prompt cache and cleared upon starting a new task\n");
                } else {
                    // without a unified KV cache, clearing a slot frees no reusable room, so we only
                    // publish a RAM-cache copy of idle slots (their KV stays in VRAM) [TAG_IDLE_SLOT_CLEAR]
                    SRV_TRC("%s", "idle slots will be saved to prompt cache upon starting a new task\n");
                }
                SRV_DBG("%s", "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__\n");
            }
        }

        {
            const std::string & cfg = params_base.ui_config_json;
            if (!cfg.empty()) {
                try {
                    json json_settings = json::parse(cfg);
                    json_ui_settings = json_settings;
                } catch (const std::exception & e) {
                    SRV_ERR("%s: failed to parse UI config: %s\n", __func__, e.what());
                    return false;
                }
            }
        }

        // populate chat template params
        {
            common_chat_templates_ptr chat_templates;
            bool enable_thinking = false;

            try {
                chat_templates = common_chat_templates_init(model_tgt, params_base.chat_template);

                SRV_TRC("%s: chat template, example_format: '%s'\n", __func__,
                    common_chat_format_example(chat_templates.get(), params_base.use_jinja, params_base.default_template_kwargs).c_str());

                // thinking is enabled if:
                // 1. It's not explicitly disabled via --reasoning off
                // 2. The chat template supports it
                const bool template_supports_thinking = params_base.use_jinja && common_chat_templates_support_enable_thinking(chat_templates.get());
                enable_thinking = params_base.enable_reasoning != 0 && template_supports_thinking;
                SRV_TRC("%s: chat template, thinking = %d\n", __func__, enable_thinking);
            } catch (const std::exception & e) {
                SRV_ERR("%s: chat template parsing error: %s\n", __func__, e.what());
                SRV_ERR("%s: please consider disabling jinja via --no-jinja, or use a custom chat template via --chat-template\n", __func__);
                SRV_ERR("%s: for example: --no-jinja --chat-template chatml\n", __func__);
                return false;
            }

            try {
                asr_prompt = server_chat_get_asr_prompt(model_tgt);
            } catch (const std::exception & e) {
                SRV_ERR("%s: \"asr\" chat template error: %s\n", __func__, e.what());
                return false;
            }

            // IMPORTANT: chat_params is reused across sleeping / resuming states,
            //            never store llama_context/llama_model pointers in chat_params,
            //            as they may be invalidated after sleeping
            chat_params = {
                /* use_jinja             */ params_base.use_jinja,
                /* prefill_assistant     */ params_base.prefill_assistant,
                /* reasoning_format      */ params_base.reasoning_format,
                /* chat_template_kwargs  */ params_base.default_template_kwargs,
                /* tmpls                 */ std::move(chat_templates),
                /* allow_image           */ mtmd_cap.vision,
                /* allow_audio           */ mtmd_cap.audio,
                /* allow_video           */ mtmd_cap.video,
                /* enable_thinking       */ enable_thinking,
                /* reasoning_budget      */ params_base.sampling.reasoning_budget_tokens,
                /* reasoning_budget_msg  */ params_base.sampling.reasoning_budget_message,
                /* media_path            */ params_base.media_path,
                /* force_pure_content    */ params_base.force_pure_content_parser
            };

            {
                auto caps = common_chat_templates_get_caps(chat_params.tmpls.get());
                auto it = params_base.default_template_kwargs.find("preserve_reasoning");
                bool supported = caps.at("supports_preserve_reasoning").get<bool>();
                bool specified = params_base.preserve_reasoning_specified;
                // note: the kwarg is enabled by default if not specified explicitly, so check the value
                bool enabled = it != params_base.default_template_kwargs.end() && it->second == "true";
                if (supported) {
                    SRV_TRC("preserve_reasoning kwarg: %s\n",
                            it == params_base.default_template_kwargs.end() ? "unset (template default)" : it->second.c_str());
                } else {
                    SRV_TRC("%s", "preserve_reasoning kwarg: not supported by template\n");
                }
                if (supported && !specified) {
                    SRV_WRN("%s", "chat template supports preserving reasoning, it is enabled by default (may use more tokens, disable via --no-reasoning-preserve)\n");
                }
                if (supported && !enabled) {
                    SRV_INF("%s", "chat template supports preserving reasoning, consider enabling it via --reasoning-preserve\n");
                }
                if (!supported && specified && enabled) {
                    SRV_WRN("%s", "chat template does NOT support preserving reasoning, --reasoning-preserve has no effect\n");
                }
            }
        }

        return true;
    }

    server_slot * get_slot_by_id(int id_slot) {
        // note: allow id_slot to be out of bounds (wrap around)
        id_slot = id_slot % slots.size();

        for (server_slot & slot : slots) {
            if (slot.id == id_slot) {
                return &slot;
            }
        }

        return nullptr;
    }

    server_slot * get_slot_by_cmpl_id(const std::string & cmpl_id) {
        if (cmpl_id.empty()) {
            return nullptr;
        }

        for (server_slot & slot : slots) {
            if (slot.is_processing() && slot.task && slot.task->params.oaicompat_cmpl_id == cmpl_id) {
                return &slot;
            }
        }

        return nullptr;
    }

    server_slot * get_available_slot(const server_task & task) {
        server_slot * ret = nullptr;

        bool update_cache = false;

        // if a specific slot is requested, use it (still goes through cache update logic below)
        if (task.id_slot != -1) {
            ret = get_slot_by_id(task.id_slot);
            if (ret) {
                // a busy slot is returned untouched, the caller defers the task
                if (ret->is_processing()) {
                    return ret;
                }

                SLT_INF(*ret, "selected slot by id (%d)\n", task.id_slot);
            }
        }

        // find the slot that has at least n% prompt similarity
        if (slot_prompt_similarity != 0.0f) {
            float f_sim_best = 0;

            for (server_slot & slot : slots) {
                if (task.id_slot != -1 && slot.id != task.id_slot) {
                    continue;
                }

                // skip the slot if it is not available
                if (slot.is_processing()) {
                    SLT_TRC(slot, " - skipping, is_processing = %d\n", slot.is_processing());
                    continue;
                }

                const auto & tokens = slot.prompt.tokens;

                // skip the slot if it does not contains cached tokens
                if (tokens.empty()) {
                    SLT_TRC(slot, "%s", " - skipping, slot is empty\n");
                    continue;
                }

                // fraction of the Longest Common Prefix length with respect to the input prompt length
                const size_t lcp_len = tokens.get_common_prefix(task.tokens);
                const float f_sim_cur = float(lcp_len) / task.tokens.size();

                SLT_TRC(slot, " - checking sim = %.3f (%zu/%zu) > %.3f\n", f_sim_cur, lcp_len, task.tokens.size(), slot_prompt_similarity);

                // select the current slot if the criteria match
                if (f_sim_cur > f_sim_best && f_sim_cur > slot_prompt_similarity) {
                    f_sim_best = f_sim_cur;

                    ret = &slot;
                }
            }

            if (ret != nullptr) {
                const float f_keep = (f_sim_best*task.tokens.size()) / ret->prompt.tokens.size();

                if (task.id_slot == -1) {
                    SLT_INF(*ret, "selected slot by LCP similarity, f_sim_best = %.3f (> %.3f thold), f_keep = %.3f\n",
                            f_sim_best, slot_prompt_similarity, f_keep);
                }

                // if we are about to lose a large portion of the existing context - save it in the prompt cache
                if (f_keep < 0.5f) {
                    update_cache = true;
                }
            }
        }

        // find an empty slot, or the one that has been least recently used
        if (ret == nullptr) {
            const auto free_slots = get_free_slots(1, -1);

            if (!free_slots.empty()) {
                ret = free_slots[0];

                SLT_INF(*ret, "selected slot by LRU, n_tokens = %d, t_last = %" PRId64 "\n", ret->prompt.n_tokens(), ret->t_last_used);

                update_cache = true;
            }
        }

        if (ret) {
            update_cache = update_cache && prompt_cache;

            // cache prompts only for the tasks that reuse them
            update_cache = update_cache && !task.is_stateless();

            if (update_cache) {
                SRV_TRC("%s", "updating prompt cache\n");

                const int64_t t_start = ggml_time_us();

                ret->prompt_save(*prompt_cache);

                if (!ret->prompt_load(*prompt_cache, task.tokens)) {
                    ret->prompt_clear();
                }

                prompt_cache->update();

                SRV_TRC("prompt cache update took %.2f ms\n", (ggml_time_us() - t_start) / 1000.0);
            }
        }

        return ret;
    }

    // return true if at least one slot has been cleared
    // TODO: improve logic
    //       - smarter decision which slot to clear (LRU or longest prompt?)
    //       - move slot to level 2 cache instead of removing?
    //       - instead of purging, try to store and resume later?
    bool try_clear_idle_slots() {
        bool res = false;

        if (!params_base.kv_unified) {
            return res;
        }

        for (auto & slot : slots) {
            if (slot.is_processing()) {
                continue;
            }

            if (slot.prompt.n_tokens() > 0) {
                SRV_WRN("purging slot %d with %zu tokens\n", slot.id, slot.prompt.tokens.size());

                slot.prompt_clear();

                res = true;

                // clear slots one by one
                break;
            }
        }

        return res;
    }

    std::vector<common_adapter_lora_info> construct_lora_list(const std::map<int, float> & config) const {
        std::vector<common_adapter_lora_info> output = params_base.lora_adapters; // copy
        for (size_t i = 0; i < output.size(); ++i) {
            auto it = config.find(i);
            if (it != config.end()) {
                output[i].scale = it->second;
            } else {
                output[i].scale = 0.0f;
            }
        }
        return output;
    }

    // [QLH-SPLIT] v5: decide, once per request, whether a remote tail may answer this request
    // with its top-k candidates instead of the whole logits row (993 KB -> 320 B per token; see
    // src/llama-ext.h).
    //
    // Called from launch_slot_with_task(), where the request's sampler has just been built: it is
    // the only moment where the whole sampling configuration is known and no decode has run yet,
    // and deciding up front is what keeps a request on one path from its first token to its last.
    // A per-token decision could only mix the two, and a runtime `token = -1` bail-out would mean
    // the fallback is discovered *after* the frame that needed the row was already answered.
    //
    // The rule itself lives in common/sampling.cpp (common_sampler_candidates_check) so that
    // there is exactly one copy of it; this function only gathers the request-level inputs and
    // writes the log line an operator needs to understand a slow request.
    bool plan_remote_candidates(server_slot & slot, const server_task & task) {
        slot.remote_cand_ok = false;
        slot.remote_cand_why.clear();

        if (!llama_context_split_has_remote(ctx_tgt)) {
            // A local chain and a whole model already hold the row; there is nothing to decide
            // and nothing worth printing.
            return false;
        }

        // Tasks that sample through their own path (TTS / realtime / transcription) read logits
        // for rows this channel was not built for; they keep the whole-row path silently, because
        // there is nothing to explain to an operator -- nothing changed for them.
        if (task.type != SERVER_TASK_TYPE_COMPLETION && task.type != SERVER_TASK_TYPE_INFILL) {
            return false;
        }

        const bool pre_sampling_logprobs =
                task.params.sampling.n_probs > 0 && !task.params.post_sampling_probs;

        common_sampler_candidates plan;
        std::string why;
        if (!common_sampler_candidates_check(slot.smpl.get(), true, pre_sampling_logprobs,
                                             spec != nullptr, plan, why)) {
            slot.remote_cand_why = why;
            SLT_INF(slot, "[split] this request takes the whole-logits path: %s\n", why.c_str());
            return false;
        }

        slot.remote_cand    = plan;
        slot.remote_cand.biases = nullptr; // owned by the copy below, never a pointer into smpl
        slot.remote_cand_bias.clear();
        if (plan.biases != nullptr) {
            slot.remote_cand_bias = *plan.biases;
        }
        slot.remote_cand_ok = true;

        std::string steps;
        for (uint32_t i = 0; i < plan.n_steps; ++i) {
            const char * name = plan.steps[i] == COMMON_CAND_STEP_PENALTIES   ? "penalties"
                              : plan.steps[i] == COMMON_CAND_STEP_TOP_N_SIGMA ? "top_n_sigma"
                              : plan.steps[i] == COMMON_CAND_STEP_TOP_K       ? "top_k" : "?";
            steps += steps.empty() ? name : std::string(" -> ") + name;
        }
        SLT_INF(slot, "[split] this request answers with %d candidates instead of the %d-token "
                      "logits row; the tail runs %s, this side runs the rest of the chain\n",
                plan.top_k, llama_vocab_n_tokens(vocab), steps.c_str());
        return true;
    }

    // [QLH-SPLIT] v7: decide, once per request, whether its speculative drafts may be verified
    // through the per-row channel, and -- because a remote chain has no other way to verify --
    // whether this request may speculate at all.
    //
    // Called from launch_slot_with_task(), next to plan_remote_candidates() and for the same
    // reason: the sampler has just been built, so the whole sampling configuration is known and
    // no decode has run. `slot.spec_blocked` is the half that changes behaviour rather than speed,
    // and it is the reason this cannot be a runtime fallback: the verify frame is built (the draft
    // tokens are in the batch) before any answer arrives, so "we cannot verify this one" has to be
    // decided before the drafter runs.
    void plan_remote_verify_rows(server_slot & slot, const server_task & task) {
        slot.remote_verify_ok  = false;
        slot.remote_verify_why.clear();
        slot.spec_blocked      = false;
        slot.spec_draft_cap    = INT32_MAX;

        if (!llama_context_split_has_remote(ctx_tgt)) {
            // A local chain and a whole model read the tail's rows in this process, so their
            // drafts are verified exactly as before; no cap and no channel.
            return;
        }

        // The chain can only verify through the per-row channel, so this is the draft cap for
        // every speculative request on it: a verify frame is draft.size() + 1 rows.
        slot.spec_draft_cap = (int32_t) qlh_split::MAX_VERIFY_ROWS - 1;

        if (spec == nullptr) {
            // No drafter configured: nothing to verify, nothing to block.
            return;
        }

        if (task.type != SERVER_TASK_TYPE_COMPLETION && task.type != SERVER_TASK_TYPE_INFILL) {
            // TTS / realtime / transcription sample through their own paths and do not speculate;
            // the drafts they would need are rows this channel does not describe.
            slot.spec_blocked = true;
            return;
        }

        const bool want_probs = task.params.sampling.n_probs > 0;

        std::string why;
        if (!common_sampler_verify_rows_check(slot.smpl.get(), true, want_probs, why)) {
            slot.remote_verify_why = why;
            // Not a fallback: there is no row to fall back to. A remote chain's tail answers one
            // row of logits (the last), which is not enough for a verification batch of k + 1 --
            // before this existed, such a request aborted the server in common_sampler_sample.
            slot.spec_blocked = true;
            SLT_INF(slot, "[split] speculative decoding is off for this request: %s "
                          "(a remote chain verifies drafts through the per-row token channel, "
                          "which this sampler cannot use)\n", why.c_str());
            return;
        }

        // [QLH-SPLIT] v9: the per-row channel is open to the reasoning budget as well as to a
        // grammar (v8), and the two verification paths below are not open to either. This is a
        // deliberate boundary, not an oversight:
        //
        //  - rejection sampling compares the target distribution with the draft's (draft_q), and
        //    accepts the *draft's* token when it wins -- it never asks the grammar or the
        //    reasoning budget, and common_sampler_accept() on a token either one rejects throws
        //    (llama_grammar_accept_token) or charges budget for a token the request never
        //    emitted. Teaching it to mask both distributions is a change to a
        //    distribution-correctness algorithm, and it is worth doing on its own.
        //  - synthetic acceptance accepts draft tokens at random *without* advancing either
        //    sampler (is_generated = false), which would leave the grammar describing a token
        //    history the request does not have and would leave rbudget's budget unspent for
        //    tokens it should have counted -- a silent divergence, not a slower request.
        //
        // A greedy chain with a grammar and/or a reasoning budget is the agent/tools case these
        // changes are for, and that case never reaches either branch (both need temp > 0), so
        // blocking here costs nothing that used to work: before v8 a grammar blocked the channel
        // entirely, and before v9 a reasoning budget did.
        const bool grammar_set  = common_sampler_has_grammar(slot.smpl.get());
        const bool rbudget_set  = common_sampler_has_reasoning_budget(slot.smpl.get());
        if ((grammar_set || rbudget_set) && (slot.use_spec_rejection() ||
                (spec && !common_speculative_get_synth_probs(spec.get()).empty()))) {
            slot.remote_verify_why = slot.use_spec_rejection()
                ? (grammar_set ? "grammar with rejection sampling" : "reasoning budget with rejection sampling")
                : (grammar_set ? "grammar with synthetic acceptance" : "reasoning budget with synthetic acceptance");
            slot.spec_blocked = true;
            SLT_INF(slot, "[split] speculative decoding is off for this request: %s "
                          "(the %s path cannot apply a grammar or a reasoning budget to the "
                          "draft's own tokens)\n",
                    slot.remote_verify_why.c_str(),
                    slot.use_spec_rejection() ? "rejection-sampling" : "synthetic-acceptance");
            return;
        }

        slot.remote_verify_ok = true;
        SLT_INF(slot, "[split] this request verifies its drafts through the per-row channel: one "
                      "argmax token per row crosses the wire (up to %d rows = %d drafts + the "
                      "target) instead of the %d-float logits row\n",
                slot.spec_draft_cap + 1, slot.spec_draft_cap, llama_vocab_n_tokens(vocab));
    }

    bool launch_slot_with_task(server_slot & slot, server_task && task) {
        // process per-request lora adapters
        if (!task.params.lora.empty()) {
            auto task_loras = construct_lora_list(task.params.lora);
            if (!are_lora_equal(task_loras, slot.lora)) {
                // if lora has changed, check to see if the cache should be cleared
                if (lora_should_clear_cache(slot.lora, task_loras)) {
                    SLT_TRC(slot, "clearing cache for lora change. %zu loras -> %zu loras\n", slot.lora.size(), task.params.lora.size());
                    slot.prompt.clear();
                } else {
                    SLT_TRC(slot, "keeping cache for alora. %zu target loras\n", task_loras.size());
                }
                slot.lora = task_loras;
            }
        } else {
            slot.lora = params_base.lora_adapters;
        }

        // if using alora, make sure it's only a single one requested and active
        size_t alora_invocation_start = task.tokens.size();
        if (lora_all_alora(slot.lora)) {
            const auto & enabled_ids = lora_get_enabled_ids(slot.lora);
            // TODO: This will error out if a user requests two aloras, but only
            // provides the activation string for one. We could, instead search
            // for all requested alora activation strings and then either keep
            // only the last one, or reject if multiple are found.
            if (enabled_ids.size() != 1) {
                send_error(task, "Cannot run multiple aLoRAs in a single request", ERROR_TYPE_INVALID_REQUEST);
                return false;
            }
            const auto & lora = slot.lora[enabled_ids[0]].ptr;

            // get the pointer and count for the invocation tokens
            const uint64_t      n_invocation_tokens = llama_adapter_get_alora_n_invocation_tokens(lora);
            const llama_token * invocation_tokens   = llama_adapter_get_alora_invocation_tokens  (lora);

            // scan backwards through the prompt tokens to find the last
            // occurrence of the invocation sequence
            int match_idx = static_cast<int>(n_invocation_tokens) - 1;
            for (int i = task.tokens.size() - 1; i >= 0; --i) {
                // the token in this position matches the next token to find in
                // the invocation sequence
                if (task.tokens[i] == invocation_tokens[match_idx]) {
                    // if it's a full match, we've found the start
                    if (match_idx == 0) {
                        alora_invocation_start = i;
                        break;
                    }
                    // otherwise, check the next token in the sequence
                    --match_idx;
                } else {
                    // no match in this position, so start looking over again
                    match_idx = static_cast<int>(n_invocation_tokens) - 1;
                }
            }

            // if the activation string is not found, disable the alora
            if (alora_invocation_start == task.tokens.size()) {
                SLT_DBG(slot, "alora %zu requested, but not found. deactivating\n", enabled_ids[0]);
                slot.lora[enabled_ids[0]].scale = 0.0f;
            } else {
                SLT_DBG(slot, "alora %zu activated starting at %zu\n", enabled_ids[0], alora_invocation_start);
                slot.alora_invocation_start = alora_invocation_start;
            }
        }

        if (!task.tokens.validate(ctx_tgt)) {
            send_error(task, "Prompt contains invalid tokens", ERROR_TYPE_INVALID_REQUEST);
            return false;
        }

        SLT_DBG(slot, "launching slot : %s\n", safe_json_to_str(slot.to_json()).c_str());

        if (task.type == SERVER_TASK_TYPE_TTS) {
            GGML_ASSERT(has_cap_tts()); // should already checked in route handler
            if (!slot.tts.is_supported()) {
                slot.tts.ctx.init(ctx_tgt, slot.mctx);
            }
            // mtmd_helper always re-eval the whole prompt
            slot.prompt_clear();
            task.tts_inp.data.seq_id = slot.id;
            if (slot.tts.ctx.set_input(task.tts_inp.get()) != 0) {
                send_error(task, "failed to process TTS prompt", ERROR_TYPE_SERVER);
                return false;
            }
        }

        // realtime transcription: the rows come from the audio stream, the helper may bias some tokens
        if (task.type == SERVER_TASK_TYPE_TRANSCRIBE_RT) {
            auto sparams = mtmd_audio_stream_params_default();
            sparams.prompt   = asr_prompt.data();
            sparams.n_prompt = asr_prompt.size();
            slot.audio_stream.reset(mtmd_audio_stream_init(mctx, sparams));
            if (!slot.audio_stream) {
                send_error(task, "failed to initialize the audio stream", ERROR_TYPE_SERVER);
                return false;
            }
            if (!asr_prompt.empty()) {
                for (const auto & b : mtmd_helper::audio_stream_logit_bias(mctx, model_tgt)) {
                    task.params.sampling.logit_bias.push_back(b);
                }
            }
        }

        // initialize samplers
        if (task.need_sampling()) {
            try {
                slot.smpl.reset(common_sampler_init(model_tgt, task.params.sampling));
            } catch (std::exception & e) {
                std::string err_msg = std::string("Failed to initialize samplers: ") + e.what();
                send_error(task, err_msg, ERROR_TYPE_INVALID_REQUEST);
                return false;
            }

            const bool need_pre_sample_logits = task.params.sampling.n_probs > 0 && !task.params.post_sampling_probs;

            bool use_backend_sampling = task.params.sampling.backend_sampling;

            // TODO: getting pre sampling logits is not yet supported with backend sampling
            use_backend_sampling &= !need_pre_sample_logits;

            // TODO: check verify if this actually works with TTS
            use_backend_sampling &= task.type != SERVER_TASK_TYPE_TTS;

            // TODO: tmp until backend sampling is fully implemented
            if (use_backend_sampling) {
                llama_set_sampler(ctx_tgt, slot.id, common_sampler_get(slot.smpl.get()));
            } else {
                llama_set_sampler(ctx_tgt, slot.id, nullptr);
            }

            SLT_TRC(slot, "sampler chain: %s\n", common_sampler_print(slot.smpl.get()).c_str());
            SLT_TRC(slot, "sampler params: \n%s\n", task.params.sampling.print().c_str());

            // [QLH-SPLIT] v5: the remote candidate channel, decided here and only here.
            plan_remote_candidates(slot, task);

            // [QLH-SPLIT] v7: and the per-row verify channel, which also decides whether this
            // request may speculate at all on a remote chain.
            plan_remote_verify_rows(slot, task);

            if (spec && !common_speculative_get_synth_probs(spec.get()).empty()) {
                const uint32_t seed = task.params.sampling.seed == LLAMA_DEFAULT_SEED
                    ? std::random_device{}()
                    : task.params.sampling.seed;
                slot.spec_synth_rng.seed(seed);
            }
        } else {
            slot.smpl.reset();
            // [QLH-SPLIT] v5: a non-sampling request (embedding, rerank) leaves the candidate
            // channel off, so nothing later can sample from a previous request's candidates.
            slot.remote_cand_ok = false;
            // [QLH-SPLIT] v7: and off for the per-row verify channel too (see above).
            slot.remote_verify_ok = false;
            slot.remote_verify_why.clear();
        }

        // the per-request limit takes priority over the global one
        slot.n_predict_max = task.params.n_predict != -1 ? task.params.n_predict : params_base.n_predict;

        slot.task = std::make_unique<const server_task>(std::move(task));

        if (slot.task->type == SERVER_TASK_TYPE_TTS) {
            slot.state = SLOT_STATE_PROCESSING_PROMPT;
        } else {
            slot.state = slot.task->is_child()
                ? SLOT_STATE_WAIT_OTHER // wait for the parent to process prompt
                : SLOT_STATE_STARTED;
        }

        // no prompt, the rows come from the audio stream, which audio controls reach from now on
        if (slot.task->type == SERVER_TASK_TYPE_TRANSCRIBE_RT) {
            slot.prompt_clear();
            slot.init_sampler();
            slot.sampled = LLAMA_TOKEN_NULL;
            slot.i_batch = -1;
            slot.stats.update_prompt_start();
            slot.state = SLOT_STATE_GENERATING;
            send_partial_response(slot, {}, false, true);
        }

        // no prompt either, the session appends text and audio and asks for responses with its controls
        if (slot.task->type == SERVER_TASK_TYPE_REALTIME) {
            slot.rt = std::make_unique<server_realtime_slot>(ctx_tgt, mctx, slot.id, slot.n_ctx, slot.task->params.sampling,
                                                             queue_results, slot.task->id, slot.task->index);
            slot.prompt_clear();
            slot.stats.update_prompt_start();
            slot.state      = SLOT_STATE_GENERATING;
            slot.audio_wait = true;
            send_partial_response(slot, {}, false, true);
        }

        // reset server kill-switch counter
        n_empty_consecutive = 0;

        SLT_INF(slot, "processing task, is_child = %d\n", slot.task->is_child());
        return true;
    }

    bool process_token(completion_token_output & result, server_slot & slot) {
        // remember which tokens were sampled - used for repetition penalties during sampling
        const std::string token_str = result.text_to_send;
        slot.sampled = result.tok;

        slot.generated.append(token_str, result.tok);
        if (slot.task->params.return_tokens) {
            slot.generated_tokens.push_back(result.tok);
        }
        slot.has_next_token = true;

        // check if there is incomplete UTF-8 character at the end
        bool incomplete = validate_utf8(slot.generated.text) < slot.generated.text.size();

        // search stop word and delete it
        if (!incomplete) {
            size_t pos = std::min(slot.n_sent_text, slot.generated.text.size());

            const std::string str_test = slot.generated.text.substr(pos);
            bool send_text = true;

            size_t stop_pos = slot.find_stopping_strings(str_test, token_str.size(), true);
            if (stop_pos != std::string::npos) {
                slot.generated.truncate(pos + stop_pos);
                pos = std::min(slot.n_sent_text, slot.generated.text.size());
            } else if (slot.has_next_token && !llama_vocab_is_eog(vocab, result.tok) ) {
                stop_pos = slot.find_stopping_strings(str_test, token_str.size(), false);
                send_text = stop_pos == std::string::npos;
            }

            // check if there is any token to predict
            if (send_text) {
                // no send the stop word in the response
                result.text_to_send = slot.generated.text.substr(pos, std::string::npos);
                slot.n_sent_text += result.text_to_send.size();
                // add the token to slot queue and cache
            } else {
                result.text_to_send = "";
            }

            slot.add_token(result);
            if (slot.task->params.stream) {
                send_partial_response(slot, result, false);
            }
        }

        if (incomplete) {
            slot.has_next_token = true;
        }

        // if context shifting is disabled, make sure that we don't run out of context
        if (!params_base.ctx_shift && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
            slot.truncated      = true;
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped due to running out of context capacity, prompt.n_tokens() = %d, task.n_tokens = %d, n_gen = %d, n_ctx = %d\n",
                    slot.prompt.n_tokens(), slot.task->n_tokens(), (int) slot.stats.n_gen, slot.n_ctx);
        }

        // check the limits
        if (slot.stats.n_gen > 0 && slot.has_next_token && !slot.has_budget()) {
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped by limit, n_gen = %d, n_predict = %d\n", (int) slot.stats.n_gen, slot.task->params.n_predict);
        }

        if (slot.has_new_line) {
            // require that each new line has a whitespace prefix (i.e. indentation) of at least slot.params.n_indent
            if (slot.task->params.n_indent > 0) {
                // check the current indentation
                // TODO: improve by not doing it more than once for each new line
                if (slot.last_nl_pos > 0) {
                    size_t pos = slot.last_nl_pos;

                    int n_indent = 0;
                    while (pos < slot.generated.text.size() && (slot.generated.text[pos] == ' ' || slot.generated.text[pos] == '\t')) {
                        n_indent++;
                        pos++;
                    }

                    if (pos < slot.generated.text.size() && n_indent < slot.task->params.n_indent) {
                        slot.stop           = STOP_TYPE_LIMIT;
                        slot.has_next_token = false;

                        // cut the last line
                        slot.generated.truncate(pos);

                        SLT_DBG(slot, "stopped by indentation limit, n_gen = %d, n_indent = %d\n", (int) slot.stats.n_gen, n_indent);
                    }
                }

                // find the next new line
                {
                    const size_t pos = slot.generated.text.find('\n', slot.last_nl_pos);

                    if (pos != std::string::npos) {
                        slot.last_nl_pos = pos + 1;
                    }
                }
            }
        }

        // check if there is a new line in the generated text
        if (result.text_to_send.find('\n') != std::string::npos) {
            slot.has_new_line = true;

            // if we have seen a new line, we stop after a certain time limit, but only upon another new line
            if (slot.task->params.t_max_predict_ms > 0 && slot.stats.t_gen_ms() > slot.task->params.t_max_predict_ms) {
                slot.stop           = STOP_TYPE_LIMIT;
                slot.has_next_token = false;

                SLT_DBG(slot, "stopped by time limit, n_gen = %d, t_max_predict_ms = %d ms\n", (int) slot.stats.n_gen, (int) slot.task->params.t_max_predict_ms);
            }
        }

        if (llama_vocab_is_eog(vocab, result.tok)) {
            slot.stop           = STOP_TYPE_EOS;
            slot.has_next_token = false;

            SLT_DBG(slot, "%s", "stopped by EOS\n");
        }

        SLT_DBG(slot, "n_gen = %d, n_remaining = %d, next token: %5d '%s'\n", (int) slot.stats.n_gen, slot.n_remaining(), result.tok, token_str.c_str());

        return slot.has_next_token; // continue
    }

    void populate_token_probs(const server_slot & slot, completion_token_output & result, bool post_sampling, bool special, int idx) const {
        const size_t n_probs_request = slot.task->params.sampling.n_probs;

        if (post_sampling) {
            const auto * cur_p = common_sampler_get_candidates(slot.smpl.get(), true);
            const size_t max_probs = cur_p->size;
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                if (cur_p->data[i].id == result.tok) {
                    result.prob = cur_p->data[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                // Some samplers do return 0.0 probabilities, others don't.
                // Filter 0.0 probailities, to ensure the behavior is consistent.
                if (cur_p->data[i].p == 0.0) {
                    break;
                }

                result.probs.push_back({
                    cur_p->data[i].id,
                    common_token_to_piece(ctx_tgt, cur_p->data[i].id, special),
                    cur_p->data[i].p
                });
            }
        } else {
            std::vector<llama_token_data> cur = get_token_probabilities(ctx_tgt, idx, n_probs_request);
            const size_t max_probs = cur.size();
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                // set probability for sampled token
                if (cur[i].id == result.tok) {
                    result.prob = cur[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                result.probs.push_back({
                    cur[i].id,
                    common_token_to_piece(ctx_tgt, cur[i].id, special),
                    cur[i].p
                });
            }
        }
    }

    void send_error(const server_task & task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(task.id, error, type);
    }

    void send_error(const server_slot & slot, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(slot.task->id, error, type, slot.task->n_tokens(), slot.n_ctx);
    }

    void send_error(const int id_task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER, const int32_t n_prompt_tokens = 0, const int32_t n_ctx = 0) {
        SRV_ERR("task id = %d, error: %s\n", id_task, error.c_str());

        if (type == ERROR_TYPE_EXCEED_CONTEXT_SIZE) {
            GGML_ASSERT(n_ctx > 0 && n_prompt_tokens > 0);
        }

        auto res = std::make_unique<server_task_result_error>();
        res->id              = id_task;
        res->err_type        = type;
        res->err_msg         = error;
        res->n_prompt_tokens = n_prompt_tokens;
        res->n_ctx           = n_ctx;

        queue_results.send(std::move(res));
    }

    void send_partial_response(server_slot & slot, const completion_token_output & tkn, bool is_progress, bool is_begin = false) {
        auto res = std::make_unique<server_task_result_cmpl_partial>();

        res->id    = slot.task->id;
        res->index = slot.task->index;

        if (is_progress) {
            res->is_progress        = true;
            res->progress.total     = slot.task->n_tokens();
            res->progress.cache     = slot.stats.n_prompt_cached;
            res->progress.processed = slot.prompt.tokens.size();
            res->progress.time_ms   = slot.stats.t_elapsed_us() / 1000;
        }
        if (is_begin) {
            res->is_begin = true;
        } else {
            res->content = slot.generated.substr(slot.n_sent_text - tkn.text_to_send.size(), tkn.text_to_send.size());
            res->tokens.assign(1, tkn.tok);
        }

        res->n_decoded             = slot.stats.n_gen;
        res->n_prompt_tokens       = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.stats.n_prompt_cached;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            res->prob_output = tkn; // copy the token probs
        }

        // populate timings if this is final response or timings_per_token is enabled
        if (slot.stop != STOP_TYPE_NONE || slot.task->params.timings_per_token) {
            res->stats = slot.stats;
        }

        queue_results.send(std::move(res));
    }

    void send_tts_result(server_slot & slot, int32_t sample_rate, const char * data, size_t data_len, bool final) {
        auto res = std::make_unique<server_task_result_tts>();

        res->id          = slot.task->id;
        res->index       = slot.task->index;
        res->sample_rate = sample_rate;
        res->audio.assign(data, data_len);
        res->final       = final;

        queue_results.send(std::move(res));
    }

    void send_final_response(server_slot & slot) {
        auto res = std::make_unique<server_task_result_cmpl_final>();

        res->id      = slot.task->id;
        res->id_slot = slot.id;

        res->index = slot.task->index;

        // keep copy of last generated text for debugging purposes
        if (slots_debug) {
            slot.debug_generated_text = slot.generated.text;
        }

        // in stream mode, content and tokens are already in last partial chunk
        if (slot.task->params.stream) {
            res->content     = {};
            res->tokens      = llama_tokens{};
        } else {
            res->content     = std::move(slot.generated);
            res->tokens      = std::move(slot.generated_tokens);
        }
        res->stats           = slot.stats;
        res->prompt          = slot.task->tokens.detokenize(ctx_tgt, true);
        res->response_fields = std::move(slot.task->params.response_fields);

        res->truncated             = slot.truncated;
        res->n_decoded             = slot.stats.n_gen;
        res->n_prompt_tokens       = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.stats.n_prompt_cached;
        res->n_tokens_cached       = slot.prompt.n_tokens();
        res->has_new_line          = slot.has_new_line;
        res->stopping_word         = slot.stopping_word;
        res->stop                  = slot.stop;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->stream            = slot.task->params.stream;
        res->include_usage     = slot.task->params.include_usage;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            if (!slot.task->params.stream && slot.stop == STOP_TYPE_WORD) {
                const llama_tokens stop_word_toks = common_tokenize(ctx_tgt, slot.stopping_word, false);

                size_t safe_offset = std::min(slot.generated_token_probs.size(), stop_word_toks.size());
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end() - safe_offset);
            } else {
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end());
            }
        }

        res->generation_params = slot.task->params; // copy the parameters

        queue_results.send(std::move(res));
    }

    void send_embedding(const server_slot & slot, const common_batch & batch) {
        auto res = std::make_unique<server_task_result_embd>();
        res->id        = slot.task->id;
        res->index     = slot.task->index;
        res->n_tokens  = slot.task->n_tokens();
        res->res_type  = slot.task->params.res_type;

        const int n_embd_out = llama_model_n_embd_out(model_tgt);

        std::vector<float> embd_res(n_embd_out, 0.0f);

        for (int i = 0; i < batch.size(); ++i) {
            if (!batch.tokens[i].output || batch.tokens[i].seq_id != slot.id) {
                continue;
            }

            const float * embd = nullptr;
            if (llama_pooling_type(slot.ctx_tgt) == LLAMA_POOLING_TYPE_NONE) {
                embd = llama_get_embeddings_ith(slot.ctx_tgt, i);
            } else {
                embd = llama_get_embeddings_seq(slot.ctx_tgt, batch.tokens[i].seq_id);
            }

            if (embd == nullptr) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.tokens[i].id, batch.tokens[i].seq_id);

                res->embedding.push_back(std::vector<float>(n_embd_out, 0.0f));
                continue;
            }

            // normalize only when there is pooling
            if (llama_pooling_type(slot.ctx_tgt) != LLAMA_POOLING_TYPE_NONE) {
                common_embd_normalize(embd, embd_res.data(), n_embd_out, slot.task->params.embd_normalize);
                res->embedding.push_back(embd_res);
                break;
            }

            res->embedding.emplace_back(embd, embd + n_embd_out);
        }

        SLT_DBG(slot, "%s", "sending embeddings\n");

        queue_results.send(std::move(res));
    }

    void send_decision(const server_slot & slot, const common_batch & batch, int32_t i_batch) {
        auto res = std::make_unique<server_task_result_decision>();
        res->id       = slot.task->id;
        res->index    = slot.task->index;
        res->n_tokens = slot.task->n_tokens();

        const auto & decision = slot.task->decision;

        if (decision.pooled) {
            const float * embd = llama_get_embeddings_seq(slot.ctx_tgt, slot.id);
            if (embd == nullptr) {
                send_error(slot, "failed to get the class scores", ERROR_TYPE_SERVER);
                return;
            }
            res->scores.assign(embd, embd + llama_model_n_embd_out(model_tgt));
        } else if (!decision.labels.empty()) {
            const float * logits = llama_get_logits_ith(slot.ctx_tgt, i_batch);
            if (logits == nullptr) {
                send_error(slot, "failed to get logits", ERROR_TYPE_SERVER);
                return;
            }
            const int32_t n_vocab = llama_vocab_n_tokens(vocab);
            for (const llama_token label : decision.labels) {
                GGML_ASSERT(label >= 0 && label < n_vocab);
                res->scores.push_back(logits[label]);
            }
            if (!decision.label_groups.empty()) {
                std::vector<float> scores;
                size_t i = 0;
                for (const int32_t n : decision.label_groups) {
                    GGML_ASSERT(n > 0 && i + n <= res->scores.size());
                    scores.push_back(*std::max_element(res->scores.begin() + i, res->scores.begin() + i + n));
                    i += n;
                }
                res->scores = std::move(scores);
            }
        } else {
            // the outputs of this slot in this batch are the last tokens of the prompt
            std::vector<int32_t> idx;
            for (int i = 0; i < batch.size(); ++i) {
                if (batch.tokens[i].output && batch.tokens[i].seq_id == slot.id) {
                    idx.push_back(i);
                }
            }
            const int32_t pos_first = slot.prompt.n_tokens() - (int32_t) idx.size();
            auto get_embd = [&](int32_t pos) -> const float * {
                const int32_t i = pos - pos_first;
                return i >= 0 && i < (int32_t) idx.size() ? llama_get_embeddings_ith(slot.ctx_tgt, idx[i]) : nullptr;
            };

            // joint head (decision model): the scores are the first rows
            for (int32_t i = 0; i < decision.n_scores; i++) {
                const float * embd = i < (int32_t) idx.size() ? llama_get_embeddings_ith(slot.ctx_tgt, idx[i]) : nullptr;
                if (embd == nullptr) {
                    send_error(slot, "failed to get embeddings", ERROR_TYPE_SERVER);
                    return;
                }
                res->scores.push_back(embd[0]);
            }

            const int32_t n_embd_out = llama_model_n_embd_out(model_tgt);
            const int32_t n_pointer  = n_embd_out / 2;
            const float * embd_q = decision.pointer >= 0 ? get_embd(decision.pointer) : nullptr;
            GGML_ASSERT(decision.column >= 0 && decision.column < n_embd_out);

            for (const int32_t marker : decision.markers) {
                const float * embd = get_embd(marker);
                if (embd == nullptr || (decision.pointer >= 0 && embd_q == nullptr)) {
                    send_error(slot, "failed to get embeddings, the question and its options must fit in one batch", ERROR_TYPE_SERVER);
                    return;
                }
                if (decision.pointer < 0) {
                    res->scores.push_back(embd[decision.column]);
                    continue;
                }
                float dot = 0.0f;
                for (int32_t i = 0; i < n_pointer; i++) {
                    dot += embd_q[i] * embd[n_pointer + i];
                }
                res->scores.push_back(dot / sqrtf((float) n_pointer));
            }
        }

        SLT_DBG(slot, "%s", "sending decision result\n");

        queue_results.send(std::move(res));
    }

    void send_rerank(const server_slot & slot, const common_batch & batch) {
        auto res = std::make_unique<server_task_result_rerank>();
        res->id       = slot.task->id;
        res->index    = slot.task->index;
        res->n_tokens = slot.task->n_tokens();

        for (int i = 0; i < batch.size(); ++i) {
            if (!batch.tokens[i].output || batch.tokens[i].seq_id != slot.id) {
                continue;
            }

            const float * embd = llama_get_embeddings_seq(ctx_tgt, batch.tokens[i].seq_id);
            if (embd == NULL) {
                embd = llama_get_embeddings_ith(ctx_tgt, i);
            }

            if (embd == NULL) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.tokens[i].id, batch.tokens[i].seq_id);

                res->score = -1e6;
                continue;
            }

            res->score = embd[0];
        }

        SLT_DBG(slot, "sending rerank result, res.score = %f\n", res->score);

        queue_results.send(std::move(res));
    }

    //
    // Functions to process the task
    //

    // tokenize the input if it's set by CLI, return false on error
    bool tokenize_cli_input(server_task & task) {
        try {
            auto & prompt = task.cli_prompt;
            if (mctx != nullptr) {
                task.tokens = process_mtmd_prompt(mctx, prompt, task.cli_files, init_opt);
            } else {
                task.tokens = std::move(tokenize_input_prompts(vocab, mctx, prompt, true, true, init_opt)[0]);
            }
            task.cli_prompt.clear();
            task.cli_files.clear();
        } catch (const std::exception & e) {
            send_error(task, std::string("Failed to format input: ") + e.what(), ERROR_TYPE_INVALID_REQUEST);
            return false;
        }
        return true;
    }

    // idle slots that lose the least by being reused: empty slots first, then the least recently used ones
    std::vector<server_slot *> get_free_slots(size_t n_slots_needed, int exclude_id_slot) {
        std::vector<server_slot *> free_slots;
        for (auto & slot : slots) {
            if (!slot.is_processing() && slot.id != exclude_id_slot) {
                free_slots.push_back(&slot);
            }
        }

        std::stable_sort(free_slots.begin(), free_slots.end(), [](const server_slot * a, const server_slot * b) {
            if ((a->prompt.n_tokens() == 0) != (b->prompt.n_tokens() == 0)) {
                return a->prompt.n_tokens() == 0;
            }
            return a->t_last_used < b->t_last_used;
        });

        free_slots.resize(std::min(free_slots.size(), n_slots_needed));

        return free_slots;
    }

    // launch multiple slots for parent + child tasks
    bool launch_slots_with_parent_task(server_slot & parent_slot, std::vector<server_slot *> & child_slots, server_task && parent_task) {
        GGML_ASSERT(!parent_slot.is_processing());
        GGML_ASSERT(parent_task.is_parent());
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());

        int id_parent = parent_task.id;

        SRV_TRC("launching slots for parent task id_task = %d with %zu child tasks\n", id_parent, parent_task.child_tasks.size());

        // to be called in case of failure to release all launched slots
        auto release_slots = [this, id_parent]() {
            for (auto & slot : slots) {
                if (slot.is_processing() && (
                        slot.task->id == id_parent ||
                        slot.task->id_parent == id_parent
                )) {
                    slot.release();
                }
            }
        };

        // launch all child tasks first
        size_t idx = 0;
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());
        for (auto * slot : child_slots) {
            // keep the state that the child task is about to overwrite
            if (prompt_cache && slot->prompt_save(*prompt_cache)) {
                prompt_cache->update();
            }

            int id_child = parent_task.child_tasks[idx].id;
            if (!launch_slot_with_task(*slot, std::move(parent_task.child_tasks[idx]))) {
                SRV_ERR("failed to launch slot with child task, id_task = %d\n", id_child);
                release_slots();
                return false;
            }
            idx++;
        }

        // finally, launch the parent task
        if (!launch_slot_with_task(parent_slot, std::move(parent_task))) {
            SRV_ERR("failed to launch slot with task, id_task = %d\n", id_parent);
            release_slots();
            return false;
        }

        return true;
    }

    // n_tokens_cur: the number of tokens added to the batch for the current slot
    void create_checkpoint(server_slot & slot, const int64_t n_tokens_cur, llama_pos pos_min, llama_pos pos_max) {
        const int id_task = slot.task->id;

        // [QLH-SPLIT] Nothing to checkpoint when this context cannot export its sequence state
        // (a chain with remote stages; see llama_context::state_seq_get_size). An empty
        // checkpoint loads as a success and rewinds the slot to a position the chain no longer
        // holds, so leaving it out keeps the server on its existing "no cache data, re-process
        // the prompt" path -- the only correct recovery for a chain that cannot restore its
        // tails.
        if (llama_state_seq_get_size_ext(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            return;
        }

        // evict checkpoints within min-step of a previous checkpoint, unless they were
        // created by the current task
        // only when the list is full, otherwise short prompts keep just the oldest checkpoint
        int64_t last = -1;
        for (auto it = slot.prompt.checkpoints.begin();
                slot.prompt.checkpoints.size() + 1 >= (size_t) params_base.n_ctx_checkpoints &&
                it != slot.prompt.checkpoints.end(); ) {
            const auto & cur = **it;
            if (cur.id_task != id_task && last >= 0 && cur.n_tokens <= last + params_base.checkpoint_min_step) {
                SLT_TRC(slot, "erasing context checkpoint too close to an earlier one (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                        cur.pos_min, cur.pos_max, cur.n_tokens, (float) cur.size() / 1024 / 1024);

                it = slot.prompt.checkpoints.erase(it);
                continue;
            }

            last = cur.n_tokens;
            ++it;
        }

        while (slot.prompt.checkpoints.size() >= (size_t) params_base.n_ctx_checkpoints) {
            // make room for the new checkpoint, if needed
            const auto & cur = *slot.prompt.checkpoints.front();

            SLT_WRN(slot, "erasing old context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                    cur.pos_min, cur.pos_max, cur.n_tokens, (float) cur.size() / 1024 / 1024);

            slot.prompt.checkpoints.erase(slot.prompt.checkpoints.begin());
        }

        // replace an existing checkpoint at the same n_tokens instead of appending a duplicate
        {
            const int64_t n_tokens_new = slot.prompt.n_tokens() - n_tokens_cur;
            for (auto it = slot.prompt.checkpoints.begin(); it != slot.prompt.checkpoints.end(); ) {
                if ((*it)->n_tokens == n_tokens_new) {
                    SLT_TRC(slot, "superseding context checkpoint at n_tokens = %" PRId64 "\n", (*it)->n_tokens);
                    it = slot.prompt.checkpoints.erase(it);
                } else {
                    ++it;
                }
            }
        }

        auto ckpt = std::make_shared<common_prompt_checkpoint>();
        auto & cur = *ckpt;

        cur.id_task = id_task;

        // [TAG_CHECKPOINTS_FIX_POS_MIN]
        // TODO: here we incorrectly deterimne that the saved checkpoint data covers the [pos_min, pos_max] range
        //       this is not true for SWA models: https://github.com/ggml-org/llama.cpp/pull/24411#issuecomment-4677983225
        cur.update_pos(slot.prompt.n_tokens() - n_tokens_cur, pos_min, pos_max);

        const bool ckpt_tgt_ok = cur.update_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        const bool ckpt_dft_ok = cur.update_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        if (!ckpt_tgt_ok || !ckpt_dft_ok) {
            // [QLH-SPLIT] No checkpoint this round, instead of aborting the server. Keeping the
            // entry would be worse than dropping it: load_tgt()/load_dft() treat an empty blob
            // as "nothing to restore" and report success, so a half-captured checkpoint would
            // rewind the prompt bookkeeping while leaving the KV where it is. Dropping it costs
            // one prompt re-processing later, which is exactly what "no checkpoint" means.
            slot.prompt.checkpoints.pop_back();
            SLT_WRN(slot, "%s", "context checkpoint could not be captured this round; not reusing it\n");
            return;
        }
        // stash the draft's speculative state with the checkpoint
        common_speculative_get_state(spec.get(), slot.id, cur.data_spec);

        slot.prompt.checkpoints.push_back(std::move(ckpt));

        SLT_TRC(slot,
                "created context checkpoint %d of %d (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                (int) slot.prompt.checkpoints.size(), params_base.n_ctx_checkpoints, cur.pos_min,
                cur.pos_max, cur.n_tokens, (float) cur.size() / 1024 / 1024);
    }

    // checkpoints are appended to the slot save file, after the llama state payload
    // they cannot be recreated from the final state alone (a recurrent state cannot be rewound)
    static constexpr uint32_t SLOT_CKPT_MAGIC   = 0x504b4353; // "SCKP"
    static constexpr uint32_t SLOT_CKPT_VERSION = 1;

    static bool ckpt_read(std::ifstream & ifs, void * dst, size_t size, size_t & n_read) {
        if (!ifs.read((char *) dst, size)) {
            return false;
        }
        n_read += size;
        return true;
    }

    template <typename Buf>
    static bool ckpt_read_buf(std::ifstream & ifs, Buf & buf, size_t n_avail, size_t & n_read) {
        uint64_t n = 0;
        // check the size against the bytes left in the file before allocating, the size field may be corrupted
        if (!ckpt_read(ifs, &n, sizeof(n), n_read) || n > n_avail - n_read) {
            return false;
        }
        buf.resize(n);
        return n == 0 || ckpt_read(ifs, buf.data(), n, n_read);
    }

    static void ckpt_write(std::ofstream & ofs, const void * src, size_t size, size_t & n_written) {
        ofs.write((const char *) src, size);
        n_written += size;
    }

    template <typename Buf>
    static void ckpt_write_buf(std::ofstream & ofs, const Buf & buf, size_t & n_written) {
        const uint64_t n = buf.size();
        ckpt_write(ofs, &n, sizeof(n), n_written);
        if (n > 0) {
            ckpt_write(ofs, buf.data(), n, n_written);
        }
    }

    // returns false if the appendix could not be written completely
    bool save_slot_checkpoints(const std::string & filepath, const server_slot & slot, size_t & n_written) const {
        n_written = 0;
        if (slot.prompt.checkpoints.empty()) {
            return true;
        }
        std::ofstream ofs(filepath, std::ios::binary | std::ios::app);
        if (!ofs) {
            SLT_WRN(slot, "failed to append context checkpoints to '%s'\n", filepath.c_str());
            return false;
        }
        const uint32_t magic   = SLOT_CKPT_MAGIC;
        const uint32_t version = SLOT_CKPT_VERSION;
        const uint32_t count   = (uint32_t) slot.prompt.checkpoints.size();
        ckpt_write(ofs, &magic,   sizeof(magic),   n_written);
        ckpt_write(ofs, &version, sizeof(version), n_written);
        ckpt_write(ofs, &count,   sizeof(count),   n_written);
        for (const auto & ckpt : slot.prompt.checkpoints) {
            const auto & cur = *ckpt;
            ckpt_write(ofs, &cur.n_tokens, sizeof(cur.n_tokens), n_written);
            ckpt_write(ofs, &cur.pos_min,  sizeof(cur.pos_min),  n_written);
            ckpt_write(ofs, &cur.pos_max,  sizeof(cur.pos_max),  n_written);
            ckpt_write_buf(ofs, cur.data_tgt,  n_written);
            ckpt_write_buf(ofs, cur.data_dft,  n_written);
            ckpt_write_buf(ofs, cur.data_spec, n_written);
        }
        ofs.flush();
        if (!ofs) {
            SLT_WRN(slot, "failed to append context checkpoints to '%s' - the appendix is incomplete\n", filepath.c_str());
            return false;
        }
        SLT_INF(slot, "appended %u context checkpoint(s) (%.3f MiB) to '%s'\n",
                count, (float) n_written / 1024 / 1024, filepath.c_str());
        return true;
    }

    // returns the number of bytes consumed, 0 if there is no usable appendix
    size_t load_slot_checkpoints(const std::string & filepath, size_t offset, server_slot & slot) const {
        std::ifstream ifs(filepath, std::ios::binary | std::ios::ate);
        const size_t file_size = ifs ? (size_t) ifs.tellg() : 0;
        if (!ifs || file_size < offset || !ifs.seekg(offset)) {
            return 0;
        }
        const size_t n_avail = file_size - offset; // bytes after the llama state payload
        size_t n_read = 0;
        uint32_t magic   = 0;
        uint32_t version = 0;
        uint32_t count   = 0;
        if (!ckpt_read(ifs, &magic, sizeof(magic), n_read) || magic != SLOT_CKPT_MAGIC) {
            return 0;
        }
        if (!ckpt_read(ifs, &version, sizeof(version), n_read) || version != SLOT_CKPT_VERSION ||
            !ckpt_read(ifs, &count,   sizeof(count),   n_read)) {
            SLT_WRN(slot, "invalid context checkpoint appendix in '%s' - ignored\n", filepath.c_str());
            return 0;
        }
        std::list<std::shared_ptr<common_prompt_checkpoint>> checkpoints;
        for (uint32_t i = 0; i < count; ++i) {
            auto ckpt = std::make_shared<common_prompt_checkpoint>();
            auto & cur = *ckpt;
            cur.id_task = -1; // not created by a task - marks a checkpoint restored from a slot file
            if (!ckpt_read(ifs, &cur.n_tokens, sizeof(cur.n_tokens), n_read) ||
                !ckpt_read(ifs, &cur.pos_min,  sizeof(cur.pos_min),  n_read) ||
                !ckpt_read(ifs, &cur.pos_max,  sizeof(cur.pos_max),  n_read) ||
                !ckpt_read_buf(ifs, cur.data_tgt,  n_avail, n_read) ||
                !ckpt_read_buf(ifs, cur.data_dft,  n_avail, n_read) ||
                !ckpt_read_buf(ifs, cur.data_spec, n_avail, n_read)) {
                SLT_WRN(slot, "truncated context checkpoint appendix in '%s' - ignored\n", filepath.c_str());
                return 0;
            }
            // a saved checkpoint always holds a target state - an empty blob would roll back without restoring anything
            if (cur.data_tgt.empty()) {
                SLT_WRN(slot, "invalid context checkpoint appendix in '%s' - ignored\n", filepath.c_str());
                return 0;
            }
            checkpoints.push_back(std::move(ckpt));
            if (checkpoints.size() > (size_t) params_base.n_ctx_checkpoints) {
                checkpoints.pop_front();
            }
        }
        // the slot file does not check the draft context - test-load one draft checkpoint, drop the draft data if it does not fit
        if (ctx_dft != nullptr && !checkpoints.empty() && !checkpoints.back()->data_dft.empty()) {
            const bool ok = checkpoints.back()->load_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            llama_memory_seq_rm(llama_get_memory(ctx_dft), slot.id, -1, -1);
            if (!ok) {
                SLT_WRN(slot, "draft context checkpoint data in '%s' does not match the draft context - dropped\n", filepath.c_str());
                for (auto & cur : checkpoints) {
                    cur->clear_dft();
                }
            }
        }
        slot.prompt.checkpoints.assign(checkpoints.begin(), checkpoints.end());
        SLT_INF(slot, "restored %zu context checkpoint(s) from '%s'\n", slot.prompt.checkpoints.size(), filepath.c_str());
        return n_read;
    }

    // the live draft state follows the checkpoints: the full draft context and the speculative state
    // the llama state payload holds the target context only
    static constexpr uint32_t SLOT_DRAFT_MAGIC   = 0x54464453; // "SDFT"
    static constexpr uint32_t SLOT_DRAFT_VERSION = 1;

    // returns false if the draft state could not be written completely
    bool save_slot_draft(const std::string & filepath, const server_slot & slot, size_t & n_written) const {
        n_written = 0;
        std::vector<uint8_t> data_dft;
        std::vector<uint8_t> data_spec;
        if (ctx_dft != nullptr) {
            data_dft.resize(llama_state_seq_get_size_ext(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_NONE));
            llama_state_seq_get_data_ext(ctx_dft, data_dft.data(), data_dft.size(), slot.id, LLAMA_STATE_SEQ_FLAGS_NONE);
        }
        common_speculative_get_state(spec.get(), slot.id, data_spec);
        if (data_dft.empty() && data_spec.empty()) {
            return true;
        }
        std::ofstream ofs(filepath, std::ios::binary | std::ios::app);
        if (!ofs) {
            SLT_WRN(slot, "failed to append the draft state to '%s'\n", filepath.c_str());
            return false;
        }
        const uint32_t magic   = SLOT_DRAFT_MAGIC;
        const uint32_t version = SLOT_DRAFT_VERSION;
        ckpt_write(ofs, &magic,   sizeof(magic),   n_written);
        ckpt_write(ofs, &version, sizeof(version), n_written);
        ckpt_write_buf(ofs, data_dft,  n_written);
        ckpt_write_buf(ofs, data_spec, n_written);
        ofs.flush();
        if (!ofs) {
            SLT_WRN(slot, "failed to append the draft state to '%s' - the draft state is incomplete\n", filepath.c_str());
            return false;
        }
        SLT_INF(slot, "appended the draft state (%.3f MiB) to '%s'\n", (float) n_written / 1024 / 1024, filepath.c_str());
        return true;
    }

    // returns the number of bytes consumed, 0 if there is no usable draft state
    // the draft starts empty unless the file holds a draft state that fits the draft context
    size_t load_slot_draft(const std::string & filepath, size_t offset, server_slot & slot) const {
        if (ctx_dft != nullptr) {
            llama_memory_seq_rm(llama_get_memory(ctx_dft), slot.id, -1, -1);
        }
        std::ifstream ifs(filepath, std::ios::binary | std::ios::ate);
        const size_t file_size = ifs ? (size_t) ifs.tellg() : 0;
        if (!ifs || file_size < offset || !ifs.seekg(offset)) {
            return 0;
        }
        const size_t n_avail = file_size - offset;
        size_t n_read = 0;
        uint32_t magic   = 0;
        uint32_t version = 0;
        if (!ckpt_read(ifs, &magic, sizeof(magic), n_read) || magic != SLOT_DRAFT_MAGIC) {
            return 0;
        }
        std::vector<uint8_t> data_dft;
        std::vector<uint8_t> data_spec;
        if (!ckpt_read(ifs, &version, sizeof(version), n_read) || version != SLOT_DRAFT_VERSION ||
            !ckpt_read_buf(ifs, data_dft,  n_avail, n_read) ||
            !ckpt_read_buf(ifs, data_spec, n_avail, n_read)) {
            SLT_WRN(slot, "invalid draft state in '%s' - ignored\n", filepath.c_str());
            return 0;
        }
        if (!data_dft.empty()) {
            if (ctx_dft == nullptr ||
                llama_state_seq_set_data_ext(ctx_dft, data_dft.data(), data_dft.size(), slot.id, LLAMA_STATE_SEQ_FLAGS_NONE) != data_dft.size()) {
                if (ctx_dft != nullptr) {
                    llama_memory_seq_rm(llama_get_memory(ctx_dft), slot.id, -1, -1);
                }
                SLT_WRN(slot, "draft state in '%s' does not match the draft context - dropped\n", filepath.c_str());
                return n_read;
            }
        }
        if (!data_spec.empty()) {
            common_speculative_set_state(spec.get(), slot.id, data_spec);
        }
        SLT_INF(slot, "restored the draft state (%.3f MiB) from '%s'\n", (float) n_read / 1024 / 1024, filepath.c_str());
        return n_read;
    }

    // returns false to decline the task, it is offered again after the decode is done
    bool process_single_task(server_task && task, bool is_yielding) {
        // while yielding, an encode / decode is running and only reading the server state is safe
        if (is_yielding && task.type != SERVER_TASK_TYPE_METRICS && task.type != SERVER_TASK_TYPE_SLOT_GET) {
            SRV_DBG("decoding, decline task, id_task = %d\n", task.id);
            return false;
        }

        switch (task.type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
            case SERVER_TASK_TYPE_TRANSCRIBE_RT:
            case SERVER_TASK_TYPE_REALTIME:
            case SERVER_TASK_TYPE_EMBEDDING:
            case SERVER_TASK_TYPE_RERANK:
            case SERVER_TASK_TYPE_TTS:
            case SERVER_TASK_TYPE_DECISION:
                {
                    if (task.need_sampling() && !decision.can_generate()) {
                        send_error(task, "This model is a decision model without text generation, use /v1/systemone", ERROR_TYPE_NOT_SUPPORTED);
                        break;
                    }

                    // special case: if input is provided via CLI, tokenize it first
                    // otherwise, no need to tokenize as it's already done inside the HTTP thread
                    if (task.cli) {
                        if (!tokenize_cli_input(task)) {
                            break;
                        }
                    }

                    const int id_task = task.id;

                    server_slot * slot = get_available_slot(task);

                    //
                    // slot scheduling logic
                    //

                    if (slot == nullptr) {
                        // if no slot is available, we defer this task for processing later
                        SRV_DBG("no slot is available, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (task.is_parent()) {
                        // try getting free slots for all child tasks
                        size_t n_child_tasks = task.child_tasks.size();
                        std::vector<server_slot *> child_slots = get_free_slots(n_child_tasks, slot->id);
                        if (child_slots.size() < n_child_tasks) {
                            SRV_DBG("not enough free slots for child tasks, n_free = %zu, n_children = %zu, defer task, id_task = %d\n", child_slots.size(), n_child_tasks, id_task);
                            queue_tasks.defer(std::move(task));
                            break;
                        }
                        if (!launch_slots_with_parent_task(*slot, child_slots, std::move(task))) {
                            SRV_ERR("failed to launch slot with parent task, id_task = %d\n", id_task);
                            break; // drop the task
                        }
                    } else if (!launch_slot_with_task(*slot, std::move(task))) {
                        SRV_ERR("failed to launch slot with task, id_task = %d\n", id_task);
                        break; // drop the task
                    }

                    if (params_base.cache_idle_slots) {
                        for (auto & slot : slots) {
                            if (!slot.is_processing()) {
                                SLT_TRC(slot, "%s", "saving idle slot to prompt cache\n");

                                if (slot.prompt_save(*prompt_cache)) {
                                    SLT_DBG(slot, "%s", "__TEST_TAG_CACHE_IDLE_SLOT__\n");
                                    prompt_cache->update();
                                }

                                if (params_base.kv_unified) {
                                    // [TAG_IDLE_SLOT_CLEAR]
                                    slot.prompt_clear();
                                }
                            }
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_CANCEL:
                {
                    // release slot linked with the task id
                    for (auto & slot : slots) {
                        if (slot.task && slot.task->id == task.id_target) {
                            slot.release();
                            break;
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_CONTROL:
                {
                    auto res = std::make_unique<server_task_result_control>();
                    res->id = task.id;

                    server_slot * slot = get_slot_by_cmpl_id(task.params.control_cmpl_id);
                    if (slot == nullptr) {
                        SRV_WRN("control %d on unknown completion id=%s, no live slot\n",
                                (int) task.params.control_action, task.params.control_cmpl_id.c_str());
                        res->success = false;
                        res->message = "no active completion for this id";
                        queue_results.send(std::move(res));
                        break;
                    }

                    if (task.params.control_action == SERVER_CONTROL_ACTION_REASONING_END) {
                        // the budget sampler only exists when reasoning control was armed
                        if (!slot->task->params.sampling.reasoning_control) {
                            res->success = false;
                            res->message = "reasoning control not enabled for this completion";
                            queue_results.send(std::move(res));
                            break;
                        }
                        // act on the live slot mid generation, never defer
                        common_sampler_reasoning_budget_force(slot->smpl.get());
                        res->success = true;
                    } else if (slot->rt) {
                        // a cancel acts now, everything else runs after what is already queued
                        if (task.params.control_action == SERVER_CONTROL_ACTION_RESPONSE_CANCEL) {
                            slot->rt->cancel();
                        } else {
                            slot->rt->push({
                                task.params.control_action,
                                std::move(task.params.control_text),
                                std::move(task.params.control_audio),
                                task.params.control_response,
                            });
                        }
                        slot->audio_wait = false;
                        res->success = true;
                    } else if (task.params.control_action == SERVER_CONTROL_ACTION_AUDIO_APPEND
                            || task.params.control_action == SERVER_CONTROL_ACTION_AUDIO_COMMIT
                            || task.params.control_action == SERVER_CONTROL_ACTION_AUDIO_END) {
                        if (!slot->audio_stream) {
                            res->success = false;
                            res->message = "not a realtime transcription slot";
                        } else {
                            auto * stream = slot->audio_stream.get();
                            if (task.params.control_action == SERVER_CONTROL_ACTION_AUDIO_APPEND) {
                                mtmd_audio_stream_push(stream, task.params.control_audio.data(), task.params.control_audio.size());
                            } else {
                                const int32_t ret = mtmd_audio_stream_flush(stream, task.params.control_action == SERVER_CONTROL_ACTION_AUDIO_END);
                                // a non-final commit completes its item once the rows of its audio are decoded
                                if (ret == 0 && task.params.control_action == SERVER_CONTROL_ACTION_AUDIO_COMMIT) {
                                    slot->audio_commit_rows.push_back(mtmd_audio_stream_get_n_rows_flushed(stream));
                                }
                            }
                            slot->audio_wait = false;
                            res->success = true;
                        }
                    } else {
                        res->success = false;
                        res->message = "unknown control action";
                    }

                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_NEXT_RESPONSE:
                {
                    // do nothing
                } break;
            case SERVER_TASK_TYPE_METRICS:
                {
                    int n_processing_slots = 0;

                    for (server_slot & slot : slots) {
                        if (slot.is_processing()) {
                            n_processing_slots++;
                        }
                    }
                    SRV_DBG("n_processing_slots = %d\n", n_processing_slots);

                    auto res = std::make_unique<server_task_result_metrics>();
                    res->id                  = task.id;
                    res->n_processing_slots  = n_processing_slots;
                    res->n_tasks_deferred    = queue_tasks.queue_tasks_deferred_size();
                    res->metrics             = metrics;

                    if (task.metrics_reset_bucket) {
                        metrics.reset_bucket();
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_GET:
                {
                    json slots_data = json::array();

                    int n_idle_slots = 0;

                    for (server_slot & slot : slots) {
                        if (!slot.is_processing()) {
                            n_idle_slots++;
                        }

                        slots_data.push_back(slot.to_json(slots_debug == 0));
                    }
                    SRV_DBG("n_idle_slots = %d\n", n_idle_slots);

                    auto res = std::make_unique<server_task_result_slots>();
                    res->id           = task.id;
                    res->slots_data   = std::move(slots_data);
                    res->n_idle_slots = n_idle_slots;

                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_SAVE:
                {
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    std::vector<char> packed;
                    try {
                        packed = slot->prompt.tokens.serialize();
                    } catch (const std::exception & err) {
                        send_error(task, err.what(), ERROR_TYPE_NOT_SUPPORTED);
                        break;
                    }

                    GGML_ASSERT(packed.size() % sizeof(llama_token) == 0);
                    const size_t nwrite = llama_state_seq_save_file(
                        ctx_tgt, filepath.c_str(), slot->id,
                        reinterpret_cast<const llama_token *>(packed.data()), packed.size() / sizeof(llama_token));
                    if (nwrite == 0) {
                        send_error(task, "Unable to save slot", ERROR_TYPE_SERVER);
                        break;
                    }

                    size_t nwrite_ckpt = 0;
                    if (!save_slot_checkpoints(filepath, *slot, nwrite_ckpt)) {
                        send_error(task, "Unable to save slot: incomplete context checkpoints", ERROR_TYPE_SERVER);
                        break;
                    }

                    size_t nwrite_dft = 0;
                    if (!save_slot_draft(filepath, *slot, nwrite_dft)) {
                        send_error(task, "Unable to save slot: incomplete draft state", ERROR_TYPE_SERVER);
                        break;
                    }

                    const int64_t t_end = ggml_time_us();
                    const double t_save_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = true;
                    res->n_tokens = slot->prompt.tokens.size();
                    res->n_bytes  = nwrite + nwrite_ckpt + nwrite_dft;
                    res->t_ms     = t_save_ms;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_RESTORE:
                {
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    size_t nread = 0;
                    try {
                        size_t n_packed = 0;
                        llama_tokens packed;
                        nread = llama_state_seq_load_file(ctx_tgt, filepath.c_str(), slot->id, nullptr, 0, &n_packed);
                        if (nread != 0) {
                            packed.resize(std::max<size_t>(1, n_packed));
                            nread = llama_state_seq_load_file(ctx_tgt, filepath.c_str(), slot->id, packed.data(), packed.size(), &n_packed);
                        }
                        if (nread == 0) {
                            throw std::runtime_error("No available space in KV cache or invalid slot save file");
                        }
                        packed.resize(n_packed);

                        server_tokens restored = server_tokens::deserialize(packed, mctx != nullptr);

                        if (restored.size() > (size_t) slot->n_ctx) {
                            throw std::runtime_error("Restored prompt does not fit in the slot context");
                        }

                        if (!restored.validate(ctx_tgt)) {
                            throw std::runtime_error("Invalid tokens in slot save file");
                        }

                        slot->prompt.clear();
                        slot->prompt.tokens = std::move(restored);
                    } catch (const std::exception & err) {
                        slot->prompt_clear();
                        send_error(task, std::string("Unable to restore slot: ") + err.what(), ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }

                    // nread is the end offset of the llama state payload within the file
                    const size_t nread_ckpt = load_slot_checkpoints(filepath, nread, *slot);
                    const size_t nread_dft  = load_slot_draft(filepath, nread + nread_ckpt, *slot);

                    const int64_t t_end = ggml_time_us();
                    const double t_restore_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = false;
                    res->n_tokens = slot->prompt.tokens.size();
                    res->n_bytes  = nread + nread_ckpt + nread_dft;
                    res->t_ms     = t_restore_ms;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_ERASE:
                {
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    // Erase token cache
                    const size_t n_erased = slot->prompt.tokens.size();

                    slot->prompt_clear();

                    auto res = std::make_unique<server_task_result_slot_erase>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->n_erased = n_erased;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_GET_LORA:
                {
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    auto & loras = params_base.lora_adapters;
                    auto res = std::make_unique<server_task_result_get_lora>();
                    res->id = task.id;
                    for (size_t i = 0; i < loras.size(); ++i) {
                        auto & lora = loras[i];
                        std::string alora_invocation_string = "";
                        const uint64_t n_alora_tokens = llama_adapter_get_alora_n_invocation_tokens(lora.ptr);
                        llama_tokens alora_invocation_tokens;
                        if (n_alora_tokens) {
                            const llama_token * alora_tokens = llama_adapter_get_alora_invocation_tokens(lora.ptr);
                            for (uint64_t j = 0; j < n_alora_tokens; ++j) {
                                alora_invocation_string += common_token_to_piece(vocab, alora_tokens[j]);
                                alora_invocation_tokens.push_back(alora_tokens[j]);
                            }
                        }
                        res->loras.push_back(server_task_result_get_lora::lora{
                            lora,
                            alora_invocation_string,
                            alora_invocation_tokens,
                        });
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SET_LORA:
                {
                    auto new_loras = construct_lora_list(task.set_lora);
                    // logging
                    for (size_t i = 0; i < new_loras.size(); ++i) {
                        SRV_TRC("set lora adapter idx=%zu scale=%f\n", i, new_loras[i].scale);
                    }
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    params_base.lora_adapters = new_loras;
                    auto res = std::make_unique<server_task_result_apply_lora>();
                    res->id = task.id;
                    queue_results.send(std::move(res));
                } break;
        }

        return true;
    }

    void iterate(std::vector<server_slot> & slots, std::function<void(server_slot &)> callback) {
        for (int pass = 0; pass < 2; ++pass) {
            for (auto & slot : slots) {
                const bool is_child = slot.is_processing() && slot.task->is_child();
                if (is_child != (pass == 1)) {
                    continue;
                }
                try {
                    callback(slot);
                } catch (const std::exception & e) {
                    SLT_ERR(slot, "got exception: %s\n", e.what());
                    send_error(slot, std::string("got exception: ") + e.what(), ERROR_TYPE_SERVER);
                    slot.release();
                }
            }
        }
    }

    void iterate(std::vector<server_slot *> & slots, std::function<void(server_slot &)> callback) {
        for (auto & slot : slots) {
            try {
                callback(*slot);
            } catch (const std::exception & e) {
                SLT_ERR(*slot, "got exception: %s\n", e.what());
                send_error(*slot, std::string("got exception: ") + e.what(), ERROR_TYPE_SERVER);
                slot->release();
            }
        }
    }

    void abort_all_slots(const std::string & reason) {
        for (auto & slot : slots) {
            if (slot.is_processing()) {
                send_error(slot, reason, ERROR_TYPE_SERVER);
                slot.release();
            }
        }
    }

    // @ngxson : for debugging only
    int64_t t_pre_decode  = 0;
    int64_t t_decode      = 0;
    int64_t t_post_decode = 0;
    int64_t t_sampl       = 0;
    int64_t n_pre_decode  = 0;
    int64_t n_decode      = 0;
    int64_t n_post_decode = 0;
    int64_t n_sampl       = 0;
// #define DEBUG_TIMINGS
#ifdef DEBUG_TIMINGS
    struct scoped_timer {
        int64_t & t;
        int64_t & n;
        int64_t t_start;
        scoped_timer(int64_t & t_, int64_t & n_) : t(t_), n(n_) {
            t_start = ggml_time_us();
        }
        ~scoped_timer() {
            t += ggml_time_us() - t_start;
            n++;
        }
    };
#else
    struct scoped_timer {
        scoped_timer(int64_t &, int64_t &) {}
        ~scoped_timer() {}
    };
#endif

    // fill slot.inp_embd with the next rows of the audio stream
    // returns false if the slot has nothing to decode now
    bool slot_audio_step(server_slot & slot) {
        if (!slot.inp_embd.empty()) {
            return true; // rows from a previous step, not decoded yet
        }

        const int32_t n = mtmd_audio_stream_step(slot.audio_stream.get(), slot.sampled);

        // complete the commits whose rows have their text sampled, before the new rows: their text goes to the next item
        const int64_t n_rows = mtmd_audio_stream_get_n_rows(slot.audio_stream.get());
        while (!slot.audio_commit_rows.empty() && slot.audio_commit_rows.front() <= n_rows) {
            slot.audio_commit_rows.pop_front();
            auto res = std::make_unique<server_task_result_cmpl_partial>();
            res->id                   = slot.task->id;
            res->index                = slot.task->index;
            res->n_decoded            = slot.stats.n_gen;
            res->n_prompt_tokens      = slot.task->n_tokens();
            res->res_type             = slot.task->params.res_type;
            res->oaicompat_model      = slot.task->params.oaicompat_model;
            res->oaicompat_cmpl_id    = slot.task->params.oaicompat_cmpl_id;
            res->audio_commit_drained = true;
            queue_results.send(std::move(res));
        }

        if (n > 0) {
            // new sequence (e.g. live ASR restart): drop the old one, the text goes on in a new line
            if (mtmd_audio_stream_is_seq_start(slot.audio_stream.get())) {
                slot.prompt_clear();
                if (slot.n_sent_text > 0) {
                    completion_token_output sep;
                    sep.tok          = LLAMA_TOKEN_NULL;
                    sep.text_to_send = "\n";
                    slot.generated.append(common_chat_input(sep.text_to_send));
                    slot.n_sent_text += sep.text_to_send.size();
                    send_partial_response(slot, sep, false);
                }
            }
            const float * rows = mtmd_audio_stream_get_output_embd(slot.audio_stream.get());
            slot.inp_embd.assign(rows, rows + (size_t) n * batch.n_embd);
            slot.audio_wait = false;
            return true;
        }
        if (n == 0) {
            slot.audio_wait = true;
            return false;
        }
        if (n == -2) {
            send_error(slot, "audio stream error", ERROR_TYPE_SERVER);
            slot.release();
            return false;
        }

        // end of stream
        slot.stop           = STOP_TYPE_EOS;
        slot.has_next_token = false;
        slot.print_timings();
        send_final_response(slot);
        slot.release();
        return false;
    }

    void update_slots() {
#ifdef DEBUG_TIMINGS
        static int64_t t_prev = 0;
        int64_t t_start = ggml_time_us();
        if (t_start - t_prev > 5 * 1000 * 1000) { // every 5 seconds
            t_prev = t_start;
            SRV_INF("n_pre_decode      = %" PRId64 "\n", n_pre_decode);
            SRV_INF("avg t_pre_decode  = %f ms\n", (double) t_pre_decode / n_pre_decode / 1000.0);
            SRV_INF("avg t_decode      = %f ms\n", (double) t_decode / n_decode / 1000.0);
            SRV_INF("avg t_post_decode = %f ms\n", (double) t_post_decode / n_post_decode / 1000.0);
            SRV_INF("avg t_sampl       = %f ms\n", (double) t_sampl / n_sampl / 1000.0);
        }
#endif

        // check if all slots are idle
        // a realtime slot waiting for audio has no work, the next audio control task wakes the loop
        {
            bool all_idle = true;

            for (auto & slot : slots) {
                if (slot.is_processing() && !slot.audio_wait) {
                    all_idle = false;
                    break;
                }
            }

            if (all_idle) {
                SRV_TRC("%s", "all slots are idle\n");

                metrics_flush_idle();

                return; // skip further processing

            } else {
                SRV_DBG("%s", "posting NEXT_RESPONSE\n");

                server_task task(SERVER_TASK_TYPE_NEXT_RESPONSE);
                task.id = queue_tasks.get_new_id();
                queue_tasks.post(std::move(task));
            }
        }

        try {
            scoped_timer t(t_pre_decode, n_pre_decode);
            pre_decode();
        } catch (const std::exception & e) {
            SRV_ERR("pre_decode() failed: %s\n", e.what());
            abort_all_slots("pre_decode() failed: " + std::string(e.what()));

            // the batch is half-built and not rendered, skip now to avoid UB
            return;
        }

        // note: TTS slots bypass the shared batch entirely
        try {
            process_tts_slots();
        } catch (const std::exception & e) {
            SRV_ERR("process_tts_slots() failed: %s\n", e.what());
            abort_all_slots("process_tts_slots() failed: " + std::string(e.what()));
        }

        // so do the realtime conversations
        try {
            process_realtime_slots();
        } catch (const std::exception & e) {
            SRV_ERR("process_realtime_slots() failed: %s\n", e.what());
            abort_all_slots("process_realtime_slots() failed: " + std::string(e.what()));
        }

        GGML_ASSERT(batch.slot_batched || batch.size() == 0);

        if (batch.slot_batched) {
            auto & slot_batched      = batch.slot_batched;
            auto & alora_scale       = batch.alora_scale;
            auto & alora_disabled_id = batch.alora_disabled_id;

            // TODO @ngxson : alora handling is too messy, need to refactor it to be more clear and maintainable
            // apply lora, only need to do it once per batch
            common_set_adapter_lora(ctx_tgt, slot_batched->lora);

            // if the lora is temporarily disabled for an alora, re-enable it
            // for next time
            if (alora_scale > 0.0f) {
                SRV_DBG("re-enabling alora with scale %f\n", alora_scale);
                slot_batched->lora[alora_disabled_id].scale = alora_scale;
            }

            llama_set_embeddings(ctx_tgt, slot_batched->need_embd());
        }

        int32_t off_next = 0;
        int32_t n_batch = llama_n_batch(ctx_tgt);
        for (int32_t off = 0; off < batch.size(); off = off_next) {
            const int32_t n_tokens = std::min(n_batch, batch.size() - off);
            try {
                scoped_timer t(t_decode, n_decode);
                // TODO @ngxson : maybe handle n_batch == 1 here instead of inside decode()

                batch.render(off, n_tokens);
                bool ok = decode(n_batch, off);
#ifdef DEBUG_TIMINGS
                llama_synchronize(ctx_tgt);
#endif

                if (ok) {
                    // move the head of the batch forward with the number of tokens we just processed
                    off_next = off + n_tokens;

                    // on successful decode, restore the original batch size
                    n_batch = llama_n_batch(ctx_tgt);
                } else {
                    // try again with the updated n_batch
                    continue;
                }
            } catch (const std::exception & e) {
                SRV_ERR("decode() failed: %s\n", e.what());
                abort_all_slots("decode() failed: " + std::string(e.what()));
                break; // stop any further processing
            }

            try {
                scoped_timer t(t_post_decode, n_post_decode);
                post_decode(n_tokens, off);
            } catch (const std::exception & e) {
                SRV_ERR("post_decode() failed: %s\n", e.what());
                abort_all_slots("post_decode() failed: " + std::string(e.what()));
                break; // stop any further processing
            }
        }
    }

    // realtime conversations (SERVER_TASK_TYPE_REALTIME) drive their own decode loop, see server_realtime_slot
    void process_realtime_slots() {
        iterate(slots, [&](server_slot & slot) {
            if (!slot.is_processing() || !slot.rt) {
                return;
            }
            if (!slot.rt->update()) {
                send_error(slot, slot.rt->error, slot.rt->error_type);
                slot.release();
                return;
            }
            slot.audio_wait = slot.rt->is_idle();
        });
    }

    void process_tts_slots() {
        iterate(slots, [&](server_slot & slot) {
            if (!slot.is_processing() || slot.task->type != SERVER_TASK_TYPE_TTS) {
                return;
            }

            llama_set_embeddings(ctx_tgt, true);

            if (slot.state == SLOT_STATE_PROCESSING_PROMPT) {
                const int32_t ret = slot.tts.ctx.step_prompt(llama_n_batch(ctx_tgt));
                if (ret < 0) {
                    send_error(slot, "TTS prompt processing failed", ERROR_TYPE_SERVER);
                    slot.release();
                } else if (ret == 0) {
                    // done prompt, do sample and save h_state_prompt
                    slot.tts.sampled = common_sampler_sample(slot.smpl.get(), ctx_tgt, -1);
                    common_sampler_accept(slot.smpl.get(), slot.tts.sampled, true);
                    const float * h_embd = llama_get_embeddings_ith(ctx_tgt, -1);
                    slot.tts.h_state_prompt.assign(h_embd, h_embd + llama_model_n_embd(model_tgt));
                    slot.tts.h_state = slot.tts.h_state_prompt.data();
                    slot.state = SLOT_STATE_GENERATING;
                }
                return;
            }

            const int32_t n_predict = slot.n_predict_max > 0 ? slot.n_predict_max : 512;
            if (slot.tts.stop || slot.tts.n_decoded >= n_predict) {
                int32_t      sample_rate = 0;
                const char * data        = nullptr;
                size_t       data_len    = 0;
                // generation truly ends here: force out any sub-window remainder still buffered
                if (slot.tts.ctx.flush() != 0 || slot.tts.ctx.get_output(&sample_rate, &data, &data_len) != 0) {
                    send_error(slot, "failed to finalize TTS output", ERROR_TYPE_SERVER);
                } else {
                    send_tts_result(slot, sample_rate, data, data_len, true);
                }
                slot.release();
                return;
            }

            const float * h_state_next = nullptr;
            bool          stop         = false;
            if (slot.tts.ctx.step_gen(slot.tts.sampled, slot.tts.h_state, &h_state_next, &stop) != 0) {
                send_error(slot, "TTS generation failed", ERROR_TYPE_SERVER);
                slot.release();
                return;
            }
            if (h_state_next == nullptr) {
                // end-of-speech without a new frame (e.g. pocket-tts eos head)
                slot.tts.stop = true;
                return;
            }
            slot.tts.h_state = h_state_next;
            slot.tts.n_decoded++;
            slot.tts.stop = stop;

            if (!stop) {
                slot.tts.sampled = common_sampler_sample(slot.smpl.get(), ctx_tgt, -1);
                common_sampler_accept(slot.smpl.get(), slot.tts.sampled, true);
            }

            if (slot.task->params.stream) {
                int32_t      sample_rate = 0;
                const char * data        = nullptr;
                size_t       data_len    = 0;
                if (slot.tts.ctx.get_output(&sample_rate, &data, &data_len) != 0) {
                    send_error(slot, "TTS streaming output failed", ERROR_TYPE_SERVER);
                    slot.release();
                } else if (data_len > 0) {
                    send_tts_result(slot, sample_rate, data, data_len, false);
                }
            }
        });
    }

    void pre_decode() {
        // apply context-shift if needed
        // TODO: simplify and improve
        iterate(slots, [&](server_slot & slot) {
            if (slot.task && (slot.task->type == SERVER_TASK_TYPE_TTS || slot.task->type == SERVER_TASK_TYPE_REALTIME)) {
                // TTS and realtime conversation slots drive their own decode loop, never enter the shared batch
                return;
            }
            if (slot.state == SLOT_STATE_GENERATING && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
                if (!params_base.ctx_shift) {
                    // this check is redundant (for good)
                    // we should never get here, because generation should already stopped in process_token()
                    send_error(slot, "context shift is disabled", ERROR_TYPE_SERVER);
                    slot.release();
                    return;
                }

                if (mctx) {
                    // we should never reach this because params_base.ctx_shift is automatically disabled if mmproj is loaded
                    // we don't support ctx_shift because an image chunk may contains multiple tokens
                    GGML_ABORT("not supported by multimodal");
                }

                if (slot.task->is_parent() || slot.task->is_child()) {
                    send_error(slot, "context shift cannot be used for shared prompt", ERROR_TYPE_SERVER);
                    slot.release();
                    return;
                }

                // Shift context
                int n_keep = slot.task->params.n_keep < 0 ? slot.task->n_tokens() : slot.task->params.n_keep;

                if (add_bos_token) {
                    n_keep += 1;
                }

                n_keep = std::min(slot.n_ctx - 4, n_keep);

                const int n_left    = slot.prompt.n_tokens() - n_keep;
                int       n_discard = slot.task->params.n_discard ? slot.task->params.n_discard : (n_left / 2);

                // ref: https://github.com/ggml-org/llama.cpp/pull/24786
                n_discard = std::clamp(n_discard, 0, std::max(0, n_left - 1));

                SLT_WRN(slot, "slot context shift, n_keep = %d, n_left = %d, n_discard = %d\n", n_keep, n_left, n_discard);

                // [QLH-SPLIT] Context shift is a *partial* removal: it drops a hole in the middle
                // and shifts the tail down by n_discard. A memory that cannot remove that range
                // (hybrid recurrent without room to roll back, a chain that no longer agrees)
                // would leave the hole in place while the prompt below is rewritten around it --
                // positions and KV disagree from then on. So the shift is all-or-nothing: stop
                // this request where it stands instead of serving the rest from a shifted prompt
                // with an unshifted KV.
                if (!slot.mem.seq_rm(slot.id, n_keep, n_keep + n_discard)) {
                    SLT_ERR(slot, "context shift could not remove [%d, %d) from the KV; stopping "
                                  "this request at %d generated tokens instead of continuing on a "
                                  "prompt that no longer matches the memory\n",
                            n_keep, n_keep + n_discard, (int) slot.stats.n_gen);
                    slot.stop           = STOP_TYPE_LIMIT;
                    slot.has_next_token = false;
                } else {
                slot.mem.seq_add(slot.id, n_keep + n_discard, slot.prompt.tokens.pos_next(), -n_discard);

                // add generated tokens to cache
                // ref: https://github.com/ggml-org/llama.cpp/pull/16818#discussion_r2473269481
                {
                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                    llama_tokens new_tokens = slot.prompt.tokens.get_tokens(); // copy
                    for (size_t i = n_keep + n_discard; i < new_tokens.size(); i++) {
                        new_tokens[i - n_discard] = new_tokens[i];
                    }

                    new_tokens.resize(slot.prompt.tokens.size() - n_discard);

                    slot.prompt.clear();
                    slot.prompt.tokens.insert(new_tokens);
                }

                slot.truncated = true;
                }
            }
        });

        // start populating the batch for this iteration
        batch.clear();

        // track if given slot can be batched with slots already in the batch
        auto & slot_batched = batch.slot_batched;

        std::vector<server_slot *> generating;
        std::vector<server_slot *> drafting;

        // determine which slots are generating and drafting
        iterate(slots, [&](server_slot & slot) {
            if (slot.state != SLOT_STATE_GENERATING || slot.task->type == SERVER_TASK_TYPE_TTS || slot.task->type == SERVER_TASK_TYPE_REALTIME) {
                return;
            }

            if (slot.audio_stream && !slot_audio_step(slot)) {
                return; // waiting for audio, or the stream ended
            }

            // check if we can batch this slot with the previous one
            if (!slot_batched) {
                slot_batched = &slot;
            } else if (!slot_batched->can_batch_with(slot)) {
                return;
            }

            generating.push_back(&slot);

            if (spec) {
                common_speculative_get_draft_params(spec.get(), slot.id).drafting = false;

                const bool use_ckpt_tgt = ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
                const bool use_ckpt_dft = ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

                const int n_draft_max = slot.skip_draft_once ? 0 : slot.get_n_draft_max();

                if (slot.skip_draft_once) {
                    // [QLH-SPLIT] v8: consume the flag set by a rolled-back verify frame. This
                    // round decodes the rolled-back position plainly; the next one may draft
                    // again (a new position, so a new chance of a grammar-legal argmax).
                    slot.skip_draft_once = false;
                }

                if (n_draft_max > 0) {
                    GGML_ASSERT(slot.can_speculate());

                    // stale candidates: a replay never reads them, a new draft refills them
                    slot.spec_draft_q.clear();

                    if (!slot.spec_draft.empty()) {
                        // we have a previous (partial) draft to reuse
                        if (use_ckpt_tgt) {
                            GGML_ASSERT(!slot.spec_ckpt.empty());
                        }
                    } else {
                        GGML_ASSERT(slot.spec_i_batch.empty());

                        // [QLH-SPLIT] a new frame: the blobs of the previous one are stale, and
                        // the whole-frame rollback reads "is spec_ckpt empty" as "was a target
                        // checkpoint captured for this frame" (see frame_needs_target_ckpt).
                        slot.spec_ckpt.clear();

                        slot.spec_ckpt.update_pos(
                                slot.prompt.n_tokens(),
                                llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id),
                                llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id));

                        if (use_ckpt_dft) {
                            // [QLH-SPLIT] This one cannot degrade like a stored checkpoint can:
                            // the draft context is rolled back through this blob later, and
                            // load_dft() reports success for an empty one while restoring
                            // nothing, which would leave the draft KV holding tokens the sample
                            // loop has already rejected. A failed capture is therefore a failed
                            // rollback, and this path has no way to abort just the request.
                            if (!slot.spec_ckpt.update_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)) {
                                common_log_flush(common_log_main());
                                GGML_ABORT("could not capture the draft checkpoint; the draft cannot be rewound\n");
                            }
                        }

                        slot.spec_prompt = slot.prompt.tokens.get_text_tokens();

                        const bool spec_reject = slot.use_spec_rejection();

                        common_speculative_get_draft_params(spec.get(), slot.id) = {
                            /* .drafting = */ true,
                            /* .n_max    = */ n_draft_max,
                            /* .pos0     = */ slot.prompt.tokens.pos_next(),
                            /* .id_last  = */ slot.sampled,
                            /* .prompt   = */ &slot.spec_prompt,
                            /* .result   = */ &slot.spec_draft,
                            /* .result_q = */ spec_reject ? &slot.spec_draft_q : nullptr,
                            /* .temp     = */ slot.task->params.sampling.temp,
                            /* .seed     = */ slot.task->params.sampling.seed,
                        };

                        drafting.push_back(&slot);
                    }
                }
            }
        });

        // generate the actual drafts (if any)
        if (!drafting.empty()) {
            queue_tasks.yield_to_queue([&]() {
                common_speculative_draft(spec.get());
            });
        }

        // make checkpoints if needed
        iterate(drafting, [&](server_slot & slot) {
            auto & draft = slot.spec_draft;
            auto & ckpt  = slot.spec_ckpt;

            slot.stats.n_draft_tokens += draft.size();

            // TODO: avoid restoring the draft context and re-evaluating the drafted tokens when not needed [TAG_SPEC_AVOID_DRAFT_REEVAL]
            const bool use_ckpt_dft = ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

            if (ctx_dft) {
                if (use_ckpt_dft) {
                    GGML_ASSERT(ckpt.load_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
                }

                // [QLH-SPLIT] keep-KV chain (default on; QLH_CHAIN_DRAFT_KEEP_KV=0 opts out): the
                // draft trims itself to its own pos0 at the start of every round, so the rows this
                // round just wrote must survive. Trimming them here would leave the route with
                // exactly the empty draft KV that the per-round clear left it with -- the route
                // would be enabled and do nothing.
                if (!common_speculative_keeps_draft_kv(spec.get())) {
                    if (!llama_memory_seq_rm(llama_get_memory(ctx_dft), slot.id, ckpt.pos_max + 1, -1)) {
                        // [QLH-SPLIT] KNOWN FATAL SITE (see the three-condition note on use_ckpt_tgt
                        // below). This one is the *draft* context, which is local even for a remote
                        // chain, so a network glitch cannot reach it; it is listed here only because
                        // it is the third abort in this block and the next reader should know it was
                        // looked at and deliberately left fatal.
                        GGML_ABORT("failed to remove sequence %d\n", slot.id);
                    }
                }
            }

            if (!draft.empty()) {
                // [QLH-SPLIT] the capture condition has to cover the *larger* of the two
                // rollbacks this frame may need, and one of them never fits the ubatch's own
                // snapshots -- see frame_needs_target_ckpt().
                const bool use_ckpt_tgt = frame_needs_target_ckpt(slot, draft.size());

                const bool use_ckpt_dft =
                   (ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && draft.size() > llama_n_rs_seq(ctx_dft));

                if (use_ckpt_tgt) {
                    //const int64_t t_start = ggml_time_us();

                    // [QLH-SPLIT] Same reasoning as the draft checkpoint above: the verify step
                    // below rewinds the target's KV through this blob, and an empty blob reports
                    // success while restoring nothing. Keep it fatal, but fatal with the reason.
                    //
                    // [QLH-SPLIT] KNOWN FATAL SITE, NOT FIXED YET -- this one can still take the
                    // server down on a remote chain, and the next person should not have to
                    // re-derive it. All three conditions are required:
                    //   1. speculative decoding is on (ctx_dft != nullptr; --spec-type ... / MTP);
                    //   2. ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL, which for a
                    //      remote chain head means it was started WITHOUT --split-rs-seq (the
                    //      startup probe then cannot do a partial rollback and degrades to FULL);
                    //   3. a remote stage stops answering.
                    // Then the capture runs over the wire (state_size + state_write), the stage's
                    // 120 s timeout makes it return a short/empty blob, update_tgt() reports false,
                    // and this line aborts the whole process -- the same "one network glitch kills
                    // every slot" failure that common_context_seq_rm and the checkpoint restore
                    // above no longer have. Fixing it is not a one-line change like those: without
                    // the blob the verified tokens cannot be rewound, so the round has to be
                    // abandoned and the batch/spec state unwound rather than continued. The safe
                    // carrier exists already (llama_context_split_failed(ctx_tgt) is true by then,
                    // because state_size/state_write refuse on a dead chain) -- what is missing is
                    // the unwind.
                    if (!ckpt.update_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)) {
                        common_log_flush(common_log_main());
                        GGML_ABORT("could not capture the speculative target checkpoint; the "
                                   "verified tokens cannot be rewound\n");
                    }

                    //const int64_t t_total = ggml_time_us() - t_start;
                    //printf("checkpoint total: %f ms\n", t_total / 1000.0);

                    SLT_DBG(slot, "created speculative checkpoint (pos_min = %d, pos_max = %d, n_tokens = %d, size = %.3f MiB, draft = %.3f MiB)\n",
                            ckpt.pos_min, ckpt.pos_max, slot.prompt.n_tokens(),
                            (float) ckpt.size() / 1024 / 1024,
                            (float) ckpt.data_dft.size() / 1024 / 1024);
                }

                if (use_ckpt_dft) {
                    // [QLH-SPLIT] Same as the two above: the draft is rewound through this blob.
                    // Also not reachable from a network glitch -- ctx_dft is the local draft
                    // context, not a chain stage -- see the three-condition note on use_ckpt_tgt.
                    if (!ckpt.update_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)) {
                        common_log_flush(common_log_main());
                        GGML_ABORT("could not capture the speculative draft checkpoint; the "
                                   "verified drafts cannot be rewound\n");
                    }
                }
            }
        });

        // update the batch with the sampled/drafted tokens
        iterate(generating, [&](server_slot & slot) {
            slot.handle_last_sampled_token(batch);
        });

        // process in chunks of params.n_batch
        int32_t n_batch  = llama_n_batch(ctx_tgt);
        int32_t n_ubatch = llama_n_ubatch(ctx_tgt);

        auto & alora_scale       = batch.alora_scale;
        auto & alora_disabled_id = batch.alora_disabled_id;

        // next, batch any pending prompts without exceeding n_batch
        if (params_base.cont_batching || batch.size() == 0) {
            bool add_ok = true; // false means the batch is full, skip remaining slots

            iterate(slots, [&](server_slot & slot) {
                if (!add_ok || batch.size() >= n_batch) {
                    return; // batch is full, skip remaining slots
                }

                if (!slot.is_processing() || slot.task->type == SERVER_TASK_TYPE_TTS || slot.task->type == SERVER_TASK_TYPE_REALTIME) {
                    return;
                }

                // check if we can batch this slot with the previous one
                if (slot_batched && !slot_batched->can_batch_with(slot)) {
                    return;
                }

                // check if this is a child slot
                if (slot.state == SLOT_STATE_WAIT_OTHER) {
                    SLT_DBG(slot, "%s", "waiting for parent slot to complete\n");
                    SLT_DBG(slot, "%s", "__TEST_TAG_WAIT_PARENT_SLOT__\n");
                    return;
                }

                // no prompt, the rows come from the audio stream
                if (slot.task->type == SERVER_TASK_TYPE_TRANSCRIBE_RT) {
                    return;
                }

                // this slot still has a prompt to be processed
                if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_STARTED) {
                    const auto & input_tokens = slot.task->tokens;

                    // used to determine the number of tokens added to the batch for the current slot
                    const auto n_tokens_prev = batch.size();

                    // TODO: maybe move branch to outside of this loop in the future
                    if (slot.state == SLOT_STATE_STARTED) {
                        slot.stats.update_prompt_start();

                        slot.state = SLOT_STATE_PROCESSING_PROMPT;

                        // [QLH-SPLIT] The request boundary is *here*, not at DONE_PROMPT.
                        //
                        // The prompt's process() batches run before DONE_PROMPT is reached, and on
                        // a local chain those batches are exactly when the tail's h rows for the
                        // prompt exist -- the only moment the draft catch-up can replay them. With
                        // begin() at DONE_PROMPT the draft KV was cleared *after* the prompt had
                        // already gone by, so a prefill-only catch-up had no window at all (it
                        // wrote rows and then watched begin() delete them). Moving it a few lines
                        // earlier changes nothing else: begin() clears this sequence's draft KV
                        // (what "a new request" means for the keep-KV route) and resets the draft
                        // sampler, both of which are as true before the prefill as after it.
                        if (spec && slot.can_speculate()) {
                            common_speculative_begin(spec.get(), slot.id,
                                    slot.task->tokens.get_text_tokens());
                        }

                        SLT_TRC(slot, "new prompt, n_ctx_slot = %d, n_keep = %d, task.n_tokens = %d\n",
                                slot.n_ctx, slot.task->params.n_keep, slot.task->n_tokens());

                        // print prompt tokens (for debugging)
                        /*if (1) {
                            // first 16 tokens (avoid flooding logs)
                            for (int i = 0; i < std::min<int>(16, input_tokens.size()); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx_tgt, input_tokens[i]).c_str());
                            }
                        } else {
                            // all
                            for (int i = 0; i < (int) input_tokens.size(); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx_tgt, input_tokens[i]).c_str());
                            }
                        }*/

                        // keep track how many tokens we can reuse from the previous state
                        int n_past = 0;

                        // empty prompt passed -> release the slot and send empty response
                        if (input_tokens.empty()) {
                            SLT_WRN(slot, "%s", "empty prompt - releasing slot\n");

                            slot.print_timings();
                            send_final_response(slot);
                            slot.release();

                            return;
                        }

                        // TODO: support memory-less logits computation
                        // a decision reads its logits from the one batch of its prompt
                        if (slot.task->need_logits() && !llama_get_memory(ctx_tgt) && slot.task->type != SERVER_TASK_TYPE_DECISION) {
                            send_error(slot, "the current context does not support logits computation. skipping", ERROR_TYPE_SERVER);
                            slot.release();
                            return;
                        }

                        if (!slot.can_split()) {
                            if (slot.task->n_tokens() > n_ubatch) {
                                send_error(slot,
                                           string_format(
                                               "input (%d tokens) is too large to process. increase the physical batch "
                                               "size (current batch size: %d)",
                                               slot.task->n_tokens(), n_ubatch),
                                           ERROR_TYPE_SERVER);
                                slot.release();
                                return;
                            }

                            if (slot.task->n_tokens() > slot.n_ctx) {
                                send_error(
                                    slot,
                                    string_format(
                                        "input (%d tokens) is larger than the max context size (%d tokens). skipping",
                                        slot.task->n_tokens(), slot.n_ctx),
                                    ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                return;
                            }
                        } else {
                            if (slot.task->n_tokens() >= slot.n_ctx) {
                                send_error(slot,
                                           string_format("request (%d tokens) exceeds the available context size (%d "
                                                         "tokens), try increasing it",
                                                         slot.task->n_tokens(), slot.n_ctx),
                                           ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                return;
                            }

                            // the outputs of a decision are read from one batch
                            const int32_t n_decision_first = slot.task->type == SERVER_TASK_TYPE_DECISION ? slot.task->decision.pos_first() : -1;
                            if (n_decision_first >= 0 && slot.task->n_tokens() - n_decision_first > n_batch) {
                                send_error(slot,
                                           string_format("the question and its options (%d tokens) are too large to process. "
                                                         "increase the batch size (current batch size: %d)",
                                                         slot.task->n_tokens() - n_decision_first, n_batch),
                                           ERROR_TYPE_INVALID_REQUEST);
                                slot.release();
                                return;
                            }

                            if (slot.task->params.cache_prompt && !slot.task->is_stateless()) {
                                // reuse any previously computed tokens that are common with the new prompt
                                n_past = slot.prompt.tokens.get_common_prefix(input_tokens);

                                // the children start from the shared prefix, do not go past it
                                if (slot.task->n_tokens_shared > 0) {
                                    n_past = std::min(n_past, slot.task->n_tokens_shared);
                                }

                                // the outputs of a decision are read from the batch, so its first position must be evaluated
                                if (n_decision_first >= 0) {
                                    n_past = std::min(n_past, n_decision_first);
                                }

                                // if there is an alora invoked, don't cache after the invocation start
                                if (slot.alora_invocation_start > 0) {
                                    SLT_DBG(slot, "only caching to alora invocation start (n_past = %d, alora_invocation_start = %d)\n", n_past, slot.alora_invocation_start);
                                    n_past = std::min(n_past, slot.alora_invocation_start - 1);
                                }

                                const auto n_cache_reuse = slot.task->params.n_cache_reuse;

                                const bool can_cache_reuse =
                                    llama_memory_can_shift(llama_get_memory(ctx_tgt)) &&
                                    !slot.prompt.tokens.has_mtmd;

                                if (!can_cache_reuse && n_cache_reuse > 0) {
                                    SLT_WRN(slot, "cache reuse is not supported - ignoring n_cache_reuse = %d\n", n_cache_reuse);
                                }

                                // reuse chunks from the cached prompt by shifting their KV cache in the new position
                                if (can_cache_reuse && n_cache_reuse > 0) {
                                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                                    size_t head_c = n_past; // cache
                                    size_t head_p = n_past; // current prompt

                                    if (mctx) {
                                        // we should never reach this
                                        GGML_ABORT("not supported by multimodal");
                                    }

                                    SLT_DBG(slot, "trying to reuse chunks with size > %d, n_past = %d\n", n_cache_reuse, n_past);

                                    while (head_c < slot.prompt.tokens.size() &&
                                           head_p < input_tokens.size()) {

                                        size_t n_match = 0;
                                        while (head_c + n_match < slot.prompt.tokens.size() &&
                                               head_p + n_match < input_tokens.size()       &&
                                               slot.prompt.tokens[head_c + n_match] == input_tokens[head_p + n_match]) {
                                            n_match++;
                                        }

                                        if (n_match >= (size_t) n_cache_reuse) {
                                            SLT_TRC(slot, "reusing chunk with size %zu, shifting KV cache [%zu, %zu) -> [%zu, %zu)\n", n_match, head_c, head_c + n_match, head_p, head_p + n_match);
                                            //for (size_t i = head_p; i < head_p + n_match; i++) {
                                            //    SLT_DBG(slot, "cache token %3zu: %6d '%s'\n", i, prompt_tokens[i], common_token_to_piece(ctx_tgt, prompt_tokens[i]).c_str());
                                            //}

                                            const int64_t kv_shift = (int64_t) head_p - (int64_t) head_c;

                                            // [QLH-SPLIT] The shift is a partial removal and can
                                            // be refused (hybrid recurrent, or a chain that no
                                            // longer agrees). If it is, the KV no longer matches
                                            // the prompt about to be rewritten: fail the request
                                            // rather than leave the two describing different
                                            // histories.
                                            if (!slot.mem.seq_rm(slot.id, head_p, head_c)) {
                                                SLT_ERR(slot, "failed to shift the cached KV at "
                                                              "[%zu, %zu); failing the request\n",
                                                        head_p, head_c);
                                                send_error(slot, "failed to shift the cached KV "
                                                                 "(seq_rm); the request cannot continue "
                                                                 "on a consistent state");
                                                slot.prompt_clear();
                                                slot.release();
                                                return;
                                            }
                                            slot.mem.seq_add(slot.id, head_c, head_c + n_match, kv_shift);

                                            for (size_t i = 0; i < n_match; i++) {
                                                slot.prompt.tokens.set_token(head_p + i, slot.prompt.tokens[head_c + i]);
                                                n_past++;
                                            }

                                            head_c += n_match;
                                            head_p += n_match;
                                        } else {
                                            head_c += 1;
                                        }
                                    }

                                    SLT_DBG(slot, "after context reuse, new n_past = %d\n", n_past);
                                }
                            } else {
                                // if we don't cache the prompt, we have to remove all previous tokens
                                n_past = 0;
                            }

                            llama_pos pos_next = slot.prompt.tokens.pos_next(n_past);

                            // number of cached tokens shared with the new prompt, before any checkpoint is restored
                            const int n_prefix = n_past;

                            // ref: https://github.com/ggml-org/llama.cpp/pull/24110
                            const bool has_new_tokens = (n_past < slot.task->n_tokens());

                            // the first evaluated token attends to the positions > pos_eval - n_swa, so a SWA state
                            // starting at pos_eval - n_swa + 1 is enough, while a state without SWA must end before pos_eval
                            const llama_pos pos_eval = pos_next - (has_new_tokens ? 0 : 1);

                            // a state or a checkpoint is useful when its pos_min is below this threshold
                            const auto pos_min_thold = std::max(0, n_swa > 0 ? pos_eval - n_swa + 2 : pos_eval);

                            if (n_past > 0 && n_past <= slot.prompt.n_tokens()) {
                                const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
                                if (pos_min == -1) {
                                    SLT_ERR(slot, "n_past = %d, slot.prompt.tokens.size() = %d, seq_id = %d, pos_min = %d\n", n_past, (int) slot.prompt.tokens.size(), slot.id, pos_min);
                                    GGML_ABORT("pos_min == -1, but n_past > 0 - should not happen: https://github.com/ggml-org/llama.cpp/pull/13833#discussion_r2116181237");
                                }

                                // when the prompt prefix does not match, print the tokens around the mismatch
                                // this is useful for debugging prompt caching
                                if (slots_debug) {
                                    const int np0 = std::max<int>(n_past - slots_n_diff, 0);
                                    const int np1 = std::min<int>(n_past + slots_n_diff + 2, std::min(slot.prompt.tokens.size(), slot.task->tokens.size()));

                                    std::stringstream ss0;
                                    std::stringstream ss1;

                                    std::stringstream st0;
                                    std::stringstream st1;

                                    ss0 << "old: ... ";
                                    ss1 << "new: ... ";

                                    for (int i = np0; i < np1; i++) {
                                        if (i == n_past) {
                                            ss0 << " | ";
                                            ss1 << " | ";
                                        }

                                        {
                                            const auto token = slot.prompt.tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx_tgt, token) : "[mtmd]";
                                            ss0 << piece;
                                            st0 << std::setw(8) << token;
                                        }

                                        {
                                            const auto token = slot.task->tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx_tgt, token) : "[mtmd]";
                                            ss1 << piece;
                                            st1 << std::setw(8) << token;
                                        }
                                    }

                                    SLT_WRN(slot, "%s\n", ss0.str().c_str());
                                    SLT_WRN(slot, "%s\n", ss1.str().c_str());

                                    SLT_WRN(slot, "%s\n", st0.str().c_str());
                                    SLT_WRN(slot, "%s\n", st1.str().c_str());
                                }

                                if (pos_min >= pos_min_thold) {
                                    // search for a context checkpoint
                                    const auto it = std::find_if(
                                        slot.prompt.checkpoints.rbegin(),
                                        slot.prompt.checkpoints.rend(),
                                        [&](const auto & ckpt) {
                                            const auto & cur = *ckpt;
                                            // guarantee that a checkpoint will result in at least one token being processed [TAG_PROMPT_LOGITS]
                                            SLT_TRC(slot, "checking checkpoint with [%d, %d] against %d...\n", cur.pos_min, cur.pos_max, pos_min_thold);
                                            // workaround for [TAG_CHECKPOINTS_FIX_POS_MIN]
                                            if (cur.pos_max > pos_next) {
                                                return false;
                                            }
                                            return cur.pos_min < pos_min_thold || cur.pos_min == 0;
                                        }
                                    );

                                    bool do_reset = it == slot.prompt.checkpoints.rend();

                                    if (!do_reset) {
                                        const auto & cur = **it;

                                        // restore the context checkpoint
                                        const bool ckpt_tgt_ok =
                                            cur.load_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                                        const bool ckpt_dft_ok = ckpt_tgt_ok && cur.load_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                                        if (!ckpt_tgt_ok || !ckpt_dft_ok) {
                                            // [QLH-SPLIT] A remote chain whose checkpoint cannot
                                            // be restored has to fail the request, not the
                                            // server. Measured on a chain with a suspended stage:
                                            // this exact abort is what took the head process down
                                            // (the field report blamed seq_rm because its first
                                            // failing request had no checkpoint to restore yet; a
                                            // warm slot hits this one instead).
                                            //
                                            // Why continuing is sound: a failed restore of the
                                            // *target* chain is recorded by the memory layer as
                                            // "chain dead" (llama_memory_split::state_read), and
                                            // decode_split then refuses every decode. The
                                            // fallback below re-processes the prompt from 0, and
                                            // its seq_rm/decode both fail with the reason -- so the
                                            // request ends as a 500 with a diagnosis, never with a
                                            // token computed from a half-restored chain.
                                            //
                                            // A local chain, or a failed *draft* restore, keeps the
                                            // abort: there is no network to blame and nothing to
                                            // rebuild in place.
                                            if (!ckpt_tgt_ok && llama_context_split_failed(ctx_tgt)) {
                                                common_log_flush(common_log_main());
                                                SLT_ERR(slot, "could not restore the context checkpoint: %s\n",
                                                        llama_context_split_failure(ctx_tgt));
                                                do_reset = true;
                                            } else if (!ckpt_tgt_ok && llama_context_split_state_stale(ctx_tgt)) {
                                                // [QLH-SPLIT] v6: the remote stage no longer holds
                                                // the snapshot this checkpoint names -- its bounded
                                                // snapshot cache evicted it, or the stage is a
                                                // different process than the one that minted the
                                                // handle. That is not a chain failure: the stage
                                                // refused before applying anything and the head is
                                                // restored last, so every stage is where it was.
                                                //
                                                // The recovery is the one a server without
                                                // checkpoints already takes: drop the checkpoints
                                                // and re-process the prompt. The erase loop below
                                                // does the dropping, because do_reset sets pos_next
                                                // to 0 and every checkpoint then covers tokens the
                                                // new prompt no longer has.
                                                SLT_WRN(slot, "%s", "the remote stage no longer holds this context checkpoint "
                                                        "(its snapshot cache evicted it); re-processing the prompt\n");
                                                do_reset = true;
                                            } else if (cur.id_task != -1) {
                                                // [QLH-SPLIT] Same reason as the flush in
                                                // common_context_seq_rm: the memory layer has
                                                // already logged which stage refused the
                                                // restore, and that line is still queued in the
                                                // log worker when abort() runs.
                                                common_log_flush(common_log_main());
                                                GGML_ABORT("failed to restore context checkpoint\n");
                                            } else {
                                                // restored from a slot file, not guaranteed to load - fall back to full prompt re-processing
                                                SLT_WRN(slot, "%s", "failed to load context checkpoint restored from a slot file\n");
                                                do_reset = true;
                                            }
                                        }
                                    }

                                    if (!do_reset) {
                                        const auto & cur = **it;

                                        // restore the draft's speculative state
                                        common_speculative_set_state(spec.get(), slot.id, cur.data_spec);

                                        pos_next = std::min(pos_next, std::max(cur.pos_min + 1, cur.pos_max));
                                        n_past   = std::min(slot.prompt.tokens.size_up_to_pos(pos_next), (size_t) cur.n_tokens);
                                        SLT_TRC(slot, "restored context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", n_past = %d, size = %.3f MiB)\n", cur.pos_min, cur.pos_max, cur.n_tokens, n_past, (float) cur.size() / 1024 / 1024);
                                    }

                                    if (do_reset) {
                                        SLT_TRC(slot, "forcing full prompt re-processing due to lack of cache data (likely due to SWA or hybrid/recurrent memory, see %s)\n",
                                                "https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055");
                                        pos_next = 0;
                                        n_past = 0;
                                    }
                                }
                            }

                            {
                                // erase any checkpoints with pos_max > pos_next, or that cover tokens not shared with the new prompt
                                for (auto it = slot.prompt.checkpoints.begin(); it != slot.prompt.checkpoints.end();) {
                                    const auto & cur = **it;
                                    if (cur.pos_max > pos_next || cur.n_tokens > n_prefix) {
                                        SLT_TRC(slot, "erased invalidated context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", n_swa = %d, pos_next = %d, size = %.3f MiB)\n", cur.pos_min, cur.pos_max, cur.n_tokens, n_swa, pos_next, (float) cur.size() / 1024 / 1024);
                                        it = slot.prompt.checkpoints.erase(it);
                                    } else {
                                        ++it;
                                    }
                                }
                            }
                        }

                        // [TAG_PROMPT_LOGITS]
                        if (n_past == slot.task->n_tokens() && n_past > 0) {
                            SLT_WRN(slot, "need to evaluate at least 1 token for each active slot (n_past = %d, task.n_tokens() = %d)\n", n_past, slot.task->n_tokens());
                            n_past--;
                            SLT_WRN(slot, "n_past was set to %d\n", n_past);
                        }

                        slot.stats.n_prompt_cached    = n_past;
                        slot.stats.n_prompt_processed = 0;

                        metrics.add_prompt_cached(n_past);

                        slot.prompt.tokens.keep_first(n_past);

                        // this is to signal the client that the request has started processing
                        if (slot.task->params.stream) {
                            if (slot.task->params.return_progress) {
                                // send initial 0% progress update if needed
                                send_partial_response(slot, {}, true);
                            } else {
                                // otherwise, for streaming without progress, signal HTTP to send the headers (i.e. 200 status)
                                send_partial_response(slot, {}, false, true);
                            }
                        }
                    } // end of SLOT_STATE_STARTED

                    if (!slot.can_split()) {
                        // cannot fit the prompt in the current batch - will try next iter
                        if (batch.size() + slot.task->n_tokens() > n_batch) {
                            return;
                        }
                    }

                    // note: the prompt timing is advanced in post_decode(), so it does not cover
                    //       the tokens added to the batch below
                    slot.print_timings_pp();

                    // truncate any tokens that are beyond n_past for this slot
                    const llama_pos p0 = slot.prompt.tokens.pos_next();

                    SLT_TRC(slot, "cached n_tokens = %d, memory_seq_rm [%d, end)\n", slot.prompt.n_tokens(), p0);

                    // [QLH-SPLIT] A cached prefix is only reusable if the KV can drop the tail the
                    // cached prompt no longer covers. When it cannot (a hybrid memory that refuses
                    // a partial removal, or a chain that no longer agrees), the alternative is not
                    // "carry on with a KV the prompt does not describe" and not "fail the request"
                    // either: it is the degradation PR #28007 chose -- throw the cache away and
                    // re-process the prompt from position 0. It keeps the request (200, correct
                    // output, only the prompt eval is paid again) and leaves nothing inconsistent
                    // behind.
                    //
                    // Rebuilding prompt.tokens is what makes the fall-through real: clearing the
                    // sequence drops the KV *and* the token list, and the next iteration of the
                    // prompt loop starts from whatever prompt.tokens holds. Without the refill it
                    // would start from an empty prompt, i.e. generate from nothing.
                    // [QLH-SPLIT] NOTE: this branch has never been exercised end-to-end. It is
                    // structurally identical to the speculative-accept trim below (which has
                    // been), but no test constructs a cache-reuse shape that reaches a partial
                    // seq_rm at p0 <= pos_max with p0 > 0. Treat it as code-review-verified, not
                    // run: if you change the degradation contract, the rebuilt-prompt line is the
                    // part that has no test behind it.
                    if (!slot.mem.seq_rm(slot.id, p0, -1)) {
                        if (llama_context_split_failed(slot.ctx_tgt)) {
                            // [QLH-SPLIT] A dead split chain answers every seq_rm with false,
                            // including the full clear inside prompt_clear(). Failing the request
                            // *here* would be wrong: self-healing needs one decode that starts at
                            // position 0 (llama_memory_split::try_rebuild reconnects and clears
                            // every stage on exactly that), and ending the request before the
                            // decode skips it -- the chain then stays dead for every following
                            // request. So: report it loudly (no longer silent) and let the normal
                            // path fail the request and release the slot; the next request starts
                            // at position 0 and rebuilds the chain.
                            SLT_ERR(slot, "cannot trim the KV to the cached prefix at pos %d: the "
                                          "split chain is no longer trustworthy (%s); this request "
                                          "will fail, the next one reconnects the chain\n",
                                    (int) p0, llama_context_split_failure(slot.ctx_tgt));
                        } else {
                            // [QLH-SPLIT] NOTE: this branch has never been exercised end-to-end.
                            // It is structurally identical to the speculative-accept trim below
                            // (which has been), but no test constructs a cache-reuse shape that
                            // reaches a partial seq_rm at p0 <= pos_max with p0 > 0. Treat it as
                            // code-review-verified, not run: if you change the degradation
                            // contract, the rebuilt-prompt line is the part with no test behind it.
                            SLT_WRN(slot, "memory could not roll back to pos %d - clearing the "
                                          "sequence and re-processing the full prompt\n", (int) p0);
                            slot.prompt_clear();
                            slot.prompt.tokens = server_tokens(slot.task->tokens.get_tokens(), slot.task->tokens.has_mtmd);
                            return;   // this iteration does nothing; the next one processes from pos 0
                        }
                    }

                    // shared prompt prefix: once it is processed, the children continue from it with their own prompt
                    bool wait_shared = false;
                    if (slot.task->n_tokens_shared > 0) {
                        const bool is_shared_done = slot.prompt.n_tokens() == slot.task->n_tokens_shared;
                        for (auto & other : slots) {
                            if (other.state != SLOT_STATE_WAIT_OTHER || other.task->id_parent != slot.task->id) {
                                continue;
                            }
                            if (is_shared_done) {
                                SLT_TRC(slot, " - copying shared prompt (%d tokens) to child %d\n", slot.prompt.n_tokens(), other.id);
                                slot.copy_prompt_to(other);
                                other.state = SLOT_STATE_STARTED;
                            } else {
                                wait_shared = true;
                            }
                        }
                    }

                    // If using an alora, there may be uncached tokens that come
                    // before the invocation sequence. When this happens, the
                    // tokens before the invocation sequence need to be
                    // processed without the adapter in a separate batch, then
                    // the adapter needs to be enabled for the remaining tokens.
                    if (lora_all_alora(slot.lora) && slot.alora_invocation_start - 1 > slot.prompt.n_tokens()) {
                        SLT_DBG(slot, "processing pre-alora tokens without the adapter (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                        const auto & enabled_loras = lora_get_enabled_ids(slot.lora);
                        GGML_ASSERT(enabled_loras.size() == 1);
                        alora_scale = slot.lora[enabled_loras[0]].scale;
                        slot.lora[enabled_loras[0]].scale = 0.0f;
                        alora_disabled_id = enabled_loras[0];
                    }

                    bool do_checkpoint = params_base.n_ctx_checkpoints > 0;

                    // make checkpoints only for completion tasks
                    do_checkpoint = do_checkpoint && slot.task->type == SERVER_TASK_TYPE_COMPLETION;

                    // make a checkpoint of the parts of the memory that cannot be rolled back.
                    // checkpoints are created only if:
                    // - the model does not support partial sequence removal
                    // - the model uses SWA (and we are not using `swa_full`)
                    // - the model supports partial sequence removal but only up to a fixed bound
                    do_checkpoint = do_checkpoint && (
                            ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                            ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS ||
                            n_swa > 0);

                    // TODO: do the same for all models, then remove process_mtmd_chunk()
                    if (use_mixed_batch() && !slot.can_split() && input_tokens.has_mtmd) {
                        if (!add_prompt_mixed(slot)) {
                            return;
                        }
                    }

                    bool has_mtmd = false;

                    // check if we should process the mtmd chunk
                    while (true) {
                        auto cur_token_idx = slot.prompt.n_tokens();
                        if (
                            cur_token_idx >= slot.task->n_tokens() ||
                            input_tokens[cur_token_idx] != LLAMA_TOKEN_NULL // encountered a text token
                        ) {
                            break;
                        }

                        // process the mtmd chunk
                        // note: it submits its own decode, potentially be async
                        //       so the timing is queued and flushed on the next sync
                        metrics_pre_decode();

                        // encode on the worker thread, so we can still handle metrics tasks
                        size_t n_tokens_out = 0;
                        int32_t res = 0;
                        queue_tasks.yield_to_queue([&]() {
                            res = process_mtmd_chunk(slot, slot.mbatch, cur_token_idx, n_tokens_out);
                        });

                        if (res != 0) {
                            SLT_ERR(slot, "failed to process mtmd chunk, res = %d\n", res);
                            send_error(slot, "failed to process mtmd chunk", ERROR_TYPE_SERVER);
                            slot.release();
                            return; // the slot is done, skip it entirely
                        }

                        metrics_queue_prompt(n_tokens_out);
                        slot.stats.n_prompt_processed += n_tokens_out;
                        slot.stats.update_prompt_last();

                        // add the mtmd chunk to cache
                        {
                            const auto & chunk = input_tokens.find_chunk(cur_token_idx);
                            // the chunk is already in the KV cache at this point, so we don't need to keep its data around
                            slot.prompt.tokens.push_back_placeholder(chunk.get());
                        }

                        has_mtmd = true;
                    }

                    const auto & spans = slot.task->params.message_spans;
                    const auto last_user_pos = spans.last_user_message_pos();

                    const int32_t n_decision_first = slot.task->type == SERVER_TASK_TYPE_DECISION ? slot.task->decision.pos_first() : -1;

                    // add prompt tokens for processing in the current batch
                    while (slot.prompt.n_tokens() < slot.task->n_tokens() && batch.size() < n_batch) {
                        // get next token to process
                        llama_token cur_tok = input_tokens[slot.prompt.n_tokens()];
                        if (cur_tok == LLAMA_TOKEN_NULL) {
                            break; // end of text chunk
                        }

                        // stop at the end of the shared prefix, the children are started from this state
                        if (wait_shared && slot.prompt.n_tokens() == slot.task->n_tokens_shared) {
                            break;
                        }

                        // the outputs of a decision are read from one batch, do not split them
                        if (slot.prompt.n_tokens() == n_decision_first && batch.size() + slot.task->n_tokens() - n_decision_first > n_batch) {
                            break;
                        }

                        // if this is an alora request with pre-invocation
                        // tokens that are not cached, we need to stop filling
                        // this batch at those pre-invocation tokens.
                        if (alora_scale > 0 && slot.prompt.n_tokens() == slot.alora_invocation_start - 1) {
                            SLT_DBG(slot, "stop prompt batch filling at (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                            break;
                        }

                        // embedding requires all tokens in the batch to be output;
                        // MTP also wants logits at every prompt position so the
                        // streaming hook can mirror t_h_nextn into ctx_dft.
                        add_ok &= batch.add(slot.id,
                            cur_tok,
                            /* pos       = */ slot.prompt.tokens.pos_next(),
                            /* output    = */ slot.need_embd(),
                            /* is_prompt = */ true);
                        if (!slot.task->decision.order.empty()) {
                            batch.set_decision_order(batch.size() - 1, slot.task->decision.order[slot.prompt.n_tokens()]);
                        }
                        slot.prompt.tokens.push_back(cur_tok);

                        // break at the last user message, or at user messages at least min step past the last checkpoint
                        if (do_checkpoint && spans.is_user_start(slot.prompt.n_tokens())) {
                            const auto pos = slot.prompt.n_tokens();
                            const auto & checkpoints = slot.prompt.checkpoints;

                            if (pos == last_user_pos || checkpoints.empty() || pos > checkpoints.back()->n_tokens + params_base.checkpoint_min_step) {
                                break;
                            }
                        }

                        // process the last few tokens of the prompt separately in order to allow for a checkpoint to be created.
                        // create checkpoints that many tokens before the end of the prompt:
                        //  - 4 + n_ubatch
                        //  - 4
                        // ref: https://github.com/ggml-org/llama.cpp/pull/20288
                        if (do_checkpoint) {
                            static const int checkpoint_offsets[] = {4 + n_ubatch, 4};

                            bool should_break = false;
                            for (int offset : checkpoint_offsets) {
                                const int n_last = std::min(n_batch, offset);
                                if (slot.task->n_tokens() == slot.prompt.n_tokens() + n_last) {
                                    should_break = true;
                                    break;
                                }
                            }
                            if (should_break) {
                                break;
                            }
                        }
                    }

                    // the number of tokens added to the batch for the current slot
                    const auto n_tokens_cur = batch.size() - n_tokens_prev;

                    const auto n_tokens_start = slot.prompt.n_tokens() - n_tokens_cur;

                    const bool near_prompt_end = slot.task->n_tokens() < slot.prompt.n_tokens() + n_ubatch;

                    const bool is_user_start = spans.is_user_start(n_tokens_start);
                    const bool is_last_user_message = n_tokens_start == last_user_pos;

                    // entire prompt has been processed
                    if (slot.prompt.n_tokens() == slot.task->n_tokens()) {
                        slot.state = SLOT_STATE_DONE_PROMPT;

                        GGML_ASSERT(batch.size() > 0);

                        // extract the logits only for the last token
                        batch.set_output(batch.size() - 1, true);

                        slot.stats.n_gen = 0;
                        slot.i_batch     = batch.size() - 1;

                        slot.init_sampler();
                    } else {
                        // skip ordinary mid-prompt checkpoints, unless the batch starts a user
                        // message or we are near the end of the prompt
                        if (!is_user_start && !near_prompt_end) {
                            do_checkpoint = false;
                        }
                    }

                    const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
                    const auto pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id);

                    // nothing to checkpoint yet
                    // TODO: is this check needed?
                    if (do_checkpoint && pos_min < 0) {
                        do_checkpoint = false;
                    }

                    // do not checkpoint after mtmd chunks
                    do_checkpoint = do_checkpoint && !has_mtmd;

                    // no need to create checkpoints that are too close together, unless it's the last user message
                    do_checkpoint = do_checkpoint && (
                            slot.prompt.checkpoints.empty() ||
                            is_last_user_message || near_prompt_end ||
                            n_tokens_start > slot.prompt.checkpoints.back()->n_tokens + params_base.checkpoint_min_step);
                    SLT_DBG(slot, "main/do_checkpoint = %s, pos_min = %d, pos_max = %d\n", do_checkpoint ? "yes" : "no", pos_min, pos_max);

                    // note: we create the checkpoint before calling llama_decode(), so the current batch is not
                    //       yet processed and therefore it is not part of the checkpoint.
                    if (do_checkpoint) {
                        create_checkpoint(slot, n_tokens_cur, pos_min, pos_max);
                    }
                }

                if (!slot_batched) {
                    slot_batched = &slot;
                }
            });
        }
    }

    // see https://github.com/ggml-org/llama.cpp/pull/29969
    // TODO @ngxson : maybe remove this once we use "mixed" batch everywhere
    bool use_mixed_batch() const {
        return !llama_get_memory(ctx_tgt) || !llama_get_causal_attn(ctx_tgt);
    }

    // add the rest of the prompt to the batch, the mtmd chunks are added as embeddings next to the text tokens
    // the caller makes sure that it fits in the batch
    // returns false on error, the slot is then released
    bool add_prompt_mixed(server_slot & slot) {
        const auto & input_tokens = slot.task->tokens;
        const auto n_tokens_prev = batch.size();

        while (slot.prompt.n_tokens() < slot.task->n_tokens()) {
            const auto cur_token_idx = slot.prompt.n_tokens();
            const llama_token cur_tok = input_tokens[cur_token_idx];

            if (cur_tok != LLAMA_TOKEN_NULL) {
                const bool add_ok = batch.add(slot.id,
                    cur_tok,
                    /* pos       = */ slot.prompt.tokens.pos_next(),
                    /* output    = */ slot.need_embd(),
                    /* is_prompt = */ true);
                GGML_ASSERT(add_ok);
                if (!slot.task->decision.order.empty()) {
                    batch.set_decision_order(batch.size() - 1, slot.task->decision.order[cur_token_idx]);
                }
                slot.prompt.tokens.push_back(cur_tok);
                continue;
            }

            const auto & chunk = input_tokens.find_chunk(cur_token_idx);

            float * embd = slot.mbatch ? mtmd_batch_get_output_embd(slot.mbatch.get(), chunk.get()) : nullptr;
            if (!embd) {
                // encode on the worker thread, so we can still handle metrics tasks
                int32_t res = 0;
                queue_tasks.yield_to_queue([&]() {
                    res = encode_mtmd_chunk(slot, slot.mbatch, cur_token_idx);
                });
                embd = res == 0 ? mtmd_batch_get_output_embd(slot.mbatch.get(), chunk.get()) : nullptr;
            }

            if (!embd || !add_mtmd_chunk(slot, chunk.get(), embd, batch)) {
                SLT_ERR(slot, "%s", "failed to process mtmd chunk\n");
                // the batch must not keep the entries of a released slot
                batch.truncate(n_tokens_prev);
                send_error(slot, "failed to process mtmd chunk", ERROR_TYPE_SERVER);
                slot.release();
                return false;
            }

            slot.prompt.tokens.push_back_placeholder(chunk.get());
        }

        return true;
    }

    // returns true = success ; false = retry with smaller batch size
    // throw std::runtime_error on fatal error
    // [QLH-SPLIT] v5: ask the remote tail for candidates for exactly the sub-batch that will be
    // sampled from.
    //
    // The tail answers for one batch row -- the last one it was handed -- so this asks only when
    // that row is the row the sampler will read (`slot.i_batch`). Asking on every decode would be
    // harmless but would make a prefilling request pay the candidate round trip for batches
    // nobody samples from; asking when the row does not match would hand the sampler an array
    // that describes a different token. A context without remote stages ignores the call.
    void request_remote_candidates(int32_t off) {
        const int32_t last_row = off + (int32_t) batch.view.size() - 1;

        server_slot * hit = nullptr;
        for (auto & slot : slots) {
            if (!slot.remote_cand_ok || slot.i_batch != last_row) {
                continue;
            }
            if (slot.state != SLOT_STATE_DONE_PROMPT && slot.state != SLOT_STATE_GENERATING) {
                continue;
            }
            if (!slot.spec_draft.empty()) {
                // The speculative tail goes through its own sampler per row; never mixed in.
                continue;
            }
            hit = &slot;
            break;
        }

        if (hit == nullptr) {
            // Explicitly clear: a request that used the channel must not leave it armed for the
            // next frame (decode_split() clears it as well; this keeps the "asked for" state
            // visible from here too).
            llama_split_request_candidates(ctx_tgt, nullptr, nullptr, 0, nullptr, 0);
            return;
        }

        const common_sampler_candidates & p = hit->remote_cand;

        qlh_split::cand_req req{};
        req.top_k           = p.top_k;
        req.top_n_sigma     = p.top_n_sigma;
        req.penalty_last_n  = p.penalty_last_n;
        req.penalty_repeat  = p.penalty_repeat;
        req.penalty_freq    = p.penalty_freq;
        req.penalty_present = p.penalty_present;
        req.n_steps         = (uint8_t) p.n_steps;
        for (uint32_t i = 0; i < p.n_steps && i < sizeof(req.steps); ++i) {
            req.steps[i] = p.steps[i];
        }

        // The history window is read from the sampler itself, never tracked here: it is the same
        // ring the penalties step would use, so the tail cannot end up penalizing a different set
        // of tokens than the row path would.
        std::vector<llama_token> hist;
        common_sampler_candidates_history(hit->smpl.get(), hist);

        llama_split_request_candidates(ctx_tgt, &req,
                hit->remote_cand_bias.empty() ? nullptr : hit->remote_cand_bias.data(),
                (uint32_t) hit->remote_cand_bias.size(),
                hist.data(), (uint32_t) hist.size());
    }

    // [QLH-SPLIT] v7: arm the per-row verify channel for exactly the frame that will be decoded
    // next, or for none.
    //
    // The condition is not "some slot has drafts": it is "this frame *is* one slot's verification
    // batch, whole". That is stronger than it looks, and every part of it matters:
    //  - `spec_i_batch.front() == off` and its last index is the frame's last row: the tail
    //    answers one token per row of the frame, indexed by row, and the accept loop reads rows
    //    spec_i_batch[i]. A frame that also carried rows from somewhere else would make the
    //    channel answer for rows the sampler does not read, and -- worse -- would leave whatever
    //    those rows were for without logits (get_logits_ith() answers null for a verify frame).
    //  - `spec_i_batch.size() == batch.view.size()`: all rows of the frame are that batch's.
    //  - the slot's plan said the channel is usable (remote_verify_ok) and its sampler samples
    //    the argmax; otherwise can_speculate() is false and there are no drafts to begin with.
    //
    // A context without remote stages ignores the call, so a local chain is unaffected.
    void request_remote_verify_rows(int32_t off) {
        const int32_t n_view = (int32_t) batch.view.size();

        server_slot * hit = nullptr;
        for (auto & slot : slots) {
            if (!slot.remote_verify_ok || slot.spec_blocked) {
                continue;
            }
            if (slot.state != SLOT_STATE_GENERATING || !slot.can_speculate()) {
                continue;
            }
            if (slot.spec_draft.empty() || slot.spec_i_batch.empty()) {
                continue;
            }
            if (slot.spec_i_batch.size() != (size_t) n_view ||
                slot.spec_i_batch.front() != off ||
                slot.spec_i_batch.back() != off + n_view - 1) {
                continue;
            }
            if (off != 0) {
                // The answer is indexed by *row of the frame* (llama_split_verified_token), while
                // the accept loop reads rows by *batch index* -- the two are the same only when
                // the frame starts at row 0. For a remote chain that is always the case (one
                // sequence, so a spec frame is the whole batch), and requiring it here is what
                // keeps a future multi-slot batch from reading the wrong row's token: an
                // off-by-offset answer would be a wrong token, not a slower one.
                continue;
            }
            if (slot.spec_i_batch.size() > (size_t) qlh_split::MAX_VERIFY_ROWS) {
                // Cannot happen while spec_draft_cap is enforced, and the check is cheap: a
                // frame over the cap would be refused by the tail's own n_outputs_max.
                continue;
            }
            hit = &slot;
            break;
        }

        llama_split_request_verify_rows(ctx_tgt, hit != nullptr);
    }

    // [QLH-SPLIT] v8: arm the h_nextn rows for the frame that will be decoded next, or for none.
    //
    // The condition is weaker than the verify channel's on purpose. The row is needed for two
    // things: the accepted row of a *verify* frame (the next draft round's input) and the last row
    // of *any* frame (the first draft round's input, and every round after a plain decode). So the
    // frame only has to end with a row some drafting slot will use -- it does not have to be the
    // verify batch itself. Which rows come back follows from the frame: all of them when the verify
    // channel is armed on it, the last one otherwise (tools/split-stage/main.cpp).
    //
    // Arming it for a chain whose tail cannot draft would make the stage answer without the rows,
    // so the check for "this chain can draft at all" happens here, not there.
    void request_remote_nextn_rows() {
        if (!llama_context_split_mtp_ready(ctx_tgt)) {
            llama_split_request_nextn_rows(ctx_tgt, false);
            return;
        }

        bool want = false;
        for (auto & slot : slots) {
            // A prompt frame needs the row too: the first draft round starts from the last prompt
            // token's h, and there is no other decode that could produce it.
            if (slot.state != SLOT_STATE_GENERATING && slot.state != SLOT_STATE_PROCESSING_PROMPT &&
                slot.state != SLOT_STATE_STARTED && slot.state != SLOT_STATE_DONE_PROMPT) {
                continue;
            }
            if (!slot.can_speculate()) {
                continue;
            }
            want = true;
            break;
        }

        llama_split_request_nextn_rows(ctx_tgt, want);
    }

    // [QLH-SPLIT] v8/v9: discard a verification frame that produced no usable token, and leave the
    // slot ready to decode its first position through the row path.
    //
    // Called when the grammar or the reasoning budget rejected a row of the frame:
    // common_sampler_sample() answered LLAMA_TOKEN_NULL for it, so the frame's rows cannot be
    // turned into tokens -- but they *were* decoded, and the rows before the rejected one may
    // already have been accepted into the sampler. Three things are therefore undone, exactly as
    // the partial-acceptance path below does:
    //
    //  1. the KV: the whole frame (sampled + draft rows) is dropped, with the checkpoint when the
    //     context cannot do a bounded partial seq_rm, otherwise with a seq_rm back to the frame's
    //     first position;
    //  2. the prompt tokens: truncated to that same position, so the next
    //     handle_last_sampled_token() re-adds `sampled` where the frame started;
    //  3. the sampler: rolled back from the pre-frame clone, which restores the grammar *and* the
    //     reasoning budget -- without this the grammar would have consumed rows the request never
    //     emitted, and the budget would have been charged for them (both silent: the next plain
    //     decode would produce a different token, and a FORCING frame would force an end
    //     sequence the request should not have).
    //
    // The next round drafts nothing (skip_draft_once): the position is the one the frame failed
    // on, so a fresh draft would be verified at the same state and fail identically. The plain
    // decode that follows samples it through common_sampler_sample()'s row path, which applies the
    // reasoning budget and the grammar first and resamples -- the token this frame could not
    // produce.
    //
    // common_sampler_copy() covers rbudget because llama_sampler_copy() falls back to clone() for
    // a sampler without copy_state, and common_reasoning_budget_clone() deep-copies the whole ctx
    // (state, remaining, force_pos, and both matchers' aho-corasick state). That is asserted
    // indirectly on every rollback: a partial copy would show up as an off-by-one in the forced
    // sequence, i.e. as a different token, never as a crash.
    // [QLH-SPLIT] Returns false when the frame could not be rolled back -- the one case where the
    // request cannot continue. prompt.tokens has already been rewound by then, so a KV that still
    // holds the rejected rows would make the next token a function of state nobody owns; the
    // caller must not sample from it. See the failure branch at the end.
    // [QLH-SPLIT] Does this verification frame need the target checkpoint to be undone?
    //
    // Two different rollbacks can be needed, and only one of them fits in the snapshots the
    // ubatch itself wrote:
    //   - a partial acceptance removes at most n_draft rows. The ubatch of the frame wrote
    //     n_draft + 1 rows, so the recurrent memory allows a rollback of n_seq_tokens - 1 =
    //     n_draft rows (cell.rs_max) -- seq_rm can do this one, no checkpoint needed;
    //   - a frame the grammar or the reasoning budget abandons removes n_draft + 1 rows: the
    //     n_draft drafts *and* the row the frame was decoded for. That is one row more than
    //     the same ubatch allows, so it can never be done with seq_rm -- it always needs the
    //     checkpoint captured before the frame. This is the definition of "the frame is
    //     abandoned", see common/sampling.h: not one row of the frame is usable.
    //
    // A draft longer than n_rs_seq rows leaves the memory with rs_max == 0 (it only guarantees
    // the final state), so it needs the checkpoint for the partial case as well.
    //
    // Requests that cannot abandon a frame (no grammar, no reasoning budget -- i.e. no path in
    // common_sampler_sample_* returns an empty result) keep the cheap path: no per-frame state
    // capture, and the partial rollback goes straight through seq_rm.
    bool frame_needs_target_ckpt(const server_slot & slot, size_t n_draft) const {
        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
            return true;
        }
        if (ctx_tgt_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_RS) {
            return false;
        }
        const bool too_long = n_draft + 1 > (size_t) llama_n_rs_seq(ctx_tgt);
        const bool may_abandon =
            common_sampler_has_grammar(slot.smpl.get()) ||
            common_sampler_has_reasoning_budget(slot.smpl.get());
        return too_long || may_abandon;
    }

    bool rollback_verify_frame(server_slot & slot, common_sampler * smpl_save, size_t n_draft, bool use_ckpt_tgt) {
        const int32_t pos_first = (int32_t) (slot.prompt.n_tokens() - (n_draft + 1));
        bool rm_ok = true;

        if (use_ckpt_tgt) {
            const auto & ckpt = slot.spec_ckpt;

            SLT_DBG(slot, "restoring speculative checkpoint (pos_min = %d, pos_max = %d, size = %zu) "
                          "after a rejected verification frame\n",
                    ckpt.pos_min, ckpt.pos_max, ckpt.size());

            GGML_ASSERT(ckpt.load_tgt(slot.ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));

            if (slot.ctx_dft) {
                GGML_ASSERT(ckpt.load_dft(slot.ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
            }

            rm_ok = slot.mem.seq_rm(slot.id, ckpt.pos_max + 1, -1);

            slot.prompt.tokens.keep_first(ckpt.n_tokens);
        } else {
            // No checkpoint was captured for this frame: the context rolls back partial sequences
            // itself, and use_ckpt_tgt was false when the draft was made because the frame's
            // n_draft + 1 rows are inside the n_rs_seq bound (see the two call sites of
            // use_ckpt_tgt above -- they are the same condition on purpose).
            slot.prompt.tokens.keep_first(pos_first);
            rm_ok = slot.mem.seq_rm(slot.id, slot.prompt.tokens.pos_next(), -1);
        }

        if (!rm_ok) {
            // [QLH-SPLIT] The KV refused to drop the rejected frame (a hybrid memory whose
            // recurrent half cannot roll back that far, or a split chain that no longer agrees).
            // prompt.tokens is already rewound, so the next decode would attend to rows that no
            // longer exist in the prompt: that is a wrong token with no way to detect it later.
            // Fail this request instead of serving it, and release the slot's memory through the
            // one operation every memory type supports (a full clear) so the next request starts
            // whole. Note what is deliberately NOT done: no GGML_ABORT -- one refused trim must
            // not take a server down that has other slots to serve.
            SLT_ERR(slot, "the verification rollback could not remove the rejected frame from the "
                          "KV (%zu row(s) from pos %d); failing the request instead of sampling "
                          "from state that no longer matches the prompt\n",
                    n_draft + 1, pos_first);
            send_error(slot, "the verification rollback failed: the KV still holds the rejected "
                             "frame, so this request cannot continue on a consistent state "
                             "(retry, or turn speculative decoding off for this chain)");
            slot.prompt_clear();
            slot.release();
            return false;
        }

        slot.spec_draft.clear();
        slot.spec_draft_q.clear();
        slot.spec_is_replay = false;
        slot.skip_draft_once = true;

        common_sampler_copy(smpl_save, slot.smpl.get());

        // counted as a verification step that accepted nothing: "generated" already counted these
        // drafts when they were made, so leaving this out would make the acceptance rate look up
        slot.stats.n_draft_verif_steps += 1;

        if (trace > 0) {
            SLT_INF(slot, "accepted  0/%2zu draft tokens (row rejected by the grammar or the "
                          "reasoning budget; rolled back %zu row(s) at pos %d)\n",
                    n_draft, n_draft + 1, pos_first);
        }
        return true;
    }

    bool decode(int32_t & n_batch, int32_t off) {
        SRV_DBG("n_batch (effective) = %d, off = %d\n", n_batch, off);

        metrics_pre_decode();

        if (batch.size() == 0) {
            SRV_WRN("%s", "no tokens to decode\n");

            if (++n_empty_consecutive > 3) {
                GGML_ABORT("fatal error - please provide logs and repro in %s\n", "https://github.com/ggml-org/llama.cpp/pull/20277");
            }

            return true; // nothing to decode
        } else {
            n_empty_consecutive = 0;
        }

        // TODO @ngxson : dft model may have different n_embd than the tgt model, so we check & reject if that's the case
        // this case is not currently used by any models, but may need to be supported in the future
        if (spec && batch.has_embd()) {
            if (llama_model_n_embd_inp(model_dft) != llama_model_n_embd_inp(model_tgt)) {
                SRV_ERR("%s", "unsupported batch.has_embd + spec case\n");
                throw std::runtime_error("unsupported batch.has_embd + spec case");
            }
        }

        bool has_output = false;
        for (int i = off; i < off + batch.view.size(); ++i) {
            has_output |= batch.tokens[i].output;
        }

        // yield to the queue, so we can still handle metrics tasks while decoding
        // note: the sync is done here too, so that the wait is also covered by the yield
        int ret = 0;
        // [QLH-SPLIT] v5: arm the candidate channel for this frame before the decode that
        // produces the row it describes. Outside the yield block on purpose: the request and the
        // decode it belongs to must not be separated by another task.
        request_remote_candidates(off);
        // [QLH-SPLIT] v7: same for the per-row verify channel -- one argmax token per row instead
        // of one row of logits, for the frame that verifies this slot's drafts.
        request_remote_verify_rows(off);
        // [QLH-SPLIT] v8: and the h_nextn rows the MTP draft starts from (see the function).
        request_remote_nextn_rows();
        queue_tasks.yield_to_queue([&]() {
            ret = llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.view.get());
            if (ret == 0 && has_output) {
                llama_synchronize(ctx_tgt);
            }
        });

        if (ret != 0) {
            {
                std::string err;

                if (n_batch == 1 && ret == 1) {
                    // TODO: try to terminate only the largest active slot/sequence and continue with the rest
                    //       need to remove the tokens from the current batch too
                    err = "Context size has been exceeded.";
                }

                if (ret == -1) {
                    err = "Invalid input batch.";
                }

                if (ret < -1) {
                    // TODO: update slot state based on llama_memory_seq_pos_min() and llama_memory_seq_pos_max()
                    err = "Compute error.";

                    // [QLH-SPLIT] Name the reason when the chain is the reason. A remote stage
                    // that stopped answering is remembered by the memory layer, and every
                    // decode refuses from then on; without this the client (and a deep health
                    // probe) sees only "Compute error." and has to guess whether the model, the
                    // request or the network is at fault. Read on this thread on purpose: the
                    // reason string is written by the thread that drives the chain, which is
                    // the one running this function.
                    const char * why = llama_context_split_failure(ctx_tgt);
                    if (why != nullptr && why[0] != '\0') {
                        err += " the split chain is no longer trustworthy: ";
                        err += why;
                    }
                }

                // TODO: handle ret == 2 (abort) when we start aborting

                if (!err.empty()) {
                    SRV_ERR("%s off = %d, n_batch = %d, ret = %d\n", err.c_str(), off, n_batch, ret);

                    for (auto & slot : slots) {
                        if (slot.is_processing()) {
                            send_error(slot, err);
                            slot.release();

                            // note: it's complicated to keep track of how much of the current batch has been
                            //       processed before the error occurred, so we simply clear the entire context
                            slot.prompt_clear();
                        }
                    }

                    // stop, do not retry with smaller batch size
                    throw std::runtime_error(err);
                }
            }

            // retry with half the batch size to try to find a free slot in the KV cache
            if (!try_clear_idle_slots()) {
                n_batch /= 2;
            }

            SRV_WRN("failed to find free space in the KV cache, retrying with smaller batch size, off = %d, n_batch = %d, ret = %d\n", off, n_batch, ret);

            return false; // retry with the updated n_batch
        } else {
            // success, apply batch metrics
            metrics_post_decode(off, batch.view.size(), has_output);
        }

        // TODO: avoid restoring the draft context and re-evaluating the drafted tokens when not needed [TAG_SPEC_AVOID_DRAFT_REEVAL]
        //       for now, always re-evaluate for simplicity
        //       ref: https://github.com/ggml-org/llama.cpp/pull/22728#issuecomment-4400925384
        if (spec) {
            bool ok = true;
            queue_tasks.yield_to_queue([&]() {
                ok = common_speculative_process(spec.get(), batch.view);
            });

            if (!ok) {
                SRV_ERR("%s", "failed to process speculative batch\n");

                // TODO: handle error
                throw std::runtime_error("failed to process speculative batch");
            }
        }

        // handle `n_cmpl > 1` tasks - when the main prompt is processed, activate all child tasks too
        for (auto & slot : slots) {
            if (slot.state == SLOT_STATE_DONE_PROMPT && slot.task->is_parent()) {
                std::vector<server_slot *> children;
                for (auto & other : slots) {
                    if (other.state == SLOT_STATE_WAIT_OTHER && slot.task->id == other.task->id_parent) {
                        children.push_back(&other);
                    }
                }

                // all children slots should already launched by launch_slots_with_parent_task()
                // copy state to the child slots
                for (auto & child : children) {
                    SLT_TRC(slot, " - copying state to child %d\n", child->id);

                    GGML_ASSERT(child->state == SLOT_STATE_WAIT_OTHER);
                    // children with their own prompt are started at the end of the shared prefix
                    GGML_ASSERT(slot.task->n_tokens_shared == 0);

                    slot.copy_state_to(*child);
                    child->state = SLOT_STATE_DONE_PROMPT;
                }
            }
        }

        return true;
    }

    // [QLH-SPLIT] v5: sample the next token, from the candidate array when this request is on
    // that path.
    //
    // The array is what the tail computed for the row the request is about to read; the sampler
    // then runs its own remaining steps on it, which is the whole point of the split (see
    // common/sampling.h). When there is no array -- a request that fell back, or a decode that
    // did not ask -- this is the ordinary path, unchanged.
    llama_token sample_next_token(server_slot & slot, int32_t idx) {
        if (slot.remote_cand_ok) {
            const llama_split_token_cand * cands = nullptr;
            int32_t row = -1;
            const uint32_t n = llama_split_take_candidates(slot.ctx_tgt, &cands, &row);
            if (n > 0) {
                if (row != idx) {
                    // Asking for the last row and sampling another one is a defect, not a
                    // fallback: the row path cannot be reconstructed here (no logits were
                    // transferred), so fail the request instead of sampling from a token the
                    // array does not describe.
                    throw std::runtime_error(string_format(
                            "[split] the tail answered candidates for batch row %d but the sampler "
                            "asked for row %d", row, idx));
                }
                std::vector<llama_token> ids(n);
                std::vector<float>       logits(n);
                for (uint32_t i = 0; i < n; ++i) {
                    ids[i]    = cands[i].id;
                    logits[i] = cands[i].logit;
                }
                return common_sampler_sample_candidates(slot.smpl.get(), ids.data(), logits.data(), n);
            }
        }

        return common_sampler_sample(slot.smpl.get(), slot.ctx_tgt, idx);
    }

    void post_decode(int32_t n_batch_tokens, int32_t off) {
        // for checking if a given batch index is inside the current sub-batch
        auto is_inside_view = [&](int32_t idx) {
            return idx >= off && idx < off + n_batch_tokens;
        };

        // TODO @ngxson : it's tricky to make sub-batch compatible with common_sampler_sample_and_accept_n,
        // so for now we will throw an error in this case: https://github.com/ggml-org/llama.cpp/issues/24840
        iterate(slots, [&](server_slot & slot) {
            for (auto & i : slot.spec_i_batch) {
                if (!is_inside_view(i)) {
                    throw std::runtime_error(string_format("speculative batch index %d is not inside the current sub-batch [%d, %d)", i, off, off + n_batch_tokens));
                }
            }
        });

        auto accept_special_token = [&](server_slot & slot, llama_token token) {
            return params_base.special ||
                slot.task->params.sampling.preserved_tokens.find(token) != slot.task->params.sampling.preserved_tokens.end();
        };

        iterate(slots, [&](server_slot & slot) {
            // optionally send prompt processing progress
            if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_DONE_PROMPT) {
                if (slot.task->params.stream && slot.task->params.return_progress) {
                    send_partial_response(slot, {}, true);
                }
            }

            if (!is_inside_view(slot.i_batch)) {
                // the required token not in this sub-batch, skip
                return;
            }

            if (slot.state == SLOT_STATE_DONE_PROMPT) {
                if (slot.task->type == SERVER_TASK_TYPE_EMBEDDING) {
                    // prompt evaluated for embedding
                    send_embedding(slot, batch.view);
                    slot.release();
                    slot.i_batch = -1;
                    return;
                }

                if (slot.task->type == SERVER_TASK_TYPE_RERANK) {
                    send_rerank(slot, batch.view);
                    slot.release();
                    slot.i_batch = -1;
                    return;
                }

                if (slot.task->type == SERVER_TASK_TYPE_DECISION) {
                    send_decision(slot, batch.view, slot.i_batch - off);
                    slot.release();
                    slot.i_batch = -1;
                    return;
                }

                GGML_ASSERT(slot.task->need_sampling());

                // prompt evaluated for next-token prediction
                slot.state = SLOT_STATE_GENERATING;

                // [QLH-SPLIT] begin() used to be called here; it moved to the SLOT_STATE_STARTED
                // branch of batch_building (see the comment there) so that the prompt's own
                // process() batches are inside the window it opens. Nothing is left to do here.
            } else if (slot.state != SLOT_STATE_GENERATING) {
                return;
            }

            if (slot.can_speculate() && !slot.spec_draft.empty()) {
                return; // sample using speculative decoding
            }

            // shifted according to the current sub-batch
            const int tok_idx = slot.i_batch - off;

            llama_token id;
            {
                scoped_timer timer(t_sampl, n_sampl);
                id = sample_next_token(slot, tok_idx);
            }

            slot.i_batch = -1;

            common_sampler_accept(slot.smpl.get(), id, true);

            // here we have synchronized the llama_context (due to the sampling above), so we can do time measurement
            const int64_t t_now = ggml_time_us();

            slot.stats.n_gen += 1;

            if (slot.stats.n_gen == 1) {
                slot.stats.update_prompt_last();
                slot.t_print_last = t_now;
                slot.n_gen_last = 0;
            }

            slot.stats.update_gen_last();

            completion_token_output result;
            result.tok          = id;
            result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, accept_special_token(slot, result.tok));
            result.prob         = 1.0f; // TODO: set it here instead of doing inside populate_token_probs

            if (slot.task->params.sampling.n_probs > 0) {
                populate_token_probs(slot, result, slot.task->params.post_sampling_probs, params_base.special, tok_idx);
            }

            if (!process_token(result, slot)) {
                // release slot because of stop condition
                slot.print_timings();
                send_final_response(slot);
                slot.release();

                return;
            }

            slot.print_timings_tg();
        });

        // speculative decoding - main model sample and accept
        iterate(slots, [&](server_slot & slot) {
            if (slot.state != SLOT_STATE_GENERATING || !slot.can_speculate() ||
                    slot.spec_draft.empty() || slot.spec_i_batch.empty()) {
                return;
            }

            // save the original draft size
            const size_t n_draft = slot.spec_draft.size();

            GGML_ASSERT(n_draft > 0);

            // batch indices of the draft tokens, used to get the token probabilities below
            auto spec_i_batch = std::move(slot.spec_i_batch);

            // verify and try to accept the draft
            {
                common_sampler_ptr smpl_save(common_sampler_clone(slot.smpl.get()));

                GGML_ASSERT(spec_i_batch.size() == n_draft + 1);
                GGML_ASSERT(slot.spec_draft_q.empty() || (slot.spec_draft_q.size() == slot.spec_draft.size()));
                const auto & synth_probs = common_speculative_get_synth_probs(spec.get());

                // drafters that fill no distribution fall back here
                const bool use_rejection = slot.use_spec_rejection() && !slot.spec_draft_q.empty();

                // [QLH-SPLIT] A partial acceptance removes at most n_draft rows, which is the
                // n_seq_tokens - 1 = n_draft bound of the ubatch that wrote them, so seq_rm can
                // always do it; only a frame longer than n_rs_seq rows leaves the memory with
                // rs_max == 0 and needs the checkpoint here.
                const bool use_ckpt_tgt =
                    ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                    (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && n_draft > (size_t) llama_n_rs_seq(ctx_tgt));

                // [QLH-SPLIT] The whole-frame rollback removes n_draft + 1 rows, one more than
                // the same ubatch allows, so it can never go through seq_rm -- it needs the
                // checkpoint captured before the frame, and "captured" is exactly "spec_ckpt is
                // not empty" (see frame_needs_target_ckpt and the clear() at the start of a
                // frame). Keying it on the same function again would risk the two evaluations
                // disagreeing, and a rollback through the wrong path is a refused seq_rm (a 500
                // on the request, "the verification rollback failed").
                const bool frame_needs_ckpt = !slot.spec_ckpt.empty();

                std::vector<llama_token> accepted;
                if (!synth_probs.empty()) {
                    // synthetic acceptance replaces verification entirely, so it comes first
                    accepted = server_sample_and_accept_synth(
                            slot.smpl.get(), slot.ctx_tgt, spec_i_batch, slot.spec_draft,
                            synth_probs, slot.spec_synth_rng, slot.spec_is_replay);
                } else if (slot.spec_is_replay && slot.use_spec_rejection()) {
                    accepted = server_accept_replay(slot.smpl.get(), slot.ctx_tgt, spec_i_batch, slot.spec_draft);
                } else if (use_rejection) {
                    accepted = common_sampler_sample_and_accept_n_rejection(slot.smpl.get(), slot.ctx_tgt, spec_i_batch, slot.spec_draft, slot.spec_draft_q);
                } else {
                    accepted = common_sampler_sample_and_accept_n(slot.smpl.get(), slot.ctx_tgt, spec_i_batch, slot.spec_draft);
                }

                // [QLH-SPLIT] v8/v9: an empty result means the grammar or the reasoning budget
                // rejected a row of the frame (common_sampler_sample answered LLAMA_TOKEN_NULL
                // for it), so not one token of the frame is usable -- not even the rows before
                // it, see the contract in common/sampling.h. The frame itself is already
                // decoded, so rewind it and let the next round sample that position with a plain
                // decode.
                if (accepted.empty()) {
                    // The one path that can return nothing: a verified row the grammar or the
                    // reasoning budget rejected. Every other path ends in a token (mismatch,
                    // rejection sampling, replay and synthetic acceptance all push at least one),
                    // so an empty result with neither sampler would mean a defect, not a fallback.
                    GGML_ASSERT((common_sampler_has_grammar(slot.smpl.get()) ||
                                 common_sampler_has_reasoning_budget(slot.smpl.get())) &&
                            "only a grammar or a reasoning budget can abandon a verification frame");
                    GGML_ASSERT(!frame_needs_ckpt || (!slot.spec_ckpt.empty() &&
                            "a whole-frame rollback needs the checkpoint captured before the frame"));
                    // false = the rollback itself failed and the request was already failed and
                    // released inside; nothing else may touch this slot.
                    (void) rollback_verify_frame(slot, smpl_save.get(), n_draft, frame_needs_ckpt);
                    return;
                }

                GGML_ASSERT(accepted.size() >= 1);

                const uint32_t n_rollback = slot.spec_draft.size() + 1 - accepted.size();

                // check for partial draft acceptance
                if (n_rollback > 0) {
                    if (use_ckpt_tgt) {
                        if (trace > 0) {
                            SLT_INF(slot, "accepted %2zu/%2zu draft tokens (restore checkpoint)\n", accepted.size() - 1, slot.spec_draft.size());
                        }

                        // partial acceptance is not supported by the context -> truncate the draft and restore the state
                        slot.spec_is_replay = true;
                        slot.spec_draft = std::move(accepted);

                        const auto & ckpt = slot.spec_ckpt;

                        SLT_DBG(slot, "restoring speculative checkpoint (pos_min = %d, pos_max = %d, size = %zu)\n", ckpt.pos_min, ckpt.pos_max, ckpt.size());

                        GGML_ASSERT(ckpt.load_tgt(slot.ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));

                        if (slot.ctx_dft) {
                            GGML_ASSERT(ckpt.load_dft(slot.ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
                        }

                        // [QLH-SPLIT] This is mid-generation, so the degradation of the prompt
                        // path (clear and re-process) would throw away tokens the client already
                        // has. What is preserved instead is exactly that: the response ends here
                        // with everything generated so far, and the slot's memory is cleared
                        // (full removal, which every memory type supports) so the next request
                        // starts from a consistent state. What is NOT done is the thing that
                        // would be silently wrong -- carrying on from a KV that still holds the
                        // rejected rows of a frame whose prompt has already been rewound.
                        if (!slot.mem.seq_rm(slot.id, ckpt.pos_max + 1, -1)) {
                            SLT_ERR(slot, "the speculative partial-accept rollback could not remove "
                                          "the rejected rows from the KV; ending the response at "
                                          "%d generated tokens instead of continuing on state that "
                                          "no longer matches the prompt\n", (int) slot.stats.n_gen);
                            slot.has_next_token = false;
                            slot.stop           = STOP_TYPE_LIMIT;
                            slot.print_timings();
                            send_final_response(slot);
                            slot.prompt_clear();
                            slot.release();
                            return;
                        }

                        slot.prompt.tokens.keep_first(ckpt.n_tokens);
                        common_sampler_copy(smpl_save.get(), slot.smpl.get());

                        return;
                    }
                }

                if (trace > 0) {
                    SLT_INF(slot, "accepted %2zu/%2zu draft tokens\n", accepted.size() - 1, n_draft);
                }

                common_speculative_accept(spec.get(), slot.id, accepted.size() - 1);

                slot.spec_draft = std::move(accepted);
            }

            const auto ids = std::move(slot.spec_draft);

            size_t n_accepted = ids.size() - 1;
            if (slot.spec_is_replay && n_accepted > 0) {
                n_accepted--;
            }
            slot.spec_is_replay = false;

            slot.stats.update_gen_last();

            // update how many tokens out of those tested were accepted
            slot.stats.n_draft_accepted += n_accepted;
            slot.stats.n_draft_verif_steps += 1;

            auto & n_accepted_per_pos = slot.n_accepted_per_pos;
            if (n_accepted_per_pos.empty()) {
                n_accepted_per_pos.resize(common_speculative_n_max(spec.get()), 0);
            }
            for (size_t i = 0; i < n_accepted && i < n_accepted_per_pos.size(); ++i) {
                n_accepted_per_pos[i]++;
            }

            // add accepted tokens to the prompt
            slot.prompt.tokens.keep_first(slot.prompt.n_tokens() - n_draft);
            slot.prompt.tokens.insert({ids.begin(), ids.end() - 1});

            slot.sampled = ids.back(); // last accepted token
            SLT_DBG(slot, "add accepted tokens: sampled=%d, ids.size=%zu, n_draft=%zu\n", slot.sampled, ids.size(), n_draft);

            // [QLH-SPLIT] Same rule one level down: mid-generation, the tokens of this frame are
            // already accepted and are about to be handed to the client, so they are still sent --
            // but the rows of the drafts that were *not* accepted must leave the KV before the
            // next decode, or it will attend to tokens the prompt no longer has. When that removal
            // is refused the response ends after these tokens (nothing generated is lost) and the
            // memory is cleared rather than carried forward inconsistent.
            const bool trim_ok = slot.mem.seq_rm(slot.id, slot.prompt.tokens.pos_next(), -1);
            if (!trim_ok) {
                SLT_ERR(slot, "%s", "the speculative accept could not trim the rejected draft rows "
                                    "from the KV; the response ends after the accepted tokens\n");
            }

            for (size_t i = 0; i < ids.size(); ++i) {
                completion_token_output result;

                result.tok          = ids[i];
                result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, accept_special_token(slot, result.tok));
                result.prob         = 1.0f; // set later

                // post_sampling_probs is not supported with speculative decoding
                if (slot.task->params.sampling.n_probs > 0 && !slot.task->params.post_sampling_probs) {
                    populate_token_probs(slot, result, false, params_base.special, spec_i_batch[i]);
                }

                slot.stats.n_gen += 1;

                if (!process_token(result, slot)) {
                    slot.print_timings();
                    send_final_response(slot);
                    slot.release();

                    return;
                }
            }

            if (!trim_ok) {
                // Everything accepted above is already in `slot.generated` (process_token sent it),
                // so ending here loses nothing the client has seen; what it avoids is the next
                // decode reading the un-accepted draft rows that the failed trim left behind.
                slot.has_next_token = false;
                slot.stop           = STOP_TYPE_LIMIT;
                slot.print_timings();
                send_final_response(slot);
                slot.prompt_clear();
                slot.release();
                return;
            }

            slot.print_timings_tg();

            SLT_DBG(slot, "accepted %d/%d draft tokens, new n_tokens = %d\n", (int) n_accepted, (int) n_draft, slot.prompt.n_tokens());
        });
    }

    // context size of a single slot, capped by --kv-unified-per-slot and by the training context of the model
    int n_ctx_slot() const {
        int res = llama_n_ctx_seq(ctx_tgt);

        if (params_base.kv_unified_per_slot > 0) {
            res = std::min(res, params_base.kv_unified_per_slot);
        }

        return std::min(res, llama_model_n_ctx_train(model_tgt));
    }

    server_response_reader get_response_reader() {
        return server_response_reader(queue_tasks, queue_results, HTTP_POLLING_SECONDS);
    }

    //
    // metrics helpers
    //

    // call before submitting a decode, so that the queued prompt stats can be timed
    void metrics_pre_decode() {
        t_decode_start = ggml_time_us();
    }

    // the batch is submitted, but its compute may not be done yet
    void metrics_queue_prompt(uint64_t n_tokens) {
        if (n_tokens == 0) {
            return;
        }
        if (n_prompt_queued == 0) {
            t_prompt_start = t_decode_start;
        }
        n_prompt_queued += n_tokens;
    }

    // call only after the context is synchronized, otherwise the time is meaningless
    void metrics_flush_prompt() {
        if (n_prompt_queued == 0) {
            return;
        }
        metrics.add_prompt(n_prompt_queued, ggml_time_us() - t_prompt_start);
        n_prompt_queued = 0;
    }

    // has_output is computed by the caller, which also already synchronized the context if it is set
    void metrics_post_decode(int32_t off, int32_t n_tokens, bool has_output) {
        metrics.n_decode++;
        for (const auto & slot : slots) {
            if (slot.is_processing()) {
                metrics.n_busy_slots++;
            }
            metrics.n_tokens_max = std::max(metrics.n_tokens_max, (uint64_t) slot.prompt.n_tokens());
        }

        // apply enqueued prompt tokens stats
        // note: a slot can be released before we get here, which clears its stats
        //       the tokens were still computed, counted in the global metrics, not in slot
        uint64_t n_prompt_tokens = 0;

        for (int i = off; i < off + n_tokens; ++i) {
            const auto & t = batch.tokens[i];

            if (!t.is_prompt) {
                continue; // generated tokens are handled after sampling
            }

            n_prompt_tokens++;

            auto & slot = slots[t.id_slot];
            if (slot.stats.is_set()) {
                slot.stats.n_prompt_processed++;
            }
        }

        metrics_queue_prompt(n_prompt_tokens);

        if (has_output) {
            // the context is already synchronized, so the timings are correct
            metrics_flush_prompt();
        }

        // advance the prompt timing of the slots that had tokens in this batch
        // note: a second pass, it must run after the sync to reflect the compute
        const int64_t t_now = ggml_time_us();
        for (int i = off; i < off + n_tokens; ++i) {
            const auto & t = batch.tokens[i];
            auto & slot = slots[t.id_slot];
            if (t.is_prompt && slot.stats.is_set()) {
                slot.stats.set_prompt_last(t_now);
            }
        }
    }

    // flush any queued prompt metrics if all slots are now idle
    void metrics_flush_idle() {
        if (n_prompt_queued == 0) {
            return;
        }

        llama_synchronize(ctx_tgt);
        metrics_flush_prompt();
    }

    void metrics_on_prediction(const server_slot & slot) {
        const uint64_t t_us    = slot.stats.t_gen_us();
        const uint64_t n       = slot.stats.n_gen;
        const uint64_t n_steps = slot.stats.n_gen_steps();

        metrics.predict       .add(n, n_steps, t_us);
        metrics.predict_bucket.add(n, n_steps, t_us);

        metrics.n_draft_tokens      += slot.stats.n_draft_tokens;
        metrics.n_draft_accepted    += slot.stats.n_draft_accepted;
        metrics.n_draft_verif_steps += slot.stats.n_draft_verif_steps;

        auto & dst = metrics.n_accepted_per_pos;
        const auto & src = slot.n_accepted_per_pos;

        if (dst.size() < src.size()) {
            dst.resize(src.size(), 0);
        }
        for (size_t i = 0; i < src.size(); i++) {
            dst[i] += src[i];
        }
    }
};

//
// server_context (public API)
//

server_context::server_context() : impl(new server_context_impl()) {}
server_context::~server_context() = default;

bool server_context::load_model(common_params & params) {
    return impl->load_model(params);
}

void server_context::start_loop() {
    auto & params = impl->params_base;
    impl->queue_tasks.start_loop(params.sleep_idle_seconds * 1000);
}

void server_context::terminate() {
    impl->queue_tasks.terminate();
}

llama_context * server_context::get_llama_context() const {
    return impl->ctx_tgt;
}

server_response_reader server_context::get_response_reader() {
    return impl->get_response_reader();
}

server_context_meta server_context::get_meta() const {
    auto bos_id = llama_vocab_bos(impl->vocab);
    auto eos_id = llama_vocab_eos(impl->vocab);
    auto bos_token_str = bos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx_tgt, bos_id, true) : "";
    auto eos_token_str = eos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx_tgt, eos_id, true) : "";

    const char * ftype_name = llama_ftype_name(llama_model_ftype(impl->model_tgt));

    std::vector<std::string> tts_voices;
    if (impl->mctx) {
        for (int32_t i = 0; i < mtmd_gen_audio_get_info(impl->mctx).n_voices; i++) {
            tts_voices.push_back(mtmd_gen_audio_voice_name(impl->mctx, i));
        }
    }

    return server_context_meta {
        /* build_info              */ std::string(llama_build_info()),
        /* model_name              */ impl->model_name,
        /* model_aliases           */ impl->model_aliases,
        /* model_tags              */ impl->model_tags,
        /* model_path              */ impl->params_base.model.path,
        /* model_output_modalities */ server_model_output_modalities(impl->model_tgt, impl->has_cap_chat(), impl->has_cap_tts()),
        /* has_mtmd                */ impl->mctx != nullptr,
        /* has_inp_image           */ impl->chat_params.allow_image,
        /* has_inp_audio           */ impl->chat_params.allow_audio,
        /* has_inp_video           */ impl->chat_params.allow_video,
        /* has_cap_chat            */ impl->has_cap_chat(),
        /* has_cap_tts             */ impl->has_cap_tts(),
        /* has_audio_stream        */ impl->mtmd_cap.audio_stream,
        /* tts_voices              */ std::move(tts_voices),
        /* json_ui_settings        */ impl->json_ui_settings,
        /* slot_n_ctx              */ impl->n_ctx_slot(),
        /* pooling_type            */ llama_pooling_type(impl->ctx_tgt),

        /* chat_params             */ impl->chat_params,
        /* chat_template_caps      */ common_chat_templates_get_caps(impl->chat_params.tmpls.get()),

        /* bos_token_str           */ bos_token_str,
        /* eos_token_str           */ eos_token_str,
        /* fim_pre_token           */ llama_vocab_fim_pre(impl->vocab),
        /* fim_sub_token           */ llama_vocab_fim_suf(impl->vocab),
        /* fim_mid_token           */ llama_vocab_fim_mid(impl->vocab),
        /* fim_pad_token           */ llama_vocab_fim_pad(impl->vocab),
        /* fim_rep_token           */ llama_vocab_fim_rep(impl->vocab),
        /* fim_sep_token           */ llama_vocab_fim_sep(impl->vocab),

        /* logit_bias_eog          */ impl->params_base.sampling.logit_bias_eog,

        /* model_vocab_type        */ llama_vocab_type(impl->vocab),
        /* model_vocab_n_tokens    */ llama_vocab_n_tokens(impl->vocab),
        /* model_n_ctx_train       */ llama_model_n_ctx_train(impl->model_tgt),
        /* model_n_embd_inp        */ llama_model_n_embd(impl->model_tgt),
        /* model_n_params          */ llama_model_n_params(impl->model_tgt),
        /* model_size              */ llama_model_size(impl->model_tgt),
        /* model_ftype             */ ftype_name,
    };
}

// generator-like API for HTTP response generation
// may have bypass_sleep = true if the task does not use ctx_server
struct server_res_generator : server_res_spipe {
    server_response_reader rd;
    server_res_generator(server_queue & queue_tasks, server_response & queue_results, int sleep_idle_seconds, bool bypass_sleep = false)
            : rd(queue_tasks, queue_results, HTTP_POLLING_SECONDS) {
        // fast path in case sleeping is disabled
        bypass_sleep |= sleep_idle_seconds < 0;
        if (!bypass_sleep) {
            queue_tasks.wait_until_no_sleep();
        }
    }
    void ok(const json & response_data) {
        status = 200;
        data = safe_json_to_str(response_data);
    }
    void error(const json & error_data) {
        status = json_value(error_data, "code", 500);
        data = safe_json_to_str({{ "error", error_data }});
    }
};

void server_context::set_state_callback(server_state_callback_t callback) {
    impl->callback_state = std::move(callback);
}

//
// server_routes
//

std::unique_ptr<server_res_generator> server_routes::handle_completions_impl(
            const server_http_req & req,
            server_task_type type,
            const json & data,
            const std::vector<raw_buffer> & files,
            task_response_type res_type,
            const common_chat_session & chat_session) {
    GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION || type == SERVER_TASK_TYPE_INFILL);

    auto res = create_response();
    auto completion_id = gen_chatcmplid();
    auto & rd = res->rd;
    auto & params = this->params;

    res->set_req(&req); // will also set spipe if needed

    if (!ctx_server.has_cap_chat()) {
        res->error(format_error_response("this server does not support chat/completions", ERROR_TYPE_NOT_SUPPORTED));
        return res;
    }

    int32_t sse_ping_interval = params.sse_ping_interval;

    try {
        std::vector<server_task> tasks;

        const auto & prompt = data.at("prompt");
        // TODO: this log can become very long, put it behind a flag or think about a more compact format
        //SRV_DBG("Prompt: %s\n", prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());

        if (!params.path_prompts_log_dir.empty()) {
            const auto file_path = std::filesystem::path(params.path_prompts_log_dir) / string_format("%012" PRId64 ".txt", ggml_time_ms());
            std::ofstream f(file_path);
            if (f) {
                f << (prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());
            } else {
                SRV_ERR("failed to create %s\n", file_path.string().c_str());
            }
        }

        // process prompt
        std::vector<server_tokens> inputs;

        if (res_type != TASK_RESPONSE_TYPE_NONE && ctx_server.mctx != nullptr) {
            // This is the case used by OAI compatible chat path with MTMD. TODO It can be moved to the path below.
            inputs.push_back(process_mtmd_prompt(ctx_server.mctx, prompt.get<std::string>(), files, ctx_server.init_opt));
        } else {
            // Everything else, including multimodal completions.
            inputs = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true, ctx_server.init_opt);
        }

        // tasks.reserve(inputs.size()); // TODO: this is inaccurate due to child tasks

        for (size_t i = 0; i < inputs.size(); i++) {
            server_task task = server_task(type);

            task.id = rd.get_new_id();

            task.tokens = std::move(inputs[i]);
            task.params = server_schema::eval_llama_cmpl_schema(
                    ctx_server.vocab,
                    params,
                    meta->logit_bias_eog,
                    data);

            task.apply_chat_session(chat_session);

            task.id_slot = json_value(data, "id_slot", -1);
            sse_ping_interval = task.params.sse_ping_interval;

            // OAI-compat
            task.params.res_type          = res_type;
            task.params.oaicompat_cmpl_id = completion_id;
            task.params.oaicompat_model   = meta->model_name;

            // prepare child tasks
            if (task.params.n_cmpl > 1) {
                int n_children = task.params.n_cmpl - 1;
                for (int j = 0; j < n_children; j++) {
                    task.add_child(task.id, rd.get_new_id());
                }
            }

            tasks.push_back(std::move(task));
        }

        rd.post_tasks(std::move(tasks), chat_session);
    } catch (const std::exception & e) {
        res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool stream = json_value(data, "stream", false);

    if (!stream) {
        // non-stream, wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            json arr = json::array();
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_cmpl_final*>(res.get()) != nullptr);
                arr.push_back(res->to_json());
            }
            GGML_ASSERT(!arr.empty() && "empty results");
            if (arr.size() == 1) {
                // if single request, return single object instead of array
                res->ok(arr[0]);
            } else if (res_type == TASK_RESPONSE_TYPE_OAI_CHAT || res_type == TASK_RESPONSE_TYPE_OAI_CMPL) {
                // if multiple results in OAI format, we need to re-format them
                json & usage   = arr[0]["usage"];
                json & choices = arr[0]["choices"];
                for (size_t i = 1; i < arr.size(); i++) {
                    choices.push_back(std::move(arr[i]["choices"][0]));
                }

                // OAI counts generated tokens across all choices
                int64_t n_completion_tokens = 0;
                for (auto & item : arr) {
                    n_completion_tokens += item["usage"]["completion_tokens"].get<int64_t>();
                }
                usage["completion_tokens"] = n_completion_tokens;
                usage["total_tokens"]      = usage["prompt_tokens"].get<int64_t>() + n_completion_tokens;

                res->ok(arr[0]);
            } else {
                // multi-results, non-OAI compat
                res->ok(arr);
            }
        }
    } else {
        // in streaming mode, the first error must be treated as non-stream response
        // this is to match the OAI API behavior
        // ref: https://github.com/ggml-org/llama.cpp/pull/16486#discussion_r2419657309
        auto first_result = rd.next(req.should_stop);
        if (first_result == nullptr) {
            GGML_ASSERT(req.should_stop());
            return res; // connection is closed
        }

        if (first_result->is_error()) {
            res->error(first_result->to_json());
            return res;
        }

        GGML_ASSERT(
            dynamic_cast<server_task_result_cmpl_partial*>(first_result.get()) != nullptr ||
            dynamic_cast<server_task_result_cmpl_final*>  (first_result.get()) != nullptr
        );

        // next responses are streamed
        // to be sent immediately
        json first_result_json = first_result->to_json();
        if (first_result_json == nullptr) {
            res->data = ""; // simply send HTTP headers and status code
        } else if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
            res->data = format_anthropic_sse(first_result_json);
        } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
            res->data = format_oai_resp_sse(first_result_json);
        } else {
            res->data = format_oai_sse(first_result_json);
        }
        res->status = 200;
        res->content_type = "text/event-stream";
        res->set_next([res_this = res.get(), res_type, sse_ping_interval](std::string & output) -> bool {
            static auto format_error = [](task_response_type res_type, const json & res_json) {
                if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                    return format_anthropic_sse({
                        {"event", "error"},
                        {"data", res_json},
                    });
                } else {
                    return format_oai_sse(json {{ "error", res_json }});
                }
            };

            auto effective_should_stop = [&res_this]() {
                return res_this->should_stop();
            };

            try {
                if (effective_should_stop()) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    return false; // should_stop condition met
                }

                if (!res_this->data.empty()) {
                    // flush the first chunk
                    output = std::move(res_this->data);
                    res_this->data.clear();
                    return true;
                }

                server_response_reader & rd = res_this->rd;

                // check if there is more data
                if (!rd.has_next()) {
                    switch (res_type) {
                        case TASK_RESPONSE_TYPE_NONE:
                        case TASK_RESPONSE_TYPE_OAI_RESP:
                        case TASK_RESPONSE_TYPE_ANTHROPIC:
                            output = "";
                            break;

                        default:
                            output = "data: [DONE]\n\n";
                            break;
                    }
                    SRV_DBG("%s", "all results received, terminating stream\n");
                    return false; // no more data, terminate
                }

                // receive subsequent results
                bool timeout = false;
                int64_t start_time = ggml_time_ms();
                auto result = rd.next([&timeout, &start_time, sse_ping_interval, &effective_should_stop]() {
                    if (effective_should_stop()) {
                        return true; // should_stop condition met
                    } else if (sse_ping_interval > 0 && ggml_time_ms() - start_time > (int64_t)sse_ping_interval * 1000) {
                        timeout = true;
                        return true; // timeout
                    }
                    return false;
                });

                if (timeout) {
                    // some clients may time out (e.g. undici) will time out if no data is received for a while, so we need to send a ping to keep the connection alive
                    SRV_DBG("%s", "sending SSE ping\n");
                    output = ":\n\n";
                    return true;
                }

                if (result == nullptr) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    GGML_ASSERT(effective_should_stop());
                    return false; // should_stop condition met
                }

                // send the results
                if (result->is_error()) {
                    json res_json = result->to_json();
                    output = format_error(res_type, res_json);
                    SRV_DBG("%s", "error received during streaming, terminating stream\n");
                    return false; // terminate on error
                } else {
                    GGML_ASSERT(
                        dynamic_cast<server_task_result_cmpl_partial*>(result.get()) != nullptr
                        || dynamic_cast<server_task_result_cmpl_final*>(result.get()) != nullptr
                    );
                    json res_json = result->to_json();
                    if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                        output = format_anthropic_sse(res_json);
                    } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
                        output = format_oai_resp_sse(res_json);
                    } else {
                        output = format_oai_sse(res_json);
                    }
                }

                // has next data, continue
                return true;

            } catch (const std::exception & e) {
                json error_json = format_error_response(e.what(), ERROR_TYPE_SERVER);
                output = format_error(res_type, error_json);

                // terminate on exception
                return false;
            }
        });
    }

    return res;
}

std::unique_ptr<server_res_generator> server_routes::create_response(bool bypass_sleep) {
    return std::make_unique<server_res_generator>(queue_tasks, queue_results, params.sleep_idle_seconds, bypass_sleep);
}

server_routes::server_routes(const common_params & params, server_context & ctx_server)
        : params(params),
          ctx_server(*ctx_server.impl),
          queue_tasks(ctx_server.impl->queue_tasks),
          queue_results(ctx_server.impl->queue_results) {
    init_routes();

    // note: this must be registered before load_model()
    //       so that on sleep phase, the callback is called before ctx is destroyed
    queue_tasks.on_sleeping_state([this](bool is_sleeping) {
        update_cached_responses(is_sleeping);
    });
}

static json get_res_model_info(const server_context_meta & meta) {
    // note: do NOT use ctx_server here, otherwise it's not possible to use this during sleep

    return {
        {"id",       meta.model_name},
        {"aliases",  meta.model_aliases},
        {"tags",     meta.model_tags},
        {"object",   "model"},
        {"architecture", server_model_architecture_json(
            meta.has_inp_image,
            meta.has_inp_audio,
            meta.has_inp_video,
            meta.model_output_modalities)},
        {"created",  std::time(0)},
        {"owned_by", "llamacpp"},
        {"meta",     {
            {"vocab_type",  meta.model_vocab_type},
            {"n_vocab",     meta.model_vocab_n_tokens},
            {"n_ctx",       meta.slot_n_ctx},
            {"n_ctx_train", meta.model_n_ctx_train},
            {"n_embd",      meta.model_n_embd_inp},
            {"n_params",    meta.model_n_params},
            {"size",        meta.model_size},
            {"ftype",       meta.model_ftype},
        }},
    };
}

static json get_res_models(const server_context_meta & meta) {
    // note: do NOT use ctx_server here, otherwise it's not possible to use this during sleep

    return json{
        {"models", json::array({
            {
                {"name",  meta.model_name},
                {"model", meta.model_name},
                {"modified_at", ""},
                {"size", ""},
                {"digest", ""}, // dummy value, llama.cpp does not support managing model file's hash
                {"type", "model"},
                {"description", ""},
                {"tags", json::array({""})},
                {"capabilities", meta.has_mtmd ? json::array({"completion","multimodal"}) : json::array({"completion"})},
                {"parameters", ""},
                {"details", {
                    {"parent_model", ""},
                    {"format", "gguf"},
                    {"family", ""},
                    {"families", json::array({""})},
                    {"parameter_size", ""},
                    {"quantization_level", ""}
                }}
            }
        })},
        {"object", "list"},
        {"data", json::array({
            get_res_model_info(meta),
        })}
    };
}

static json get_res_props(const server_context_meta & meta, const common_params & params, bool is_sleeping) {
    // note: do NOT use ctx_server here, otherwise it's not possible to use this during sleep

    task_params tparams;
    tparams.sampling = params.sampling;
    json default_generation_settings_for_props = json {
        { "params", tparams.to_json(true) },
        { "n_ctx",  meta.slot_n_ctx },
    };

    std::string tmpl_default = common_chat_templates_source(meta.chat_params.tmpls.get(), "");
    std::string tmpl_tools   = common_chat_templates_source(meta.chat_params.tmpls.get(), "tool_use");

    json props = {
        { "default_generation_settings", default_generation_settings_for_props },
        { "total_slots",                 params.n_parallel },
        { "model_alias",                 meta.model_name },
        { "model_ftype",                 meta.model_ftype },
        { "model_path",                  meta.model_path },
        { "modalities",                  json {
            {"vision", meta.has_inp_image},
            {"video",  meta.has_inp_video},
            {"audio",  meta.has_inp_audio},
        } },
        { "media_marker",                get_media_marker() },
        { "endpoint_slots",              params.endpoint_slots },
        { "endpoint_props",              params.endpoint_props },
        { "endpoint_metrics",            params.endpoint_metrics },
        { "ui",                          params.ui },
        { "ui_settings",                 meta.json_ui_settings },
        { "chat_template",               tmpl_default },
        { "chat_template_caps",          meta.chat_template_caps },
        { "bos_token",                   meta.bos_token_str },
        { "eos_token",                   meta.eos_token_str },
        { "build_info",                  meta.build_info },
        { "is_sleeping",                 is_sleeping },
        { "cors_proxy_enabled",          params.ui_mcp_proxy },
    };

    if (!params.server_connect_code.empty()) {
        props["connect_code"] = params.server_connect_code;
    }
    if (!meta.tts_voices.empty()) {
        props["voices"] = meta.tts_voices;
    }
    if (params.use_jinja) {
        if (!tmpl_tools.empty()) {
            props["chat_template_tool_use"] = tmpl_tools;
        }
    }

    return props;
}

json server_routes::get_model_info() const {
    return get_res_model_info(*meta);
}

// body of /tts and /v1/audio/speech; multipart form fields arrive as strings, the numeric and boolean params get their types back
static json parse_tts_body(const server_http_req & req) {
    json body = json::parse(req.body);
    for (const char * k : { "top_k", "n_predict", "max_tokens", "seed" }) {
        if (body.contains(k) && body[k].is_string()) {
            body[k] = std::stoll(body[k].get<std::string>());
        }
    }
    for (const char * k : { "top_p", "temperature", "repeat_penalty", "speed" }) {
        if (body.contains(k) && body[k].is_string()) {
            body[k] = std::stof(body[k].get<std::string>());
        }
    }
    if (body.contains("stream") && body["stream"].is_string()) {
        body["stream"] = body["stream"] == "true";
    }
    return body;
}

void server_routes::init_routes() {
    // IMPORTANT: all lambda functions must start with create_response()
    // this is to ensure that the server_res_generator can handle sleeping case correctly

    this->get_health = [this](const server_http_req & req) {
        // error and loading states are handled by middleware
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        // [QLH-SPLIT] `?deep=1` asks a different question from the one this endpoint answers by
        // default. A bare /health reports that the process has a model loaded -- that is what an
        // orchestrator polls, and it must not start flapping because a network stage had a bad
        // second (measured: /health answered 200 for the whole 120 s in which a suspended remote
        // stage was blocking a request). deep=1 instead drives one real token through the whole
        // chain, so it can only answer 200 if a request would have worked as well.
        if (req.get_param("deep") != "1") {
            res->ok({{"status", "ok"}});
            return res;
        }

        // The probe is a request, so it needs a model to run on. Asleep means the model was
        // unloaded to save power; waking it up from a health check would defeat the point of
        // sleeping, so say what is true instead. A router runs no model here at all (see
        // server_role_detect() in server.cpp: a server with no model is a router, and routes.
        // get_health is deliberately not proxied there).
        //
        // Both checks deliberately avoid the llama_context: it is owned by the main loop and the
        // shadowing `ctx_server` bool above is there to keep this handler from touching it. The
        // queue's sleep flag is mutex-protected and the model path is startup configuration, so
        // nothing here can race a decode or a sleep transition.
        if (queue_tasks.is_sleeping()) {
            res->error(format_error_response(
                    "the server is sleeping; a deep probe needs the model loaded (use /health without deep=1 for liveness)",
                    ERROR_TYPE_UNAVAILABLE));
            return res;
        }
        if (params.model.empty()) {
            res->error(format_error_response(
                    "no model is loaded on this server (router mode); a deep probe has nothing to drive",
                    ERROR_TYPE_UNAVAILABLE));
            return res;
        }

        // [QLH-SPLIT] The probe is an ordinary completion task, on purpose. It runs on the
        // thread that owns the context, through the slot machinery, so it cannot race a decode
        // and cannot leave a slot's KV bookkeeping describing tokens that its cache no longer
        // holds -- the two things a hand-rolled llama_decode() from an HTTP thread would do.
        // `cli = true` makes the main thread tokenize it, so this thread never touches the
        // vocab. One token, one generated token: enough for every stage of the chain to run.
        server_task task(SERVER_TASK_TYPE_COMPLETION);
        task.id         = res->rd.get_new_id();
        task.cli        = true;
        task.cli_prompt = "a";
        task.params.n_predict    = 1;
        // Decoding the probe token is the entire point: with the prompt cache on, the server may
        // find this 1-token prompt in the slot's cache and answer from logits it already has,
        // verifying nothing. cache_prompt = false forces a full re-processing of that token (and
        // a clear of the slot's KV) so every stage really runs.
        task.params.cache_prompt = false;
        // A streamed task signals "a slot took it" with a first progress result. That is what
        // separates "the probe is running" from "every slot is busy", which look identical from
        // outside until the 120 s stage timeout fires.
        task.params.stream           = true;
        task.params.sse_ping_interval = -1;
        task.params.res_type         = TASK_RESPONSE_TYPE_NONE;
        res->rd.post_task(std::move(task), true); // high-priority: a health probe should not queue behind traffic

        // 125 s: the stage connection gives up after 120 s (src/llama-split-remote.cpp), so a
        // probe that touched a dead stage answers well before this; the extra 5 s is for the
        // error to travel through the slot. Deliberately NOT shorter: a suspended stage has to
        // be allowed to reach its own timeout, otherwise a hung chain would be reported as
        // "busy" instead of as broken.
        const int64_t t0 = ggml_time_ms();
        bool started = false;
        while (true) {
            server_task_result_ptr result = res->rd.next([&] {
                return req.should_stop() || ggml_time_ms() - t0 > 125000;
            });
            if (result == nullptr) {
                if (started) {
                    res->error(json {
                        { "code",    503 },
                        { "type",    "unavailable_error" },
                        { "message", "the deep probe did not finish within 125 s: the chain stopped "
                                     "answering without failing. Run /health (shallow) to tell the "
                                     "server itself apart from the stage." },
                    });
                } else {
                    // No slot picked the probe up. That is not a chain failure -- a server busy
                    // with real work would fail its own requests only if the chain were broken,
                    // and those requests would have failed visibly. Report it as "not verified"
                    // rather than as a failed chain so a deep probe does not take a healthy busy
                    // machine out of rotation.
                    res->ok(json {
                        { "status", "ok" },
                        { "deep",   "busy" },
                        { "detail", "no slot was free for the probe within 125 s; the chain was not "
                                    "exercised. /health (shallow) still answers this server as alive" },
                    });
                }
                return res;
            }

            if (result->is_error()) {
                const std::string why = json_value(result->to_json(), "message", std::string("unknown error"));
                json body {
                    { "code",    503 },
                    { "type",    "unavailable_error" },
                    { "message", "the deep probe failed: " + why },
                };
                // Name the chain that failed: the endpoints are the operator's own
                // --split-remote arguments and the only name a stage has on this side.
                if (!params.split_remote.empty()) {
                    body["chain"] = params.split_remote;
                }
                res->error(body);
                return res;
            }

            if (result->is_stop()) {
                // A final result with no prompt tokens means the probe never decoded anything
                // (an empty tokenization is released as an empty completion), which must not be
                // reported as "the chain works".
                const auto * cmpl = dynamic_cast<const server_task_result_cmpl_final *>(result.get());
                const int32_t n_prompt = cmpl != nullptr ? cmpl->n_prompt_tokens : 0;
                if (n_prompt <= 0) {
                    res->error(json {
                        { "code",    503 },
                        { "type",    "unavailable_error" },
                        { "message", "the deep probe produced no prompt token to decode; nothing was "
                                     "verified (this is a probe bug, not a chain failure)" },
                    });
                    return res;
                }
                res->ok(json {
                    { "status",         "ok" },
                    { "deep",           "probed" },
                    { "n_prompt_tokens", n_prompt },
                    { "n_decoded",       cmpl->n_decoded },
                    { "detail", "one token was decoded through every stage of the chain" },
                });
                return res;
            }

            // a partial result: the slot took the task, from here on a timeout means "hung",
            // not "busy"
            started = true;
        }
    };

    this->get_metrics = [this](const server_http_req & req) {
        auto res = create_response(true);
        if (!params.endpoint_metrics) {
            res->error(format_error_response("This server does not support metrics endpoint. Start it with `--metrics`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // render response using cached_metrics
        auto use_cached_metrics = [&]() {
            std::unique_lock<std::mutex> lock(mutex_cache);
            res->headers["Process-Start-Time-Unix"] = std::to_string(cached_metrics.t_start);
            server_task_result_metrics tmp;
            tmp.metrics = cached_metrics;
            res->content_type = "text/plain; version=0.0.4";
            res->status = 200;
            res->data = tmp.to_metrics();
            // the gauges are averaged over the window between two scrapes
            cached_metrics.reset_bucket();
            should_reset_buckets = true;
        };

        if (queue_tasks.is_sleeping()) {
            use_cached_metrics();

        } else {
            // request slots data using task queue
            {
                server_task task(SERVER_TASK_TYPE_METRICS);
                task.id = res->rd.get_new_id();
                // the gauges are averaged over the window between two scrapes
                task.metrics_reset_bucket = true;
                res->rd.post_task(std::move(task), true); // high-priority task
            }

            // a task posted right before sleeping is never processed, do not wait for it
            auto result = res->rd.next([&]{
                return req.should_stop() || queue_tasks.is_sleeping();
            });
            if (!result) {
                if (!req.should_stop()) {
                    use_cached_metrics();
                }
                return res;
            }

            if (result->is_error()) {
                res->error(result->to_json());
                return res;
            }

            auto res_task = dynamic_cast<server_task_result_metrics*>(result.get());
            GGML_ASSERT(res_task != nullptr);

            res->headers["Process-Start-Time-Unix"] = std::to_string(res_task->metrics.t_start);
            res->content_type = "text/plain; version=0.0.4";
            res->status = 200;
            res->data = res_task->to_metrics();
        }

        return res;
    };

    this->get_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.endpoint_slots) {
            res->error(format_error_response("This server does not support slots endpoint. Start it with `--slots`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // request slots data using task queue
        {
            server_task task(SERVER_TASK_TYPE_SLOT_GET);
            task.id = res->rd.get_new_id();
            res->rd.post_task(std::move(task), true); // high-priority task
        }

        // get the result
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        auto * res_task = dynamic_cast<server_task_result_slots*>(result.get());
        GGML_ASSERT(res_task != nullptr);

        // optionally return "fail_on_no_slot" error
        if (!req.get_param("fail_on_no_slot").empty()) {
            if (res_task->n_idle_slots == 0) {
                res->error(format_error_response("no slot available", ERROR_TYPE_UNAVAILABLE));
                return res;
            }
        }

        res->ok(res_task->to_json());
        return res;
    };

    this->post_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (params.slot_save_path.empty()) {
            res->error(format_error_response("This server does not support slots action. Start it with `--slot-save-path`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::string id_slot_str = req.get_param("id_slot");

        int id_slot;
        try {
            id_slot = std::stoi(id_slot_str);
        } catch (const std::exception &) {
            res->error(format_error_response("Invalid slot ID", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::string action = req.get_param("action");

        if (action == "save") {
            return handle_slots_save(req, id_slot);
        }
        if (action == "restore") {
            return handle_slots_restore(req, id_slot);
        }
        if (action == "erase") {
            return handle_slots_erase(req, id_slot);
        }

        res->error(format_error_response("Invalid action", ERROR_TYPE_INVALID_REQUEST));
        return res;
    };

    this->get_props = [this](const server_http_req &) {
        auto res = create_response(true);
        // note: do NOT use ctx_server here, this endpoint must be accessible during sleep
        if (queue_tasks.is_sleeping()) {
            std::unique_lock<std::mutex> lock(mutex_cache);
            res->ok(cached_props);
        } else {
            res->ok(get_res_props(*meta, params, false));
        }
        return res;
    };

    this->post_props = [this](const server_http_req &) {
        auto res = create_response();
        if (!params.endpoint_props) {
            res->error(format_error_response("This server does not support changing global properties. Start it with `--props`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
        // update any props here

        res->ok({{ "success", true }});
        return res;
    };

    this->post_infill = [this](const server_http_req & req) {
        auto res = create_response();
        // check model compatibility
        std::string err;
        if (llama_vocab_fim_pre(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "prefix token is missing. ";
        }
        if (llama_vocab_fim_suf(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "suffix token is missing. ";
        }
        if (llama_vocab_fim_mid(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "middle token is missing. ";
        }
        if (!err.empty()) {
            res->error(format_error_response(string_format("Infill is not supported by this model: %s", err.c_str()), ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // validate input
        json data = json::parse(req.body);
        if (data.contains("prompt") && !data.at("prompt").is_string()) {
            // prompt is optional
            res->error(format_error_response("\"prompt\" must be a string", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_prefix")) {
            res->error(format_error_response("\"input_prefix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_suffix")) {
            res->error(format_error_response("\"input_suffix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (data.contains("input_extra") && !data.at("input_extra").is_array()) {
            // input_extra is optional
            res->error(format_error_response("\"input_extra\" must be an array of {\"filename\": string, \"text\": string}", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        json input_extra = json_value(data, "input_extra", json::array());
        for (const auto & chunk : input_extra) {
            // { "text": string, "filename": string }
            if (!chunk.contains("text") || !chunk.at("text").is_string()) {
                res->error(format_error_response("extra_context chunk must contain a \"text\" field with a string value", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            // filename is optional
            if (chunk.contains("filename") && !chunk.at("filename").is_string()) {
                res->error(format_error_response("extra_context chunk's \"filename\" field must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        }
        data["input_extra"] = input_extra; // default to empty array if it's not exist

        std::string prompt = json_value(data, "prompt", std::string());
        std::vector<server_tokens> tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, false, true, ctx_server.init_opt);
        SRV_DBG("creating infill tasks, n_prompts = %d\n", (int) tokenized_prompts.size());
        data["prompt"] = format_prompt_infill(
            ctx_server.vocab,
            data.at("input_prefix"),
            data.at("input_suffix"),
            data.at("input_extra"),
            params.n_batch,
            params.n_predict,
            meta->slot_n_ctx,
            params.spm_infill,
            tokenized_prompts[0].get_tokens() // TODO: this could maybe be multimodal.
        );

        std::vector<raw_buffer> files; // dummy
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_INFILL,
            data,
            files,
            TASK_RESPONSE_TYPE_NONE); // infill is not OAI compatible
    };

    this->post_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_NONE);
    };

    this->post_completions_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_OAI_CMPL);
    };

    this->post_chat_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = json::parse(req.body);
        common_chat_session session;
        json body_parsed = oaicompat_chat_params_parse(
            ctx_server.vocab,
            body,
            meta->chat_params,
            files,
            session);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_CHAT,
            session);
    };

    this->post_chat_completions_tok = [this](const server_http_req & req) {
        return handle_count_tokens(req, TASK_RESPONSE_TYPE_OAI_CHAT);
    };

    this->post_control = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        const std::string cmpl_id = json_value(body, "id", std::string());
        const std::string action  = json_value(body, "action", std::string());
        if (cmpl_id.empty()) {
            res->error(format_error_response("missing completion id", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (action != "reasoning_end") {
            res->error(format_error_response("unknown control action", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_CONTROL);
            task.id              = rd.get_new_id();
            task.params.control_cmpl_id = cmpl_id;
            task.params.control_action  = SERVER_CONTROL_ACTION_REASONING_END;
            rd.post_task(std::move(task));
        }

        auto result = rd.next(req.should_stop);
        if (!result) {
            GGML_ASSERT(req.should_stop());
            return res;
        }
        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }
        res->ok(result->to_json());
        return res;
    };

    this->post_responses_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = server_chat_convert_responses_to_chatcmpl(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: OpenAI Responses -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        common_chat_session session;
        json body_parsed = oaicompat_chat_params_parse(
            ctx_server.vocab,
            body,
            meta->chat_params,
            files,
            session);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_RESP,
            session);
    };

    this->post_responses_tok_oai = [this](const server_http_req & req) {
        return handle_count_tokens(req, TASK_RESPONSE_TYPE_OAI_RESP);
    };

    this->post_transcriptions_oai = [this](const server_http_req & req) {
        auto res = create_response();

        if (!meta->has_mtmd || !meta->chat_params.allow_audio) {
            res->error(format_error_response("The current model does not support audio input.", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::vector<raw_buffer> files;
        json body = convert_transcriptions_to_chatcmpl(
            json::parse(req.body),
            meta->chat_params.tmpls.get(),
            req.files,
            files);
        SRV_DBG("%s\n", "Request converted: OpenAI Transcriptions -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        common_chat_session session;
        json body_parsed = oaicompat_chat_params_parse(
            ctx_server.vocab,
            body,
            meta->chat_params,
            files,
            session);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_ASR,
            session);
    };

    // realtime transcription over websocket, modeled after the OpenAI realtime API
    // audio is PCM16 mono at the model sample rate, sent as base64 in input_audio_buffer.append events
    // input_audio_buffer.commit with "final": true ends the input, closing the socket also ends it
    this->ws_realtime = [this](const server_http_req & req, server_http_context::ws_conn & conn) {
        auto send_event = [&conn](json ev) {
            ev["event_id"] = "event_" + random_string();
            conn.send(safe_json_to_str(ev));
        };
        auto send_error_event = [&send_event](const std::string & message, const std::string & type) {
            send_event(json {
                {"type", "error"},
                {"error", {
                    {"type",    type},
                    {"code",    nullptr},
                    {"message", message},
                    {"param",   nullptr},
                }},
            });
        };

        // models that answer after the audio hold a conversation, see server-realtime.h
        // live transcription: Voxtral Realtime (no chat), or an "asr" model in a session opened with ?intent=transcription
        const bool live_asr = !ctx_server.asr_prompt.empty() && req.get_param("intent") == "transcription";
        if (ctx_server.mtmd_cap.audio_stream && ctx_server.mtmd_cap.chat && !live_asr) {
            server_realtime_conversation({queue_tasks, queue_results, *meta, params, ctx_server.mctx, ctx_server.mtmd_cap, ctx_server.vocab}, conn);
            return;
        }

        if (!meta->has_audio_stream) {
            send_error_event("The current model does not support realtime transcription.", "invalid_request_error");
            conn.close();
            return;
        }

        const std::string session_id = "sess_" + random_string();
        const int model_rate = mtmd_get_audio_sample_rate(ctx_server.mctx);
        json session_config = json {
            {"id",     session_id},
            {"object", "realtime.transcription_session"},
            {"type",   "transcription"},
            {"audio",  {{"input", {
                {"format",          {{"type", "audio/pcm"}, {"rate", model_rate}}},
                {"transcription",   {{"model", meta->model_name}}},
                {"turn_detection",  nullptr},
                {"noise_reduction", nullptr},
            }}}},
        };

        auto res = create_response();
        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_TRANSCRIBE_RT);
            task.id = rd.get_new_id();
            task.params.sampling          = params.sampling;
            task.params.stream            = true;
            task.params.oaicompat_cmpl_id = session_id;
            task.params.oaicompat_model   = meta->model_name;
            rd.post_task(std::move(task));
        }

        // the slot answers once its audio stream is live, the session starts there
        {
            auto result = rd.next([&conn]() { return !conn.is_open(); });
            if (!result || result->is_error()) {
                if (result) {
                    send_error_event(json_value(result->to_json(), "message", std::string("server error")), "server_error");
                }
                conn.close();
                return;
            }
        }
        send_event(json {{"type", "session.created"}, {"session", session_config}});

        // one item per committed audio buffer, deltas are attributed to the current item
        std::mutex  mutex_item;
        std::string item_id = "item_" + random_string();
        std::string item_prev; // the committed item before item_id, empty for the first one
        std::string item_text;

        // called with mutex_item held
        auto send_completed = [&]() {
            send_event(json {
                {"type",          "conversation.item.input_audio_transcription.completed"},
                {"item_id",       item_id},
                {"content_index", 0},
                {"transcript",    item_text},
            });
            item_text.clear();
        };

        auto post_control = [&](server_control_action action, std::vector<float> && audio) {
            server_response_reader ctl(queue_tasks, queue_results, HTTP_POLLING_SECONDS);
            server_task task(SERVER_TASK_TYPE_CONTROL);
            task.id                     = ctl.get_new_id();
            task.params.control_action  = action;
            task.params.control_cmpl_id = session_id;
            task.params.control_audio   = std::move(audio);
            ctl.post_task(std::move(task));
            auto result = ctl.next([&conn]() { return !conn.is_open(); });
            auto * control = result ? dynamic_cast<server_task_result_control *>(result.get()) : nullptr;
            return control && control->success;
        };

        // client events are read on this thread, the transcript is sent from the thread below
        std::atomic<bool> input_ended{false};
        std::thread reader([&]() {
            int                        input_rate = model_rate;
            mtmd_helper::resampler_ptr resampler; // input_rate -> model_rate
            std::string msg;
            while (conn.read(msg) != 0) {
                json ev;
                try {
                    ev = json::parse(msg);
                } catch (const std::exception &) {
                    send_error_event("invalid JSON", "invalid_request_error");
                    continue;
                }
                const std::string type = json_value(ev, "type", std::string());
                if (type == "session.update" || type == "transcription_session.update") {
                    const json session = json_value(ev, "session", json::object());
                    const json audio   = session.is_object() ? session.value("audio", json::object()) : json::object();
                    const json in      = audio.is_object() ? audio.value("input", json::object()) : json::object();
                    if (in.is_object() && in.contains("format") && in["format"].is_object()) {
                        const int rate = json_value(in["format"], "rate", input_rate);
                        if (rate < 8000 || rate > 192000) {
                            send_error_event("session.update: the input audio rate must be between 8000 and 192000 Hz", "invalid_request_error");
                            continue;
                        }
                        if (rate != input_rate) {
                            input_rate = rate;
                            resampler.reset();
                            session_config["audio"]["input"]["format"]["rate"] = rate;
                        }
                    }
                    send_event(json {{"type", "session.updated"}, {"session", session_config}});
                } else if (type == "input_audio_buffer.append") {
                    std::string pcm;
                    try {
                        pcm = base64::decode(json_value(ev, "audio", std::string()));
                    } catch (const std::exception &) {
                        send_error_event("invalid base64 audio", "invalid_request_error");
                        continue;
                    }
                    std::vector<float> samples(pcm.size() / 2);
                    for (size_t i = 0; i < samples.size(); i++) {
                        int16_t v;
                        memcpy(&v, pcm.data() + 2 * i, sizeof(v));
                        samples[i] = v / 32768.0f;
                    }
                    if (!resampler) {
                        resampler.reset(mtmd_helper_resampler_init(input_rate, model_rate));
                        if (!resampler) {
                            send_error_event("failed to initialize the audio resampler", "server_error");
                            break;
                        }
                    }
                    const float * out = nullptr;
                    const int64_t n   = mtmd_helper_resampler_process(resampler.get(), samples.data(), samples.size(), &out);
                    if (n < 0) {
                        send_error_event("failed to resample the audio", "server_error");
                        break;
                    }
                    if (n == 0) {
                        continue;
                    }
                    if (out != samples.data()) {
                        samples.assign(out, out + n);
                    }
                    if (!post_control(SERVER_CONTROL_ACTION_AUDIO_APPEND, std::move(samples))) {
                        break;
                    }
                } else if (type == "input_audio_buffer.commit") {
                    // "final": true ends the input, the last item is completed when the stream drains
                    if (json_value(ev, "final", false)) {
                        input_ended = true;
                        post_control(SERVER_CONTROL_ACTION_AUDIO_END, {});
                        continue;
                    }
                    if (!post_control(SERVER_CONTROL_ACTION_AUDIO_COMMIT, {})) {
                        break;
                    }
                    // completed is sent once the rows of the committed audio are decoded (audio_commit_drained below)
                    std::lock_guard<std::mutex> lock(mutex_item);
                    send_event(json {
                        {"type",             "input_audio_buffer.committed"},
                        {"previous_item_id", item_prev.empty() ? json(nullptr) : json(item_prev)},
                        {"item_id",          item_id},
                    });
                } else if (type == "input_audio_buffer.clear") {
                    send_event(json {{"type", "input_audio_buffer.cleared"}});
                } else {
                    send_error_event("unknown event type: " + type, "invalid_request_error");
                }
            }
            if (!input_ended.exchange(true)) {
                post_control(SERVER_CONTROL_ACTION_AUDIO_END, {});
            }
            // the result loop waits on the socket, nobody reads it from here on
            conn.close();
        });

        while (true) {
            auto result = rd.next([&conn]() { return !conn.is_open(); });
            if (!result) {
                break; // connection closed
            }
            if (result->is_error()) {
                send_error_event(json_value(result->to_json(), "message", std::string("server error")), "server_error");
                break;
            }
            if (auto * partial = dynamic_cast<server_task_result_cmpl_partial *>(result.get())) {
                if (partial->audio_commit_drained) {
                    // complete the committed item and open the next one
                    std::lock_guard<std::mutex> lock(mutex_item);
                    send_completed();
                    item_prev = item_id;
                    item_id   = "item_" + random_string();
                } else if (!partial->content.text.empty()) {
                    std::lock_guard<std::mutex> lock(mutex_item);
                    item_text += partial->content.text;
                    send_event(json {
                        {"type",          "conversation.item.input_audio_transcription.delta"},
                        {"item_id",       item_id},
                        {"content_index", 0},
                        {"delta",         partial->content.text},
                    });
                }
            } else if (result->is_stop()) {
                std::lock_guard<std::mutex> lock(mutex_item);
                send_completed();
                break;
            }
        }

        input_ended = true;
        conn.close();
        reader.join();
    };

    this->post_anthropic_messages = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = server_chat_convert_anthropic_to_oai(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: Anthropic -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        common_chat_session session;
        json body_parsed = oaicompat_chat_params_parse(
            ctx_server.vocab,
            body,
            meta->chat_params,
            files,
            session);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_ANTHROPIC,
            session);
    };

    this->post_anthropic_count_tokens = [this](const server_http_req & req) {
        return handle_count_tokens(req, TASK_RESPONSE_TYPE_ANTHROPIC);
    };

    // same with handle_chat_completions, but without inference part
    this->post_apply_template = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy, unused
        common_chat_session session;   // dummy, unused
        json body = json::parse(req.body);
        json data = oaicompat_chat_params_parse(
            ctx_server.vocab,
            body,
            meta->chat_params,
            files,
            session);
        res->ok({{ "prompt", std::move(data.at("prompt")) }});
        return res;
    };

    this->get_models = [this](const server_http_req &) {
        auto res = create_response(true);
        // note: do NOT use ctx_server here, this endpoint must be accessible during sleep
        if (queue_tasks.is_sleeping()) {
            std::unique_lock<std::mutex> lock(mutex_cache);
            res->ok(cached_models);
        } else {
            res->ok(get_res_models(*meta));
        }
        return res;
    };

    this->post_tokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        json tokens_response = json::array();
        if (body.count("content") != 0) {
            const bool add_special = json_value(body, "add_special", false);
            const bool parse_special = json_value(body, "parse_special", true);
            const bool with_pieces = json_value(body, "with_pieces", false);

            llama_tokens tokens = tokenize_mixed(ctx_server.vocab, body.at("content"), add_special, parse_special);

            if (with_pieces) {
                for (const auto& token : tokens) {
                    std::string piece = common_token_to_piece(ctx_server.vocab, token);
                    json piece_json;

                    // Check if the piece is valid UTF-8
                    if (is_valid_utf8(piece)) {
                        piece_json = piece;
                    } else {
                        // If not valid UTF-8, store as array of byte values
                        piece_json = json::array();
                        for (unsigned char c : piece) {
                            piece_json.push_back(static_cast<int>(c));
                        }
                    }

                    tokens_response.push_back({
                        {"id", token},
                        {"piece", piece_json}
                    });
                }
            } else {
                tokens_response = tokens;
            }
        }

        res->ok(json{{"tokens", std::move(tokens_response)}});
        return res;
    };

    this->post_detokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        std::string content;
        if (body.count("tokens") != 0) {
            const llama_tokens tokens = body.at("tokens").get<llama_tokens>();
            content = tokens_to_str(ctx_server.vocab, tokens);
        }

        res->ok(json{{"content", std::move(content)}});
        return res;
    };

    this->post_embeddings = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_NONE);
    };

    this->post_embeddings_oai = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_OAI_EMBD);
    };

    this->post_rerank = [this](const server_http_req & req) {
        return handle_rerank_impl(req);
    };

    this->post_tts = [this](const server_http_req & req) {
        return handle_tts_impl(req, parse_tts_body(req));
    };

    this->post_speech_oai = [this](const server_http_req & req) {
        json body = convert_speech_to_tts(parse_tts_body(req));
        SRV_DBG("%s\n", "Request converted: OpenAI Speech -> TTS");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        return handle_tts_impl(req, std::move(body));
    };

    this->post_systemone = [this](const server_http_req & req) {
        auto res = create_response();
        const auto & decision = ctx_server.decision;
        if (decision.type == COMMON_DECISION_TYPE_NONE) {
            res->error(format_error_response("This model is not a decision model", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        const json body = json::parse(req.body);
        const auto questions = decision.parse_questions(body);

        std::vector<raw_buffer> files;
        const json state = decision.prepare_state(decision.parse_state(body, files));
        if (!files.empty() && (!decision.can_use_images() || !meta->has_inp_image)) {
            res->error(format_error_response("This server does not support image input for decisions. For a model that supports it, start it with `--mmproj`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // one task per variant of each question, or one task for all the questions
        auto & rd = res->rd;
        {
            std::vector<server_task> tasks;
            if (decision.is_joint()) {
                server_task task = server_task(SERVER_TASK_TYPE_DECISION);
                task.id = rd.get_new_id();
                decision.fill_task_joint(state, questions, files, ctx_server.mctx, ctx_server.init_opt, task);
                tasks.push_back(std::move(task));
            } else {
                for (const auto & question : questions) {
                    for (size_t variant = 0; variant < decision.n_variants(question); variant++) {
                        server_task task = server_task(SERVER_TASK_TYPE_DECISION);
                        task.id = rd.get_new_id();
                        decision.fill_task(state, questions, question, variant, files, ctx_server.mctx, ctx_server.init_opt, task);
                        tasks.push_back(std::move(task));
                    }
                }
            }
            if (decision.can_share_prompt()) {
                tasks = server_decision_group_tasks(std::move(tasks), params.n_parallel);
            }
            rd.post_tasks(std::move(tasks));
        }

        auto all_results = rd.wait_for_all(req.should_stop);

        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        }

        json answers = json::object();
        int32_t n_tokens = 0;
        size_t i_result = 0;
        size_t i_score  = 0;
        for (const auto & question : questions) {
            std::vector<std::vector<float>> scores;
            if (decision.is_joint()) {
                // one result with the scores of all the questions, in order
                auto * result = dynamic_cast<server_task_result_decision *>(all_results.results[0].get());
                GGML_ASSERT(result != nullptr && i_score + question.options.size() <= result->scores.size());
                scores.emplace_back(result->scores.begin() + i_score, result->scores.begin() + i_score + question.options.size());
                i_score += question.options.size();
                n_tokens = result->n_tokens;
            }
            for (size_t variant = 0; !decision.is_joint() && variant < decision.n_variants(question); variant++) {
                auto * result = dynamic_cast<server_task_result_decision *>(all_results.results[i_result++].get());
                GGML_ASSERT(result != nullptr);
                scores.push_back(result->scores);
                n_tokens += result->n_tokens;
            }
            answers[question.id] = decision.format_answer(question, scores, !files.empty());
        }

        res->ok(json{
            {"model",   meta->model_name},
            {"answers", answers},
            {"usage",   {
                {"input_tokens",  n_tokens},
                {"output_tokens", 0},
            }},
        });
        return res;
    };

    this->get_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_GET_LORA);
            task.id = rd.get_new_id();
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_get_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };

    this->post_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        if (!body.is_array()) {
            res->error(format_error_response("Request body must be an array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_SET_LORA);
            task.id = rd.get_new_id();
            task.set_lora = parse_lora_request(body);
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_apply_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_save(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_SAVE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_restore(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_RESTORE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_save_load*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_erase(const server_http_req & req, int id_slot) {
    auto res = create_response();
    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_ERASE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot = id_slot;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_erase*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_embeddings_impl(const server_http_req & req, task_response_type res_type) {
    auto res = create_response();
    if (!params.embedding) {
        res->error(format_error_response("This server does not support embeddings. Start it with `--embeddings`", ERROR_TYPE_NOT_SUPPORTED));
        return res;
    }

    if (res_type != TASK_RESPONSE_TYPE_NONE && meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
        res->error(format_error_response("Pooling type 'none' is not OAI compatible. Please use a different pooling type", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    const json body = json::parse(req.body);

    // for the shape of input/content, see tokenize_input_prompts()
    json prompt;
    if (body.count("input") != 0) {
        prompt = body.at("input");
    } else if (body.contains("content")) {
        res_type = TASK_RESPONSE_TYPE_NONE; // "content" field is not OAI compatible
        prompt = body.at("content");
    } else {
        res->error(format_error_response("\"input\" or \"content\" must be provided", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool use_base64 = false;
    if (body.count("encoding_format") != 0) {
        const std::string & format = body.at("encoding_format");
        if (format == "base64") {
            use_base64 = true;
        } else if (format != "float") {
            res->error(format_error_response("The format to return the embeddings in. Can be either float or base64", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    // same shapes as tokenize_input_prompts(), plus OAI content: { "content": [ { "type": "text"|"image_url"|"input_audio"|"input_video", ... } ] }
    auto tokenize_entry = [&](const json & p) {
        if (p.is_object() && p.contains("content")) {
            return tokenize_oai_content_array(ctx_server.vocab, ctx_server.mctx, meta->chat_params, p.at("content"), true, true, ctx_server.init_opt);
        }
        return tokenize_input_subprompt(ctx_server.vocab, ctx_server.mctx, p, true, true, ctx_server.init_opt);
    };

    // dispatch to the appropriate tokenizer based on the shape of the input
    std::vector<server_tokens> tokenized_prompts;
    if (prompt.is_array() && !json_is_array_and_contains_numbers(prompt)) {
        for (const auto & p : prompt) {
            tokenized_prompts.push_back(tokenize_entry(p));
        }
    } else {
        tokenized_prompts.push_back(tokenize_entry(prompt));
    }
    if (tokenized_prompts.empty()) {
        res->error(format_error_response("\"input\" must not be empty", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    for (const auto & tokens : tokenized_prompts) {
        // this check is necessary for models that do not add BOS token to the input
        if (tokens.empty()) {
            res->error(format_error_response("Input content cannot be empty", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    int embd_normalize = params.embd_normalize;
    if (body.count("embd_normalize") != 0) {
        embd_normalize = body.at("embd_normalize").get<int>();
        if (meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
            SRV_DBG("embd_normalize is not supported by pooling type %d, ignoring it\n", meta->pooling_type);
        }
    }

    // create and queue the task
    json responses = json::array();
    auto & rd = res->rd;
    {
        std::vector<server_task> tasks;
        for (size_t i = 0; i < tokenized_prompts.size(); i++) {
            server_task task = server_task(SERVER_TASK_TYPE_EMBEDDING);

            task.id     = rd.get_new_id();
            task.tokens = std::move(tokenized_prompts[i]);

            // OAI-compat
            task.params.res_type = res_type;
            task.params.embd_normalize = embd_normalize;

            tasks.push_back(std::move(task));
        }
        rd.post_tasks(std::move(tasks));
    }

    // wait for the results
    auto all_results = rd.wait_for_all(req.should_stop);

    // collect results
    if (all_results.is_terminated) {
        return res; // connection is closed
    } else if (all_results.error) {
        res->error(all_results.error->to_json());
        return res;
    } else {
        for (auto & res : all_results.results) {
            GGML_ASSERT(dynamic_cast<server_task_result_embd*>(res.get()) != nullptr);
            responses.push_back(res->to_json());
        }
    }

    // write JSON response
    json root = res_type == TASK_RESPONSE_TYPE_OAI_EMBD
        ? format_embeddings_response_oaicompat(body, meta->model_name, responses, use_base64)
        : json(responses);
    res->ok(root);
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_rerank_impl(const server_http_req & req) {
    auto res = create_response();
    if (!params.embedding || params.pooling_type != LLAMA_POOLING_TYPE_RANK) {
        res->error(format_error_response("This server does not support reranking. Start it with `--reranking`", ERROR_TYPE_NOT_SUPPORTED));
        return res;
    }

    const json body = json::parse(req.body);

    // if true, use TEI API format, otherwise use Jina API format
    // Jina: https://jina.ai/reranker/
    // TEI: https://huggingface.github.io/text-embeddings-inference/#/Text%20Embeddings%20Inference/rerank
    bool is_tei_format = body.contains("texts");

    // expand one rerank input item: plain string or {"text": str|[str], "image": url|data-uri|raw-base64|[urls]}
    auto parse_item = [&](const json & item, std::string & out_text, std::vector<raw_buffer> & out_files) {
        if (item.is_string()) {
            out_text = item.get<std::string>();
            return;
        }
        if (!item.is_object()) {
            throw std::invalid_argument("rerank input must be a string or an object of the form {\"text\": ..., \"image\": ...}");
        }
        if (item.contains("video") || item.contains("fps") || item.contains("max_frames")) {
            throw std::invalid_argument("video/fps/max_frames are not supported for rerank input");
        }
        json text = item.value("text", json());
        if (text.is_string()) {
            out_text += text.get<std::string>();
        } else if (text.is_array()) {
            for (const auto & t : text) {
                if (!t.is_string()) {
                    throw std::invalid_argument("\"text\" array elements must be strings");
                }
                out_text += t.get<std::string>();
            }
        } else if (!text.is_null()) {
            throw std::invalid_argument("\"text\" must be a string or an array of strings");
        }
        json image = item.value("image", json());
        auto add_image = [&](const std::string & url) {
            // marker is appended by format_prompt_rerank, one per decoded file
            try {
                handle_media(out_files, url, params.media_path);
            } catch (const std::runtime_error & e) {
                throw std::invalid_argument(e.what());
            }
        };
        if (image.is_string()) {
            add_image(image.get<std::string>());
        } else if (image.is_array()) {
            for (const auto & u : image) {
                if (!u.is_string()) {
                    throw std::invalid_argument("\"image\" array elements must be strings");
                }
                add_image(u.get<std::string>());
            }
        } else if (!image.is_null()) {
            throw std::invalid_argument("\"image\" must be a string or an array of strings");
        }
    };

    // Parse query
    std::string query_str;
    std::vector<raw_buffer> query_files;
    if (body.count("query") == 1) {
        try {
            parse_item(body.at("query"), query_str, query_files);
        } catch (const std::invalid_argument & e) {
            res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    } else {
        res->error(format_error_response("\"query\" must be provided", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    // Resolve Jina/TEI format ambiguity
    json documents_json = json_value(body, "documents",
        json_value(body, "texts", json::array()));
    if (!documents_json.is_array() || documents_json.empty()) {
        res->error(format_error_response("\"documents\" must be a non-empty array of strings or objects", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    // Parse documents
    std::vector<std::string>          documents(documents_json.size());
    std::vector<std::vector<raw_buffer>> document_files(documents_json.size());
    for (size_t i = 0; i < documents_json.size(); i++) {
        try {
            parse_item(documents_json[i], documents[i], document_files[i]);
        } catch (const std::invalid_argument & e) {
            res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    int top_n = json_value(body, "top_n", (int)documents.size());
    if (top_n < 0) {
        res->error(format_error_response("\"top_n\" must be >= 0", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    // create and queue the task
    json responses = json::array();
    auto & rd = res->rd;
    {
        std::vector<server_task> tasks;
        tasks.reserve(documents.size());
        for (size_t i = 0; i < documents.size(); i++) {
            auto tmp = format_prompt_rerank(ctx_server.model_tgt, ctx_server.vocab, ctx_server.mctx, query_str, documents[i], query_files, document_files[i], ctx_server.init_opt);
            server_task task = server_task(SERVER_TASK_TYPE_RERANK);
            task.id     = rd.get_new_id();
            task.tokens = std::move(tmp);
            tasks.push_back(std::move(task));
        }
        rd.post_tasks(std::move(tasks));
    }

    // wait for the results
    auto all_results = rd.wait_for_all(req.should_stop);

    // collect results
    if (all_results.is_terminated) {
        return res; // connection is closed
    } else if (all_results.error) {
        res->error(all_results.error->to_json());
        return res;
    } else {
        for (auto & res : all_results.results) {
            GGML_ASSERT(dynamic_cast<server_task_result_rerank*>(res.get()) != nullptr);
            responses.push_back(res->to_json());
        }
    }

    // write JSON response
    json root = format_response_rerank(
        body,
        meta->model_name,
        responses,
        is_tei_format,
        documents,
        top_n);

    res->ok(root);
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_tts_impl(const server_http_req & req, json body) {
    auto res = create_response();
    res->set_req(&req); // will also set spipe if needed

    if (!ctx_server.has_cap_tts()) {
        res->error(format_error_response("this server does not support audio generation", ERROR_TYPE_NOT_SUPPORTED));
        return res;
    }

    const auto info = mtmd_gen_audio_get_info(ctx_server.mctx);

    std::string prompt = json_value(body, "input", json_value(body, "prompt", std::string()));
    if (prompt.empty()) {
        res->error(format_error_response("\"input\" must be a non-empty string", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    const std::string response_format = json_value(body, "response_format", std::string("wav"));
    const bool pcm16 = response_format == "pcm16";

    server_task task(SERVER_TASK_TYPE_TTS);
    server_schema::eval_tts_schema(params, task, body);
    const bool stream = task.params.stream;
    // a model with a "tts" chat template (e.g. LFM3-Audio) speaks the prompt rendered from it
    const std::string lang       = json_value(body, "lang", params.tts_lang);
    // speaker_ref in the request overrides the default voice
    const bool has_speaker_ref   = req.files.count("speaker_ref") > 0 || !json_value(body, "speaker_ref_b64", std::string()).empty();
    const std::string voice      = json_value(body, "voice", has_speaker_ref ? std::string() : params.tts_voice);
    const std::string tts_prompt = common_chat_get_tts_prompt(meta->chat_params.tmpls.get(), prompt, lang, voice);
    // a model with predefined voices only speaks with one of them
    mtmd::bitmap_ptr voice_ref;
    if (!voice.empty() && info.n_voices > 0) {
        voice_ref.reset(mtmd_gen_audio_get_voice(ctx_server.mctx, voice.c_str()));
        if (!voice_ref) {
            res->error(format_error_response("unknown voice \"" + voice + "\", see \"voices\" in GET /props", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }
    task.tts_inp.set_prompt(tts_prompt.empty() ? prompt : tts_prompt);
    task.tts_inp.set_lang(lang);
    task.tts_inp.data.stream    = stream;
    task.tts_inp.data.out_type  = response_format == "pcm" || pcm16
        ? MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM
        : MTMD_HELPER_GEN_AUDIO_OUTTYPE_WAV;
    // note: -1 is clamped to 0 by the sampler, it will disable the penalty
    task.params.sampling.penalty_last_n = task.params.n_predict > 0 ? task.params.n_predict : 512;
    task.params.sampling.seed = task.tts_inp.data.seed; // same seed for the codec/vocoder RNG
    if (task.tts_inp.data.top_k > 0) {
        task.params.sampling.top_k = task.tts_inp.data.top_k;
    }
    if (task.tts_inp.data.top_p > 0) {
        task.params.sampling.top_p = task.tts_inp.data.top_p;
    }
    if (task.tts_inp.data.temp >= 0) {
        task.params.sampling.temp = task.tts_inp.data.temp;
    }

    // speaker reference: either an uploaded form file ("speaker_ref") or a base64 JSON field ("speaker_ref_b64")
    const unsigned char * speaker_ref_data = nullptr;
    size_t speaker_ref_len  = 0;
    std::string speaker_ref_b64_decoded;

    auto speaker_ref_file = req.files.find("speaker_ref");
    if (speaker_ref_file != req.files.end()) {
        speaker_ref_data = speaker_ref_file->second.data.data();
        speaker_ref_len  = speaker_ref_file->second.data.size();
    } else {
        std::string speaker_ref_b64 = json_value(body, "speaker_ref_b64", std::string());
        if (!speaker_ref_b64.empty()) {
            try {
                speaker_ref_b64_decoded = base64::decode(speaker_ref_b64);
            } catch (const std::exception &) {
                res->error(format_error_response("\"speaker_ref_b64\" is not valid base64", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            speaker_ref_data = (const unsigned char *) speaker_ref_b64_decoded.data();
            speaker_ref_len  = speaker_ref_b64_decoded.size();
        }
    }

    if (speaker_ref_len > 0) {
        if (voice_ref) {
            res->error(format_error_response("\"voice\" and \"speaker_ref\" cannot be used together", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (!mtmd_support_audio(ctx_server.mctx)) {
            res->error(format_error_response("this model has no voice encoder, \"speaker_ref\" cannot be used", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        auto wrapper = mtmd_helper_bitmap_init_from_buf(ctx_server.mctx, speaker_ref_data, speaker_ref_len, false, ctx_server.init_opt);
        if (!wrapper.bitmap) {
            res->error(format_error_response("failed to decode \"speaker_ref\"", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        task.tts_inp.set_speaker_ref(mtmd::bitmap_ptr(wrapper.bitmap));
    } else if (voice_ref) {
        task.tts_inp.set_speaker_ref(std::move(voice_ref));
    } else if (info.n_voices > 0) {
        // without a voice prompt the model produces near silence: the first voice is the default
        const char * first = mtmd_gen_audio_voice_name(ctx_server.mctx, 0);
        SRV_WRN("no voice given, speaking with \"%s\"\n", first);
        task.tts_inp.set_speaker_ref(mtmd::bitmap_ptr(mtmd_gen_audio_get_voice(ctx_server.mctx, first)));
    } else {
        SRV_WRN("%s", "no speaker reference provided, the model may behave randomly\n");
    }

    auto & rd = res->rd;
    task.id = rd.get_new_id();
    rd.post_task(std::move(task));

    // raw LE mono samples; audio/L16 would imply 16-bit big-endian (RFC 2586)
    const std::string pcm_type = "audio/pcm;rate=" + std::to_string(info.sample_rate);
    const std::string content_type = pcm16 ? pcm_type + ";encoding=signed-int;bits=16"
        : response_format == "pcm" ? pcm_type + ";encoding=float;bits=32"
        : "audio/wav";

    // the pipeline outputs float32 samples
    auto take_audio = [pcm16](server_task_result_tts * tts_res) {
        const std::string & audio = tts_res->audio;
        return pcm16 ? pcm_f32_to_s16((const float *) audio.data(), audio.size() / sizeof(float)) : std::move(tts_res->audio);
    };

    if (!stream) {
        auto result = rd.next(req.should_stop);
        if (!result) {
            GGML_ASSERT(req.should_stop());
            return res; // connection is closed
        }
        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }
        auto * tts_res = dynamic_cast<server_task_result_tts *>(result.get());
        GGML_ASSERT(tts_res != nullptr);
        res->status       = 200;
        res->content_type = content_type;
        res->data         = take_audio(tts_res);
        return res;
    } else {
        auto first_result = rd.next(req.should_stop);
        if (!first_result) {
            GGML_ASSERT(req.should_stop());
            return res; // connection is closed
        }
        if (first_result->is_error()) {
            res->error(first_result->to_json());
            return res;
        }
        auto * first_tts_res = dynamic_cast<server_task_result_tts *>(first_result.get());
        GGML_ASSERT(first_tts_res != nullptr);

        res->status       = 200;
        res->content_type = content_type;
        res->data          = take_audio(first_tts_res);
        bool is_done       = first_tts_res->final;

        res->set_next([res_this = res.get(), is_done, take_audio](std::string & output) mutable -> bool {
            // flush buffered audio before is_done - the first result can already be final
            if (!res_this->data.empty()) {
                output = std::move(res_this->data);
                res_this->data.clear();
                return true;
            }
            if (is_done) {
                return false;
            }
            if (res_this->should_stop()) {
                return false;
            }

            server_response_reader & rd = res_this->rd;
            if (!rd.has_next()) {
                return false;
            }
            auto result = rd.next([&res_this]() { return res_this->should_stop(); });
            if (!result || result->is_error()) {
                return false;
            }
            auto * tts_res = dynamic_cast<server_task_result_tts *>(result.get());
            GGML_ASSERT(tts_res != nullptr);
            output  = take_audio(tts_res);
            is_done = tts_res->final;
            return true;
        });
    }

    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_count_tokens(const server_http_req & req, task_response_type res_type) {
    auto res = create_response();
    std::vector<raw_buffer> files;
    json body = json::parse(req.body);
    bool is_oai = false;

    switch (res_type) {
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            {
                is_oai = true;
            } break;
        case TASK_RESPONSE_TYPE_OAI_RESP:
            {
                is_oai = true;
                body = server_chat_convert_responses_to_chatcmpl(body);
            } break;
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            {
                body = server_chat_convert_anthropic_to_oai(body);
            } break;
        default:
            res->error(format_error_response("invalid res_type", ERROR_TYPE_INVALID_REQUEST));
            return res;
    }

    common_chat_session session; // dummy, unused
    json body_parsed = oaicompat_chat_params_parse(
            ctx_server.vocab,
            body,
            meta->chat_params,
            files,
            session);
    json prompt = body_parsed.at("prompt");
    // SRV_DBG("prompt = %s\n", prompt.dump().c_str());

    // TODO @ngxson : refactor this code block, move this to server-common and reuse it in other places
    size_t n_tokens;
    if (ctx_server.mctx != nullptr) {
        if (!prompt.is_string()) {
            throw std::runtime_error("for mtmd, input prompt must be a string.");
        }
        n_tokens = process_mtmd_prompt(ctx_server.mctx, prompt.get<std::string>(), files, ctx_server.init_opt, true).size();
    } else {
        n_tokens = tokenize_mixed(ctx_server.vocab, prompt, true, true).size();
    }

    json response = {{"input_tokens", static_cast<int64_t>(n_tokens)}};
    if (is_oai) {
        response["object"] = "response.input_tokens";
    }
    res->ok(response);
    return res;
}

void server_routes::update_cached_responses(bool is_sleeping) {
    // caller is task_queue, so ctx_server can be accessed without holding locks
    std::unique_lock<std::mutex> lock(mutex_cache);

    if (is_sleeping) {
        cached_models  = get_res_models(*meta);
        cached_props   = get_res_props(*meta, params, true);
        cached_metrics = ctx_server.get_metrics();

        should_reset_buckets = false;

        SRV_DBG("%s\n", "cached responses updated");

    } else if (should_reset_buckets) {
        // a scrape during sleep already reported these buckets
        ctx_server.reset_metrics_bucket();

        should_reset_buckets = false;
    }
}
