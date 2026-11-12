#pragma once

#include "llama.h"
#include "llama-ext.h"
#include "llama-cparams.h"
#include "llama-graph.h"
#include "llama-adapter.h"
#include "llama-impl.h"
#include "llama-memory.h"
#include "llama-split-remote.h"
#include "llama-moe-cache.h"

#include "ggml-cpp.h"
#include "ggml-opt.h"

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

struct llama_model;
class llama_batch_allocr;
class llama_moe_cache;

class llama_io_read_i;
class llama_io_write_i;

// "memory" as in abstract memory for the context
struct llama_memory_i;
struct llama_memory_context_i;

// [QLH-SPLIT] The aggregate memory view over a chain; defined in llama-memory-split.h. Only a
// unique_ptr to it is held here, and the destructor of llama_context lives in the .cpp that
// includes the definition, so a forward declaration is enough.
struct llama_memory_split;

// stores copy of the memory in device buffer. used for fast state save/load
struct llama_memory_buffer {
    int n_tensors = 0;
    size_t total_size = 0;

    ggml_backend_buffer_ptr buf;

    ggml_context_ptr ctx;

    std::vector<ggml_tensor *> org;
    std::vector<ggml_tensor *> cpy;
};

using llama_memory_buffers = std::map<ggml_backend_buffer_type_t, llama_memory_buffer>;

struct llama_context {
    // init scheduler and compute buffers, reserve worst-case graphs
    llama_context(
            const llama_model & model,
                  llama_context_params params);

    ~llama_context();

    // reserve a new backend scheduler (if needed)
    // for example, when:
    //   - changing loras
    //   - changing samplers
    //   - changing attention type
    //   - etc.
    void sched_reserve();

    void synchronize();

    const llama_model   & get_model()   const;
    const llama_cparams & get_cparams() const;

    ggml_backend_sched_t get_sched() const;

    uint32_t n_ctx()     const;
    uint32_t n_ctx_seq() const;
    uint32_t n_batch()   const;
    uint32_t n_ubatch()  const;
    uint32_t n_seq_max() const;

    uint32_t n_threads()       const;
    uint32_t n_threads_batch() const;

    llama_memory_t get_memory() const;

    // return true if the memory was updated
    bool memory_update(bool optimize);

    enum llama_pooling_type pooling_type() const;

    float * get_logits();
    float * get_logits_ith(int32_t i);

    float * get_embeddings();
    float * get_embeddings_ith(int32_t i);
    float * get_embeddings_seq(llama_seq_id seq_id);

    float * get_embeddings_nextn();
    float * get_embeddings_nextn_ith(int32_t i);

    float * get_embeddings_layer_inp(uint32_t lid);

    ggml_tensor * get_layer_inp_dev() const { return layer_inp_dev; }
    ggml_tensor * get_h_nextn_dev()   const { return h_nextn_dev; }

    llama_token * get_sampled_tokens() const;
    llama_token   get_sampled_token_ith(int32_t idx);

    float * get_sampled_logits_ith(int32_t idx);
    size_t  get_sampled_logits_count(int32_t idx);

    float * get_sampled_probs_ith(int32_t idx);
    size_t  get_sampled_probs_count(int32_t idx);

    const llama_token * get_sampled_candidates_ith(int32_t idx);
    size_t get_sampled_candidates_count(int32_t idx);

    bool get_causal_attn() const;

    void attach_threadpool(
            ggml_threadpool_t threadpool,
            ggml_threadpool_t threadpool_batch);

    void detach_threadpool();

    void set_n_threads(int32_t n_threads, int32_t n_threads_batch);

    void set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data);

    void set_embeddings (bool value);
    void set_embeddings_nextn(bool value, bool masked);
    void set_embeddings_layer_inp(uint32_t lid, bool enable);
    void set_embeddings_layer_inp_dev(bool value);

    // whether decode accepts entries whose embd is a row of ctx_other's layer input device buffer
    bool supports_embd_row() const;

    // h_nextn device buffer, see llama_set_embeddings_nextn_dev()
    bool set_embeddings_nextn_dev(bool value);
    bool get_h_nextn_rows(int64_t row0, int64_t n_rows, float * out);
    bool set_h_nextn_rows(int64_t row0, int64_t n_rows, const float * data);

    // state rows read from the h_nextn device buffer of src, see llama_set_embd_state_src()
    bool set_state_src(llama_context * src);
    bool supports_state_row() const;
    void set_nextn_layer_offset(int32_t offset);
    void set_causal_attn(bool value);
    void set_warmup(bool value);

    void set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool set_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end);

    // process a single ubatch with a specific graph type
    // if memory_context is provided, it will be applied first to the context's memory
    // ret contains the status of the graph computation
    // returns nullptr only if ret != GGML_STATUS_SUCCESS
    llm_graph_result * process_ubatch(
                const llama_ubatch & ubatch,
                    llm_graph_type   gtype,
            llama_memory_context_i * mctx,
                       ggml_status & ret);

    int encode(const llama_batch_ext & batch_inp);
    // [QLH-SPLIT] decode() dispatches: with split_stages empty it is exactly decode_plain(),
    // which is the pre-existing whole-model path.
    int decode(const llama_batch_ext & batch_inp);
    int decode_plain(const llama_batch_ext & batch_inp);
    int decode_split(const llama_batch_ext & batch_inp);

    // Borrowed continuation stages; see split_stages.
    int  split_attach(llama_context ** stages, int32_t n);
    int  split_attach_remote(const std::vector<std::string> & endpoints, std::string & err);
    // Both carriers count: a chain can be local, remote, or (never both -- see split_attach)
    // one or the other, and a caller asking how long the chain is means what decode drives.
    int32_t split_n_stages() const { return (int32_t) (split_stages.size() + split_remotes.size()); }

    // [QLH-SPLIT] Whether the chain is still trustworthy, and why not. Defined in the .cpp: the
    // answer lives in llama_memory_split, which is only forward-declared here. See
    // llama_context_split_failed() in llama-ext.h for the contract.
    bool         split_failed()  const;
    const char * split_failure() const;

    // [QLH-SPLIT] v6: the last state restore found a snapshot the stage no longer holds. Not a
    // chain failure -- nothing was applied -- so it is answered separately. See
    // llama_context_split_state_stale() in llama-ext.h.
    bool         split_state_stale() const;

    // compat version
    int encode(const llama_batch & batch_inp);
    int decode(const llama_batch & batch_inp);

    //
    // state save/load
    //

    size_t state_get_size();
    size_t state_get_data(      uint8_t * dst, size_t size);
    size_t state_set_data(const uint8_t * src, size_t size);

    size_t state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags);

    size_t state_seq_get_data(llama_seq_id seq_id,       uint8_t * dst, size_t size, llama_state_seq_flags flags);
    size_t state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags);

    bool state_load_file(
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    bool state_save_file(
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    size_t state_seq_load_file(
          llama_seq_id   seq_id,
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    size_t state_seq_save_file(
          llama_seq_id   seq_id,
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    //
    // perf
    //

    llama_perf_context_data perf_get_data() const;
    void perf_reset();

    llama_memory_breakdown memory_breakdown() const;

    //
    // training
    //

    void opt_init(struct llama_model * model, struct llama_opt_params lopt_params);

    // TODO: more flexible combinations of logical/physical batch size and context size
    void opt_epoch(
            ggml_opt_dataset_t      dataset,
            ggml_opt_result_t       result_train,
            ggml_opt_result_t       result_eval,
            int64_t                 idata_split,
            ggml_opt_epoch_callback callback_train,
            ggml_opt_epoch_callback callback_eval);

    void opt_epoch_iter(
            ggml_opt_dataset_t               dataset,
            ggml_opt_result_t                result,
            const std::vector<llama_token> & tokens,
            const std::vector<llama_token> & labels_sparse,
            llama_batch                    & batch,
            ggml_opt_epoch_callback          callback,
            bool                             train,
            int64_t                          idata_in_loop,
            int64_t                          ndata_in_loop,
            int64_t                          t_loop_start);

private:
    //
    // output
    //

    // Make sure enough space is available for outputs.
    // Returns max number of outputs for which space was reserved.
    uint32_t output_reserve(int32_t n_outputs);

    void output_reorder();

    // map the output row index `i` to batch index
    int64_t output_resolve_row(int32_t i) const;

    // async-copy enabled layer-input tensors (per cparams.output_layer_inp)
    // from backend into host-side embd_layer_inp buffers
    bool extract_layer_inputs(const llm_graph_result * res, size_t token_offset, size_t n_tokens);

    // (re)allocate the device buffer for cparams.embeddings_layer_inp_dev
    void layer_inp_dev_alloc();

    // (re)allocate the device buffer for cparams.embeddings_nextn_dev, false if it does not fit
    bool h_nextn_dev_alloc();

    ggml_backend_t backend_from_dev(ggml_backend_dev_t dev) const;

    //
    // graph
    //

public:
    uint32_t graph_max_nodes() const;

    // can reuse the llm_graph_result instance of the context (for example to update a memory module)
    llm_graph_result * get_gf_res_reserve() const;

    // returns the result of ggml_backend_sched_graph_compute_async execution
    ggml_status graph_compute(ggml_cgraph * gf, bool batched);

    // reserve a graph with a dummy ubatch of the specified size
    ggml_cgraph * graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only = false, size_t * sizes = nullptr);

    bool set_sampler(llama_seq_id seq_id, llama_sampler * sampler);

    // [QLH-SPLIT] v5: the remote candidate channel. These are the three operations llama-ext.h
    // exports; they live here because the state they touch (split_remotes, split_cand_*) is
    // private, and because "does this context have a remote stage" must not be answered by
    // inspecting a member from outside the class.
    bool     split_has_remote() const { return !split_remotes.empty(); }

    // Arms the next remote decode. Safe for a local chain and a whole model: it just clears.
    void     split_request_candidates(const qlh_split::cand_req * req,
                                      const llama_logit_bias * biases, uint32_t n_bias,
                                      const llama_token * hist, uint32_t n_hist);

    // Candidates of the last remote decode; see llama-ext.h for the row contract.
    uint32_t split_take_candidates(const llama_split_token_cand ** cands, int32_t * row) const;

    // [QLH-SPLIT] v7: the per-row verify channel, same shape as the two above. `on` arms the next
    // remote decode; a local chain and a whole model ignore it (there is no wire to spare), which
    // is why the caller does not have to distinguish the chain kinds.
    void        split_request_verify_rows(bool on);
    llama_token split_verified_token(int32_t idx) const;

    // [QLH-SPLIT] v8: the remote MTP draft. Same rule as the channels above -- a local chain and a
    // whole model answer "no" and the caller keeps its own path, so nothing upstream has to know
    // which kind of chain it is driving.
    //
    // `split_mtp_ready()` is a startup property (the tail reported a draft context in OP_HELLO);
    // `split_mtp_draft()` is the per-round call. The latter runs one round trip, not k: the tail
    // runs the recursive draft loop in its own memory, which is the only way the draft can be
    // remote at all (draft i needs draft i-1's h, so the steps cannot be batched).
    bool split_mtp_ready() const;

    // Arms the next remote decode to answer with the h_nextn rows of the rows it answers for.
    // Ignored by a local chain and a whole model, like every other arming call here.
    void     split_request_nextn_rows(bool on);

    // h_nextn rows of the last remote decode, for the caller that feeds the draft. `row0` is the
    // batch row the first returned row belongs to; `n_embd` the width. All are 0 when the last
    // decode did not carry them.
    uint32_t split_take_nextn(const float ** rows, int32_t * row0, uint32_t * n_embd) const;

    // Runs one draft round on the remote tail. Returns the number of drafts written to `out`
    // (0 when the chain cannot draft -- whole model, local chain, or a stage that lost its MTP
    // context) and -1 on a transport failure, which the caller turns into "no drafts" as well.
    int split_mtp_draft(const float * h, int32_t token, int32_t pos, uint32_t k,
                        int32_t seq_id, int32_t * out, uint32_t * n_out);

    // [QLH-SPLIT] v9: the layer-input tap channel. Unlike the three remote channels above this one is
    // *not* armed per frame: which layers a draft model reads is a property of the chain, so the set
    // is sticky and every decode asks for it until it is cleared. That is also what keeps the arming
    // out of the server's per-frame path -- the speculative driver sets it once, from the draft
    // model's own target_layers.
    //
    // Ignored by a local chain and by a whole model (both clear the set): a local chain reads the
    // taps out of the context that owns them, which is the same process.
    void split_set_layer_inp(const uint32_t * ids, uint32_t n);

    // The taps the last remote decode answered: `n` blocks, each `n_rows` rows of `n_embd` floats and
    // the source-layer id it belongs to. Only the ids some remote stage owned are here, so a caller
    // must look its layers up by id rather than by position.
    uint32_t split_take_layer_inp_n() const { return (uint32_t) split_answer.tap.size(); }
    const float * split_take_layer_inp(uint32_t i, uint32_t * id, uint32_t * n_rows, uint32_t * n_embd) const;

private:
    // [QLH-SPLIT] decode_split() is composed of these. They are members only because they reach
    // into the chain (split_stages / split_remotes / split_cand_*); everything that needs nothing
    // but the public context API stays a file-local helper in llama-context.cpp. Splitting it
    // this way is what puts "admit, head, then remote-or-local" on one screen.
    int  split_admit_batch(const llama_batch_ext & batch_inp);
    int  split_self_heal(const llama_batch_ext & batch_inp);
    int  split_drive_remote(const llama_batch_ext & batch_inp, std::vector<float> & carrier,
                            int32_t n_tok, int32_t n_embd);
    int  split_drive_local(const llama_batch_ext & batch_inp, std::vector<float> & carrier,
                           int32_t n_tok, int32_t n_embd);

    // [QLH-SPLIT] Is this context driving a chain? Spelled out once, because three spellings of
    // it (split_stages, split_remotes, split_memory) had grown apart, and the third one is the
    // authoritative one: split_memory is what get_memory() answers with, so a context whose view
    // is gone is a whole-model context however the two lists read.
    //
    // The two attach paths install the view last and detach drops it first, so outside a
    // construction this is exactly "a stage is attached". Inside one they differ for a moment:
    // split_attach() assigns split_stages before it builds the view, and split_attach_remote()
    // grows split_remotes before it builds one. In those two windows split_active() answers false
    // while the list-based spelling (!stages.empty() || !remotes.empty()) answers true. No reader
    // runs there (neither path calls back into decode()/get_logits() while it is assembling a
    // chain), so from outside the two spellings are equivalent -- but this is the one place where
    // "the lists are non-empty" and "the view exists" are not the same statement. Use
    // split_has_remote() for the narrower question.
    bool split_active() const { return split_memory != nullptr; }

    // [QLH-SPLIT] What the head fixes for every stage in a chain: the source depth, the
    // vocabulary, and the one width a hop carries.
    struct head_expect {
        int32_t  n_layer_src = 0;
        uint32_t n_vocab     = 0;
        uint32_t n_embd      = 0;
    };

    // [QLH-SPLIT] The head half of the chain check, shared by split_attach() and
    // split_attach_remote(): a pre-norm exit, the two context flags that select it, a proper
    // prefix of the source model, one hop width. Fills `expect` on success and returns ""; on
    // failure returns the reason (one line, no trailing newline) and leaves `expect` unusable.
    // The two callers only differ in where that line goes (the log vs the `err` out-parameter).
    //
    // `expect` is an out-parameter on purpose: on both attach paths it can only be derived from the
    // head itself, so filling it and checking the head against it belong in the same place, and the
    // caller does not have to build an identity it would only be echoing. Splitting this into
    // split_head_expect(head) + split_check_head(head, expect) would leave the comparison inside
    // checking a value that was just copied from the same head -- true by construction, worth
    // nothing. If the identity ever gets an external source (a config file, a supervisor), this
    // becomes the seam: take `expect` as input and drop the filling.
    static std::string split_check_head(const llama_context & head, head_expect & expect);

    // [QLH-SPLIT] One frame's answer from the remote chain: the five slots and the scalars that
    // describe them, plus the two bit sets that record which channels were asked for and which
    // "the tail cannot answer this" lines have been logged.
    //
    // The contract is "one frame, one answer":
    //   - begin_frame() clears every slot. The remote path of decode_split() calls it once, before
    //     anything can fail, so a frame that ends up returning hidden (or a stage that errors out)
    //     answers null from get_logits_ith() instead of handing back the previous frame's logits.
    //   - store_*() write their own slot and touch nothing else.
    //
    // The old stores called the reset first, which made the order they were called in matter: a
    // store of the h_nextn rows or of the tap blocks before the logits/candidates store had them
    // wiped -- measured, the stage answered 2048 floats and the draft was told there were none.
    // That dependency is gone because clearing is no longer part of storing: the only rule left is
    // "begin_frame() before any store_*()", which the frame-start call already guarantees, so the
    // stores can be written in any order (they still are, in the order the answer arrives).
    struct split_tap_block {
        uint32_t           id = 0;
        std::vector<float> rows;
    };

    struct split_frame_answer {
        // the whole-row answer
        std::vector<float> logits;
        uint32_t           n_vocab    = 0;
        int32_t            logits_row = -1;

        // the candidate answer (sorted by descending logit); empty when the tail answered a row
        std::vector<qlh_split::token_cand> cands;
        int32_t                            cands_row = -1;

        // the per-row verify answer: one token id per row of the frame
        std::vector<int32_t> verify;

        // the h_nextn rows: [n_rows][nextn_embd], the first one belonging to batch row nextn_row0
        std::vector<float> nextn;
        uint32_t           nextn_embd = 0;
        int32_t            nextn_row0 = -1;

        // the layer-input taps: one block per answered source-layer id
        std::vector<split_tap_block> tap;
        uint32_t                     tap_embd = 0;
        uint32_t                     tap_rows = 0;

        // Channels armed for exactly one frame. The request calls set a bit; decode_split() reads
        // and clears them before it can fail, so a failed frame cannot leave the next one asking
        // for something the sampler did not plan for.
        enum : uint32_t {
            ARMED_CANDS  = 1u << 0,
            ARMED_VERIFY = 1u << 1,
            ARMED_NEXTN  = 1u << 2,
        };
        uint32_t armed = 0;

        // Channels whose "the tail cannot answer this" line has already been logged, so an
        // operator can tell "the channel is off" from "this batch was too big" without a line per
        // token. WARNED_TAP is cleared when the tap set is set: it names a property of the chain,
        // not of the frame.
        enum : uint32_t {
            WARNED_VERIFY_ROWS = 1u << 0,
            WARNED_NEXTN       = 1u << 1,
            WARNED_TAP         = 1u << 2,
        };
        uint32_t warned = 0;

        void begin_frame();
        void store_cands(std::vector<qlh_split::token_cand> && c, int32_t row);
        void store_logits(std::vector<float> && l, uint32_t vocab, int32_t row);
    };

    llm_graph_result * get_gf_res_prev();
    llm_graph_result * get_gf_res_decode(uint32_t n_tokens);
    void reset_cached_graphs();

    llm_graph_params graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype) const;

    llm_graph_cb graph_get_cb() const;

    // parameters defining the backend scheduler instance
    struct sched_params {
        size_t max_nodes  = 0;
        bool   parallel   = false;
        bool   op_offload = false;

        // a scheduler can serve any graph of up to max_nodes nodes
        bool can_reuse(const sched_params & other) const {
            return parallel   == other.parallel
                && op_offload == other.op_offload
                && max_nodes  <= other.max_nodes;
        }
    };

    // create a new backend scheduler with the given parameters, unless the current one can be reused
    void sched_new(const sched_params & params);

    // ggml_backend_sched copy callback, copies only the experts used by MUL_MAT_ID and updates the MoE cache
    static bool sched_copy_experts(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph, void * user_data);

    // ggml_backend_sched realloc callback, counts and traces the graph reallocations of this context
    static void sched_graph_realloc(ggml_backend_sched_t sched, ggml_cgraph * gf, bool backend_ids_changed, bool graph_size_changed, void * user_data);

    // disable auto fused ops (Flash Attention, Gated Delta Net) whose op lands on a device
    // that differs from the layer it belongs to (usually due to missing backend support)
    void resolve_fused_ops(const llama_memory_context_i * mctx, uint32_t n_seqs);

    // TODO: read/write lora adapters and cvec
    size_t state_write_data(llama_io_write_i & io);
    size_t state_read_data (llama_io_read_i  & io);

    size_t state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags);
    size_t state_seq_read_data (llama_io_read_i  & io, llama_seq_id seq_id, llama_state_seq_flags flags);

    //
    // members
    //

    const llama_model & model;

    llama_cparams cparams;

    llama_adapter_cvec_ptr  cvec;
    llama_adapter_loras_ptr loras;

    llama_cross cross; // TODO: tmp for handling cross-attention - need something better probably

    llama_memory_ptr memory;
    llama_moe_cache_ptr moe_cache;

    // [QLH-SPLIT] Optional continuation stages. Empty means "this context runs the whole
    // model", which is the only shape that existed before this field and must keep
    // behaving identically. When non-empty, decode() drives this context as the head and
    // hands the hidden states to each stage in turn; the stages are borrowed, not owned.
    // See the split_* comments in llama-context.cpp for the contracts each stage must meet.
    std::vector<llama_context *> split_stages;

    // Non-null iff the chain has stages (local or remote). It is what llama_get_memory()
    // returns, so that llama-server's KV bookkeeping sees the whole chain rather than just the
    // head -- for a remote chain the head is the only stage with a local memory object, which is
    // exactly why the remote ones are dispatched from this view instead.
    std::unique_ptr<llama_memory_split> split_memory;

    // [QLH-SPLIT] Remote stages. A chain is either all-local or local-head-plus-remote-tails;
    // mixing both kinds in one chain is rejected rather than half-supported, because the two
    // paths have genuinely different failure modes and a half-tested blend of them would be
    // worse than either.
    std::vector<std::unique_ptr<llama_split_remote>> split_remotes;

    // [QLH-SPLIT] One frame's answer from the remote chain, plus the per-frame request bits that
    // go with it; see split_frame_answer above for the contract. A remote tail answers exactly
    // one row (its last token), so the batch index that row belongs to is kept next to it instead
    // of being assumed to be 0 -- that is what `logits_row` and `cands_row` are for.
    split_frame_answer split_answer;

    // [QLH-SPLIT] v5: the candidate channel request. The driving side asks for candidates one
    // decode at a time (llama_split_request_candidates in llama-ext.h), because whether the k
    // candidates are enough is a property of *this* request's sampler chain -- grammar,
    // pre-sampling logprobs, top_k <= 0 all change it -- and the tail can only answer what it was
    // asked. Kept here rather than in llama_split_remote because the decision is per request and
    // the connection outlives it.
    qlh_split::cand_req           split_cand_req{};
    std::vector<llama_logit_bias> split_cand_bias;
    std::vector<llama_token>      split_cand_hist;

    // [QLH-SPLIT] v9: the layer-input tap channel. `split_tap_ids` is the set of *source-model*
    // layer ids every remote decode asks for (empty for a whole model and for a local chain, where
    // nothing has to cross a wire). It is a property of the chain, not of the frame, so it lives
    // outside split_answer and survives begin_frame(); each stage answers the ids inside its own
    // range, so one set covers a chain of any length. The answered blocks (and the row/width they
    // were answered at) are per frame, so they are in split_answer.tap.
    std::vector<uint32_t> split_tap_ids;

    // decode output (2-dimensional array: [n_outputs][n_vocab])
    buffer_view<float> logits = {nullptr, 0};

    // embeddings output (2-dimensional array: [n_outputs][n_embd])
    // populated only when pooling_type == LLAMA_POOLING_TYPE_NONE
    buffer_view<float> embd = {nullptr, 0};

    // hidden state required by the nextn layers (2-dimensional array: [n_outputs][n_embd])
    // populated only when cparams.embeddings_nextn is enabled and the model graph
    // sets llm_graph_result::t_h_nextn
    buffer_view<float> embd_nextn = {nullptr, 0};

    // host buffers for output layer input embeddings, per layer
    // populated when cparams.output_layer_inp[il] is true
    std::vector<buffer_view<float>> embd_layer_inp;
    std::vector<int32_t> embd_batch_idxs; // extracted index -> original batch index

    struct sampling_info {
        // !samplers.empty() to check if any samplers are active
        std::map<llama_seq_id, llama_sampler *> samplers;

        buffer_view<float>       logits     = {nullptr, 0};
        buffer_view<llama_token> sampled    = {nullptr, 0};
        buffer_view<float>       probs      = {nullptr, 0};
        buffer_view<llama_token> candidates = {nullptr, 0};

        std::vector<uint32_t> logits_count;
        std::vector<uint32_t> probs_count;
        std::vector<uint32_t> candidates_count;

        // optimization
        std::vector<llama_token> token_ids_full_vocab;
    };

    sampling_info sampling;

    // sequence embeddings output (map of [n_embd] vectors)
    // populated only when pooling_type != LLAMA_POOLING_TYPE_NONE
    std::map<llama_seq_id, std::vector<float>> embd_seq;

    // reuse the batch_allocr to avoid unnecessary memory allocations
    std::unique_ptr<llama_batch_allocr> balloc;

    uint32_t n_input_tensors = 0; // number of tensors marked as input during the last graph reserve
    uint32_t n_outputs = 0; // number of actually-used outputs in the current ubatch or last logical batch

    std::vector<int32_t> output_ids; // map batch token positions to ids of the logits and embd buffers

    struct swap_info {
        uint32_t i0;
        uint32_t i1;
    };

    std::vector<swap_info> output_swaps;

    ggml_backend_sched_ptr sched;

    sched_params sched_params_cur;

    bool sched_need_reserve = true;

    // whether the reserve graph has a logits tensor, set in sched_reserve()
    // true until then, because the first output_reserve() runs before any graph is reserved
    bool graph_has_logits = true;

    // state of sched_copy_experts, reset before each graph compute
    struct copy_experts_info {
        const ggml_tensor *  ids = nullptr;
        std::vector<int32_t> ids_data;
        std::vector<bool>    used;

        void reset() {
            ids = nullptr;
            ids_data.clear();
            used.clear();
        }
    };

    copy_experts_info copy_experts;

    // graph reallocations during inference where the backend assignments did not change, split by whether the
    // graph size is the same as the previously scheduled one (unexpected) or different
    int32_t n_reallocs_same = 0;
    int32_t n_reallocs_diff = 0;

    ggml_backend_t backend_cpu = nullptr;
    std::vector<ggml_backend_ptr> backends;

    // training
    ggml_opt_context_t opt_ctx = nullptr;

    ggml_threadpool_t threadpool       = nullptr;
    ggml_threadpool_t threadpool_batch = nullptr;

    ggml_abort_callback abort_callback      = nullptr;
    void *              abort_callback_data = nullptr;

    std::vector<std::pair<ggml_backend_t, ggml_backend_set_n_threads_t>> set_n_threads_fns;

    // pointers and buffer types used for the compute buffer of each backend
    std::vector<ggml_backend_t>             backend_ptrs;
    std::vector<ggml_backend_buffer_type_t> backend_buft;
    std::vector<size_t>                     backend_buf_exp_size; // expected buffer sizes

    // Separate arenas give batches with and without outputs distinct CUDA graph cache keys.
    std::array<llm_graph_result_ptr, 2> gf_res_prev;
    llm_graph_result_ptr gf_res_reserve;

    llm_graph_result * gf_res_prev_active = nullptr;

    // One cached graph per decode width. MTP verify is usually n_draft+1 tokens
    // and a rejected step is 1 token; sharing a single slot makes those widths
    // rebuild each other, so the CUDA graph never finishes warmup.
    static constexpr uint32_t n_decode_graph_slots = 8;
    std::array<llm_graph_result_ptr, n_decode_graph_slots + 1> gf_res_decode;
    std::array<uint64_t, n_decode_graph_slots + 1> decode_graph_stamp = {};

    // host buffer for the model output (logits and embeddings)
    ggml_backend_buffer_ptr buf_output;

    // device buffer with enabled layer inputs, [n_layer_inp*n_embd, n_batch], row = batch index
    ggml_context_ptr        ctx_layer_inp_dev;
    ggml_backend_buffer_ptr buf_layer_inp_dev;
    ggml_tensor *           layer_inp_dev = nullptr;
    ggml_backend_dev_t      layer_inp_dev_dev = nullptr;
    ggml_backend_t          layer_inp_backend = nullptr;

    // ordering with the reader context (no host sync):
    //   written - recorded by this context after the batch, waited on by the reader
    //   read    - recorded by the reader after it consumed the rows, waited on by this context
    ggml_backend_event_ptr ev_layer_inp_written;
    ggml_backend_event_ptr ev_layer_inp_read;
    bool ev_layer_inp_read_recorded = false;

    // device buffer with the h_nextn rows, F32 [n_embd_out, n_batch + n_seq_max]:
    // row r < n_batch is the token at batch index r of the last decode, row n_batch + s is set by the host for seq s
    // ordered with the reader context like the layer input buffer
    ggml_context_ptr        ctx_h_nextn_dev;
    ggml_backend_buffer_ptr buf_h_nextn_dev;
    ggml_tensor *           h_nextn_dev = nullptr;
    ggml_backend_dev_t      h_nextn_dev_dev = nullptr;
    ggml_backend_t          h_nextn_backend = nullptr;
    ggml_backend_event_ptr  ev_h_nextn_written;
    ggml_backend_event_ptr  ev_h_nextn_read;
    bool ev_h_nextn_written_recorded = false;
    bool ev_h_nextn_read_recorded    = false;

    // keep copies of the per-sequence memory on the device
    std::map<llama_seq_id, llama_memory_buffers> mem_storage;

    bool has_evaluated_once = false;

    // env: LLAMA_GRAPH_REUSE_DISABLE
    bool graph_reuse_disable = false;

    // true after the prism.hadamard coverage check passes on a reserved graph
    bool hadamard_verified = false;

    // perf
    mutable int64_t t_start_us  = 0;
    mutable int64_t t_load_us   = 0;
    mutable int64_t t_p_eval_us = 0;
    mutable int64_t t_eval_us   = 0;

    mutable int64_t t_compute_start_us = 0;
    mutable int64_t n_queued_tokens    = 0;

    mutable int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)
    mutable int32_t n_eval   = 0; // number of eval calls

    mutable int32_t n_reused = 0; // number of times the previous graph was reused
};
