#pragma once

// this is a staging header for new llama.cpp API
// breaking changes and C++ are allowed. everything here should be considered WIP
// try as much as possible to not include this header in the rest of the codebase

#include "llama.h"
#include "llama-arch.h"
#include "llama-split-proto.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

//
// llama architecture queries
//

// all the LLM architectures known to the library
LLAMA_API std::vector<llm_arch> llama_arch_all();

// the canonical name of an architecture (e.g. "llama"), or "unknown"
LLAMA_API const char * llama_arch_name(llm_arch arch);

// the architecture with the given canonical name, or LLM_ARCH_UNKNOWN
LLAMA_API llm_arch llama_arch_from_string(const std::string & name);

// the arch-resolved GGUF key of a KV id, e.g. LLM_KV_CONTEXT_LENGTH -> "llama.context_length"
LLAMA_API std::string llama_arch_kv_name(llm_arch arch, llm_kv key);

LLAMA_API bool llama_arch_supports_sm_tensor(llm_arch arch);

LLAMA_API bool llama_arch_supports_mixed_batch(llm_arch arch);

// FIXME temporary function for better error messages
LLAMA_API bool llama_arch_supports_save(llm_arch arch);

// Reserve a new compute graph. It is valid until the next call to llama_graph_reserve.
LLAMA_API struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs);

// Get the default ggml_type for a given ftype.
LLAMA_API ggml_type llama_ftype_get_default_type(llama_ftype ftype);

struct quantize_state_impl;

LLAMA_API quantize_state_impl * llama_quant_init(
        const llama_model * model,
        const llama_model_quantize_params * params);

LLAMA_API void llama_quant_free(quantize_state_impl * qs);

// Descriptor for constructing a mock model for quantization testing.
struct llama_quant_model_desc {
    const char * architecture;
    uint32_t n_embd;
    uint32_t n_ff;
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_expert;
    uint32_t n_embd_head_k;
    uint32_t n_embd_head_v;
};

// Create a mock model from a metadata descriptor (for testing).
// The returned model must be freed with llama_model_free().
LLAMA_API llama_model * llama_quant_model_from_metadata(const llama_quant_model_desc * desc);

// Returns true if this tensor should be quantized (based on name, dims, params).
LLAMA_API bool llama_quant_tensor_allows_quantization(
        const quantize_state_impl * qs,
        const ggml_tensor * tensor);

// Compute quantization type assignments for a list of tensors.
// All tensors should be quantizable (use llama_quant_tensor_allows_quantization to filter).
// result_types: caller-allocated array of n_tensors elements, filled with assigned types.
LLAMA_API void llama_quant_compute_types(
        quantize_state_impl * qs,
        llama_ftype ftype,
        ggml_tensor ** tensors,
        ggml_type * result_types,
        size_t n_tensors);

//
// device memory querying
//

// "memory" as in physical memory for a buffer type, in bytes
struct llama_memory_breakdown_data {
    size_t model   = 0; // memory allocated for the model
    size_t context = 0; // memory allocated for the context
    size_t compute = 0; // memory allocated for temporary compute buffers

    size_t total() const {
        return model + context + compute;
    }
};

struct llama_device_memory_data {
    int64_t total;
    int64_t free;
    llama_memory_breakdown_data mb;
};

// TODO: convert to C-style data structure
using llama_memory_breakdown = std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data>;

LLAMA_API int32_t llama_model_n_expert (const struct llama_model * model);
LLAMA_API int32_t llama_model_n_devices(const struct llama_model * model);

LLAMA_API ggml_backend_dev_t llama_model_get_device(const struct llama_model * model, int i);

LLAMA_API llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx);

// the backend buffers that a state write of this memory would touch - empty caches report none
LLAMA_API std::vector<ggml_backend_buffer_t> llama_memory_get_buffers(const llama_memory_t mem);

// Turn the KV cache attention rotation on or off for the contexts created from this model.
// Useful for testing purposes.
LLAMA_API void llama_model_set_attn_rot(struct llama_model * model, bool value);

// Set whether the context outputs nextn embeddings or not
// If masked == true,  output the embeddings only for the tokens with batch.logits != 0
// If masked == false, output the embeddings for all tokens in the batch regardless of batch.logits
//
// [C-scheme] extern "C": this header is C++ (staging), but a C consumer such as the keep-head
// shim links by plain name. Without the linkage specification the symbol is mangled and
// GetProcAddress/ld fails even though the function is exported.
extern "C" LLAMA_API void llama_set_embeddings_nextn(struct llama_context * ctx, bool value, bool masked);

// Select which appended NextN block the DECODER_MTP graph runs (offset past
// the trunk: il = n_layer() + offset). Used by the speculative NextN driver to
// chain multiple trained NextN heads. Default 0 (first head).
LLAMA_API void llama_set_nextn_layer_offset(struct llama_context * ctx, int32_t offset);

// Marks the entries that a decision head (clef, vllm-decision2) reads, the default is 0
// See https://github.com/ggml-org/llama.cpp/pull/29831 for details
// A run of entries with the same value is one span, spans must be separated by entries with value 0
// An option belongs to the last question before it
enum llama_decision_order {
    LLAMA_DECISION_ORDER_NONE            = 0, // not read by the head
    LLAMA_DECISION_ORDER_QUESTION_NOUL   = 1, // text of a question
    LLAMA_DECISION_ORDER_QUESTION_CHOICE = 2,
    LLAMA_DECISION_ORDER_QUESTION_SCORE  = 3,
    LLAMA_DECISION_ORDER_OPTION          = 4, // text of an option
};
// The embeddings output has one value per entry: row i is the score of option i
LLAMA_API bool llama_batch_ext_set_decision_order(struct llama_batch_ext * batch, int32_t idx, enum llama_decision_order order);

// Label of option i of a decision model ("<arch>.decision.labels"), nullptr past the last one
LLAMA_API const char * llama_model_decision_label(const struct llama_model * model, uint32_t i);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
extern "C" LLAMA_API float * llama_get_embeddings_nextn(struct llama_context * ctx);

// LLAMA_API float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);
LLAMA_API float * llama_get_embeddings_nextn_ith(struct llama_context * ctx, int32_t i);

// Set whether the context outputs the input embeddings of a specific layer
LLAMA_API void llama_set_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid, bool value);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid);

// Write the enabled layer inputs to a device buffer [n_layer_inp*n_embd, n_batch] instead of host memory.
// Row r holds the token at batch index r, layers in ascending layer id order. Only the layers *this*
// context computes are in it; a DFlash/DSpark decoder may read it only when that is all of its extract
// layers (see llama_supports_embd_row).
LLAMA_API void llama_set_embeddings_layer_inp_dev(struct llama_context * ctx, bool value);

// Whether decode accepts entries added with llama_batch_ext_add_embd_row()
// (needs a DFlash/DSpark decoder context whose ctx_other has a layer input device buffer wide
// enough to hold every layer the decoder's encoder input expects -- a split chain that keeps some
// extract layers in another context or on another stage does not qualify).
LLAMA_API bool llama_supports_embd_row(const struct llama_context * ctx);

// Add an entry whose embedding is row `row` of the layer input device buffer of ctx_other.
// All entries of the batch must be of this kind. Positions are set with llama_batch_ext_set_pos(), n_pos_per_embd of them.
// Returns the batch index (>= 0), or the llama_batch_ext_add() error codes.
LLAMA_API int32_t llama_batch_ext_add_embd_row(struct llama_batch_ext * batch, llama_seq_id seq_id, int32_t row);

LLAMA_API llama_context * llama_get_ctx_other(struct llama_context * ctx);

// Write the unmasked h_nextn rows (see llama_set_embeddings_nextn) of each decode to a device buffer
// [n_embd_out, n_batch + n_seq_max] instead of host memory: row r < n_batch is the token at batch index r of the last decode,
// row n_batch + s is free for the caller, e.g. for seq s (see llama_set_embeddings_nextn_rows()).
// Returns false if the buffer cannot be used, the rows are then extracted to host memory.
LLAMA_API bool llama_set_embeddings_nextn_dev(struct llama_context * ctx, bool value);

// Copy rows of the h_nextn device buffer to or from host memory, ordered after the last decode and its readers
LLAMA_API bool llama_get_embeddings_nextn_rows(struct llama_context * ctx, int64_t row0, int64_t n_rows, float * out);
LLAMA_API bool llama_set_embeddings_nextn_rows(struct llama_context * ctx, int64_t row0, int64_t n_rows, const float * data);

// Read the state rows of batches (see llama_batch_ext_set_embd_state_row()) from the h_nextn device buffer of src.
// Returns whether decode accepts batches with state rows.
LLAMA_API bool llama_set_embd_state_src(struct llama_context * ctx, struct llama_context * src);

// The state of entry idx is row `row` of the h_nextn device buffer of the context set with llama_set_embd_state_src().
// All entries of a batch have a state row, or none.
LLAMA_API bool llama_batch_ext_set_embd_state_row(struct llama_batch_ext * batch, int32_t idx, int32_t row);

//
// model/context data extraction
//

// [QLH-SPLIT] The source layer range this model was loaded with, after both entry points have
// been resolved: llama_model_params::split_layer_lo/hi wins, LLAMA_LAYER_RANGE is the fallback
// (src/llama-model-loader.cpp). Returns false for a whole-model load, where there is no range
// to continue. A caller that only reads the load params cannot tell an env-provided range from
// no range at all -- the params stay -1 either way -- so it would judge a range that is not the
// one this model actually loaded. Ask the model instead.
LLAMA_API bool llama_model_layer_range(const struct llama_model * model, int32_t * lo, int32_t * hi);

// [QLH-SPLIT] Is this context's chain still trustworthy? True once a remote stage failed an
// operation that leaves the stages holding different state (a range that could not be trimmed, a
// clear the stage never took). decode_split answers -3 for every following request, so a caller
// only needs this to tell "the request failed because the chain is gone" apart from "the request
// failed for its own reasons" -- and to say which stage, for a human.
//
// Both functions are safe to call from the thread that drives the context, i.e. the same thread
// llama_decode() is called on; they do not block, allocate, or touch the network. From another
// thread they are a data race on the reason string, which is why the server asks from its main
// loop (the failure is reported through the request that hits it) and not from /health.
//
// false for a whole-model context and for a local chain: no remote stage, nothing to lose.
LLAMA_API bool llama_context_split_failed(const struct llama_context * ctx);

// The same answer as a string ("remote stage 1 (host:port) failed to ...") or "" when the chain
// is fine. Only the reason string; llama_context_split_failed() is the decision.
LLAMA_API const char * llama_context_split_failure(const struct llama_context * ctx);

// [QLH-SPLIT] v6: the last state restore failed because a remote stage no longer holds the
// snapshot the checkpoint names -- it evicted it from its snapshot cache, or it is a different
// process than the one that minted the handle. True only for that attempt; the next state
// restore clears it.
//
// This is deliberately *not* llama_context_split_failed(). The stage refuses before applying
// anything and the head is restored last, so no stage moved: the chain is intact and the only
// thing missing is one checkpoint. A caller that has a recovery for "no cached state" (the
// server re-processes the prompt) must take it; treating this as a dead chain would turn a
// bounded snapshot cache into a permanently failing request. Nothing else in the tree needs to
// ask -- decode() does not consult it.
LLAMA_API bool llama_context_split_state_stale(const struct llama_context * ctx);

//
// [QLH-SPLIT] v5: the remote candidate channel
//
// A remote tail answers a decode with the whole logits row by default, and that row is 993 KB for
// the model this fork ships (248320 f32). On a 16 MB/s link it costs ~59 ms per generated token,
// which is most of the gap between a cross-machine chain and a whole model. It is also almost
// entirely wasted: the sampler chain truncates to `top_k` candidates before it does anything
// else, and every step after top_k only ever looks at those k entries.
//
// So a decode may ask the tail to run the steps up to and including top_k and answer with the k
// (id, logit) pairs instead of the row: 320 bytes for k = 40. The steps after top_k then run on
// this side, on exactly the candidate array the row path would have left in cur_p.
//
// This is only an optimization when it is exact, so the decision is not made here: the caller
// checks that the k candidates are a faithful substitute for this request (no grammar, no
// pre-sampling logprobs, top_k > 0, and a sampler chain whose split is supported) and otherwise
// never calls llama_split_request_candidates() at all. See common/sampling.h for that rule; it
// exists once, so that no caller can invent its own.
//
// Why the request is per frame and not a mode: the tail has to rebuild the penalty state from the
// history in every frame anyway (see cand_req), and a decode that was not asked for candidates
// must answer with the row -- a sticky flag would let one fallback request poison the next one.
LLAMA_API bool llama_context_split_has_remote(const struct llama_context * ctx);

// Ask the next remote decode for candidates. `req` describes the steps the tail must run and
// their parameters (qlh_split::cand_req), `biases` are the logit biases that belong to those
// steps (the tail has no common_params to read them from) and `hist` is the recent token window
// the penalties step needs, oldest first.
//
// Ignored (and the next decode answers with the row) when this context has no remote stage, so a
// caller does not have to distinguish the two chain kinds. The request covers exactly one decode.
LLAMA_API void llama_split_request_candidates(
                struct llama_context * ctx,
        const struct qlh_split::cand_req * req,
        const struct llama_logit_bias * biases, uint32_t n_bias,
        const llama_token * hist, uint32_t n_hist);

// [QLH-SPLIT] Candidates of the last remote decode. Returns their count and, through the out
// parameters, the array (sorted by descending logit) and the batch row they describe; 0 means the
// tail answered with the whole row instead (or nothing was decoded yet).
//
// `row` exists because a remote tail answers for one batch row -- the last one it was handed --
// and the caller must not sample a different row from this array. Comparing it against the batch
// index is the same check get_logits_ith() performs for the row path.
struct llama_split_token_cand {
    int32_t id;
    float   logit;
};

LLAMA_API uint32_t llama_split_take_candidates(
        const struct llama_context * ctx,
        const struct llama_split_token_cand ** cands,
        int32_t * row);

//
// [QLH-SPLIT] v7: the per-row verify channel
//
// Speculative verification decodes k drafts + 1 target row in one batch and needs the *token* each
// row produces, not its distribution: under a greedy chain (temp 0, or llama_sampler_init_greedy)
// the token of a row is its argmax, and common_sampler_sample_and_accept_n() only re-samples to
// compare against the draft. A remote tail already holds those rows in its own memory, so the
// only thing that has to cross the wire is one int32 per row: 16 B for the default k = 3, against
// 993 KB for a single logits row and 3.9 MB for the four that verification actually reads.
//
// Why this is a separate channel from the candidates above: the candidate channel replaces the
// *sampler* for one row (it needs the chain's parameters to reproduce the steps before top_k),
// while this one answers a question that has no parameters at all -- argmax. Mixing them would
// tie a parameter-free answer to a per-request sampling plan for no reason, and the two would
// then have to be kept mutually consistent. One flag each.
//
// The rule is the same as the candidate channel's and lives in the same place
// (common_sampler_verify_rows_check, common/sampling.h): a caller may only arm this when every
// row it will read produces its argmax through the chain. A token id carries no distribution, so
// there is no way to undo a wrong decision after the frame -- the check is the only guard.
//
// Rows are all-or-nothing per frame: an armed decode answers with n_tokens ids (the whole batch),
// and a frame with more rows than qlh_split::MAX_VERIFY_ROWS falls back to the whole-row path
// before anything is sent. A caller that must not fall back (it reads tokens, so it cannot read
// logits) caps its batch instead -- llama-server caps the draft block.
LLAMA_API void llama_split_request_verify_rows(struct llama_context * ctx, bool on);

// The token the tail chose for batch row `idx` of the last remote decode, or LLAMA_TOKEN_NULL
// when that frame did not use the channel (or the index is outside it).
//
// No synchronization: like llama_get_logits_ith(), call it on the thread that drives the context
// after the decode returned.
LLAMA_API llama_token llama_split_verified_token(const struct llama_context * ctx, int32_t idx);

//
// [QLH-SPLIT] v8: the remote MTP draft
//
// The MTP head is a full transformer block appended after the trunk (blk.<n_layer>.nextn.*), it has
// no lm_head and no embedding of its own, and its input is t_h_nextn = RMSNorm(last trunk layer
// output). In a split chain that last layer runs in the tail, so the head process cannot compute
// the head's input at all -- which is why a remote chain could only draft with ngram until v8.
//
// The division of labour is the same one the rest of the protocol makes: the stage that owns the
// weights does the compute, and the wire carries only what the other side cannot derive.
//
//  - the tail sends back the h_nextn row(s) of the rows it answered for, on any decode that was
//    armed with llama_split_request_nextn_rows(). One row for a plain decode (the last one), all
//    rows for a verify frame -- that is the only row whose h the next draft round can start from.
//  - the head hands one of those rows back with the token at that position and gets k draft tokens.
//    The k steps are recursive (draft i's h is draft i-1's output), so they cannot be batched and
//    they run inside the stage; what the protocol makes one round trip is the whole block.
//
// Everything below answers "no drafts" instead of failing when the chain cannot do it: a whole
// model, a local chain, a tail without a NextN block, a stage that lost its MTP context, a
// transport error. Drafting is an optimization and none of its failures may take a request down --
// the request that follows is exactly the one the chain would have served without speculation.
//
// Two rules live on the caller's side and are not enforced here, because they are properties of the
// sampler, not of the chain:
//  - the drafts are greedy, so they may only be verified by a chain that samples the argmax
//    (common_sampler_verify_rows_check); a probabilistic verify would need q over the whole
//    vocabulary, which is the cost this layer exists to avoid.
//  - k is capped at qlh_split::MAX_MTP_DRAFT (= MAX_VERIFY_ROWS - 1), because a round has to fit
//    one verify frame.
LLAMA_API bool llama_context_split_mtp_ready(const struct llama_context * ctx);

// Arms the next remote decode to answer with its h_nextn rows (see above). Ignored by a local chain
// and by a whole model, so a caller does not have to distinguish the chain kinds.
LLAMA_API void llama_split_request_nextn_rows(struct llama_context * ctx, bool on);

// The h_nextn rows of the last remote decode: `n_rows` rows of `n_embd` floats each, the first one
// belonging to batch row `row0`. Returns 0 (and nulls the out parameters) when that decode carried
// none -- which is the answer for every chain that cannot draft, and also for a decode nobody armed.
//
// Call it on the thread that drives the context, after the decode returned, for the same reason
// llama_get_logits_ith() has to be called there.
LLAMA_API uint32_t llama_split_take_nextn_rows(
        const struct llama_context * ctx,
        const float ** rows,
        int32_t * row0,
        uint32_t * n_embd);

// Runs one draft round on the remote tail: `h` is `n_embd` floats (one t_h_nextn row, from
// llama_split_take_nextn_rows), `token` is the token at position `pos`, and k drafts are written to
// `out` (which must hold k entries). Returns the number written (>= 0; 0 means "this chain cannot
// draft, carry on without it") or -1 on a transport failure, which the caller treats as "no drafts"
// as well -- never as a dead chain.
//
// `seq_id` names the sequence the draft belongs to; the tail's draft context is per sequence, so a
// frame that named another one would draft over a different request's KV.
LLAMA_API int32_t llama_context_split_mtp_draft(
                struct llama_context * ctx,
                     const float * h,
                       int32_t   token,
                       int32_t   pos,
                      uint32_t   k,
                       int32_t   seq_id,
                 llama_token *   out,
                    uint32_t *   n_out);

//
// [QLH-SPLIT] v9: the layer-input tap channel
//
// DFlash and DSpark do not draft from logits or from the last layer's output: they read the *input*
// row of a set of trunk layers (dflash.target_layers -- 8 layers for DFlash, 5 for DSpark), project
// them with the draft model's own encoder and inject one row per batch position into the draft model's
// KV cache. The layers themselves are identified by source-model id, and a split chain cuts the model
// in the middle of that list, so the upper taps are computed by the tail. The local chain reads them
// straight out of the tail context (common/speculative.cpp, tap_ctx/tap_lid) and moves no bytes; a
// remote chain had no way to, which is why both drafters worked only on a local chain -- and the
// measurement that makes this worth a wire says so plainly: the only drafter that does work over a
// remote chain (MTP, through its own stage op) loses 24.4% of its acceptance and 0.44 of mean accepted
// length, while DFlash and DSpark lose nothing on a local chain.
//
// What crosses the wire is what the head cannot compute: n_taps rows per batch row, n_embd floats
// each. For the shipped model that is 8 KB per row per tap, i.e. 48 KB for a k = 2 verify frame with
// DFlash's two tail taps -- against 993 KB for one row of logits. The frame that costs is the prefill:
// the draft needs every prompt position injected, so an n_ubatch-row prefill frame moves
// n_ubatch * n_taps * 8 KB. That cost is paid once per prompt, not per token, which is why this is
// still the right trade.
//
// The set is *sticky*, unlike the three channels above: which layers a draft model reads is a property
// of the chain, not of a frame, so it is set once and every decode asks for it until it is cleared.
// Ids are source-model ids; each stage answers the ones inside its own range, so one call covers a
// chain of any length, including one where a relay owns some of the taps.
LLAMA_API void llama_split_set_layer_inp(
                struct llama_context * ctx,
                     const uint32_t * ids,
                           uint32_t   n);

// How many layer-input blocks the last remote decode answered, and for block `i` its source-layer id,
// its row count and the width of a row. The pointer is the block's rows, [n_rows][n_embd], and is
// valid until the next decode -- like every other answer on this path, read it after the decode
// returned.
//
// The blocks are the ids some remote stage owned, not the ids this side asked for: a caller must look
// its layers up by id. 0 blocks is a valid answer (a local chain, a whole model, a frame that asked
// for nothing, or a chain none of whose stages own the asked layers); the draft driver treats it as
// "draft without the injection" rather than as a failure.
LLAMA_API uint32_t llama_split_take_layer_inp_n(const struct llama_context * ctx);

LLAMA_API const float * llama_split_take_layer_inp(
                const struct llama_context * ctx,
                                       uint32_t i,
                                       uint32_t * id,
                                       uint32_t * n_rows,
                                       uint32_t * n_embd);

LLAMA_API int32_t llama_model_dflash_selector_top_k(const struct llama_model * model);
// returns pointer to the target-model layer indices
LLAMA_API const int32_t * llama_model_target_layer_ids  (const struct llama_model * model);
// returns the number of extracted layers from target model
LLAMA_API uint32_t        llama_model_target_layer_ids_n(const struct llama_model * model);

// retrieves the whole token embedding matrix in F32 format (n_embd * n_vocab)
// returns total number of elements or 0 on error
// if out is nullptr, returns the number of tokens without writing to out
// caller must allocate enough memory for out before calling
LLAMA_API uint32_t llama_model_get_tok_embd(const struct llama_model * model, float * out);

// same as above, but only the rows of the given tokens, in order (n_tokens * n_embd elements)
// returns the number of elements written or 0 on error
LLAMA_API uint32_t llama_model_get_tok_embd_rows(const struct llama_model * model, const llama_token * tokens, uint32_t n_tokens, float * out);

// realtime models: number of tokens the output lags behind the audio input, used by mtmd; 0 if not applicable
LLAMA_API uint32_t llama_model_n_tokens_delay(const struct llama_model * model);

// each sequence is paired with an unconditional one at n_seq_max / 2 + seq_id, in a unified KV cache
LLAMA_API bool llama_model_classifier_free_guidance(const struct llama_model * model);
