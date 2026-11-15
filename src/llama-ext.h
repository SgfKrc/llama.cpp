#pragma once

// this is a staging header for new llama.cpp API
// breaking changes and C++ are allowed. everything here should be considered WIP
// try as much as possible to not include this header in the rest of the codebase

// Fork addition: the block between the "begin/end: K-Llama" markers below is this fork's
// split-chain API -- the stage chain, the source layer range and the per-frame remote channels
// (src/llama-split-proto.h). It is exported with LLAMA_API because llama-server and
// tools/split-stage both have to reach it, but it is *not* upstream staging API: C++ only, may
// change without deprecation, like the rest of this header. It lives here instead of in a header
// of its own because every consumer already includes this one for llama_arch_* and the model
// getters; a second header would add a second include to the same translation units without
// removing split's private types from any of them (llama_split_request_candidates needs
// qlh_split::cand_req, so the include of llama-split-proto.h has to stay either way).
//
// Keep the two markers when applying upstream changes: everything outside them is upstream's.

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

LLAMA_API void llama_quant_free(const quantize_state_impl * qs);

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

LLAMA_API std::vector<ggml_tensor *> llama_memory_get_tensors(const llama_memory_t mem);

// the backend buffers that a state write of this memory would touch - empty caches report none
LLAMA_API std::vector<ggml_backend_buffer_t> llama_memory_get_buffers(const llama_memory_t mem);

// Turn the KV cache attention rotation on or off for the contexts created from this model.
// Useful for testing purposes.
LLAMA_API void llama_model_set_attn_rot(struct llama_model * model, bool value);

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

// Questions packed after a shared state in one sequence, contexts without memory only
// 0 (default) is the state, k > 0 the k-th question; full attention layers hide other questions from a question
LLAMA_API bool llama_batch_ext_set_decision_group(struct llama_batch_ext * batch, int32_t idx, int32_t group);

// Label of option i of a decision model ("<arch>.decision.labels"), nullptr past the last one
LLAMA_API const char * llama_model_decision_label(const struct llama_model * model, uint32_t i);

//
// ==== begin: K-Llama split-chain additions (this fork, not upstream staging API) ====
//

//
// model/context data extraction
//

// Encoder-input width. DFlash/EAGLE3 draft graphs consume
// target_layers * n_embd (8 x 2048 = 16384) wide rows, which is neither n_embd_inp
// (2048) nor n_embd_out (2048). Without this accessor the only way to size a buffer
// for those graphs is to guess -- and guessing n_embd_out made the warmup empty-run
// read past a 2048-wide zero state and take an access violation.
LLAMA_API int32_t llama_model_n_embd_inp_enc(const struct llama_model * model);

// Trunk layer count of the source model, for a model loaded with
// llama_model_params::split_layer_lo/hi. llama_model_n_layer() is the local depth of that
// range, so it says nothing about where the range sits in the source model. Returns 0 when
// the model was loaded whole (no range was requested).
LLAMA_API int32_t llama_model_n_layer_src  (const struct llama_model * model);

// The source layer range this model was loaded with, after both entry points have
// been resolved: llama_model_params::split_layer_lo/hi wins, LLAMA_LAYER_RANGE is the fallback
// (src/llama-model-loader.cpp). Returns false for a whole-model load, where there is no range
// to continue. A caller that only reads the load params cannot tell an env-provided range from
// no range at all -- the params stay -1 either way -- so it would judge a range that is not the
// one this model actually loaded. Ask the model instead.
LLAMA_API bool llama_model_layer_range(const struct llama_model * model, int32_t * lo, int32_t * hi);

// Whether this model's range loads the two global tensors at the ends of the source model:
// output_norm/output (the head that turns hidden into logits) and token_embd (the input layer).
//
// The definition is src/qlh-layer-range.h -- the same three functions the loader calls to decide
// which tensors to read -- so "this stage produces logits" and "the loader read output.weight"
// cannot be two different answers. Six call sites used to spell the arithmetic out; the attach
// check's copy differed from the loader's for layer_hi == 0, which is what a comment saying "same
// predicate here" is worth when the predicate is written twice.
//
// Ask the model, not the load params: LLAMA_LAYER_RANGE is the loader's fallback, so params stay
// -1 for a range that came from the environment. A whole-model load answers true for both.
LLAMA_API bool llama_model_has_output   (const struct llama_model * model);
LLAMA_API bool llama_model_has_tok_embd (const struct llama_model * model);

//
// The stage chain: tails attached to a head context
// unrelated to upstream llama_split_path() / llama_split_prefix(), which name gguf shards;
// "split" here means a pipeline of source layer ranges
//

// Attach continuation stages to a context. `stages` are borrowed: the
// caller owns them and must keep them alive while `ctx` decodes. With zero stages the
// context behaves exactly as before. A context with stages acts as the head of a
// pipeline: decode() runs the whole chain and the final logits stay reachable through
// the usual llama_get_logits* calls on `ctx`.
//
// The stages must tile the source model around the head: the head must start at layer 0
// (otherwise the layers below it would never be computed, and the head would decode from
// hidden states that nobody produced), the first stage starts where the head ends, every
// stage starts where the previous one ended, and the last one ends at the source model's
// trunk depth. Returns the number of stages attached, or -1 (with a logged reason) when
// the chain is inconsistent or the arguments are invalid; on failure the context keeps
// the chain it had.
LLAMA_API int32_t llama_split_attach(
        struct llama_context * ctx,
        struct llama_context ** stages,
                      int32_t   n_stages);

LLAMA_API int32_t llama_split_n_stages(const struct llama_context * ctx);

// Attach remote stages (host:port) to a context. The chain is then
// local head + remote tails; mixing with llama_split_attach is rejected. Returns the
// number attached, or -1 when it could not connect, when the chain already has local
// stages, or when the context cannot be expressed over the stage protocol (n_seq_max > 1
// is refused, with the reason in err_out). Decode refuses any batch whose M-RoPE position
// the protocol cannot carry (more than one nonzero position dimension per token).
LLAMA_API int32_t llama_split_attach_remote(
        struct llama_context * ctx,
        const char * const   * endpoints,
                      int32_t   n_endpoints,
                        char *      err_out,
                      size_t      err_len);

// Is this context's chain still trustworthy? True once a remote stage failed an
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

// The last state restore failed because a remote stage no longer holds the
// snapshot the checkpoint names -- it evicted it from its snapshot cache, or it is a different
// process than the one that minted the handle. True only for that attempt; the next state
// restore clears it.
//
// This is deliberately *not* llama_context_split_failed(). The stage refuses before applying
// anything and the head is restored last, so no stage moved: the chain is intact and the only
// thing missing is one checkpoint. A caller that has a recovery for "no cached state" (the
// server re-processes the prompt) must take it; treating this as a dead chain would turn a
// bounded snapshot cache into a permanently failing request. Nothing else needs to
// ask -- decode() does not consult it.
LLAMA_API bool llama_context_split_state_stale(const struct llama_context * ctx);

//
// The remote candidate channel
//
// A remote tail answers a decode with the whole logits row by default, and that row is 993 KB for
// the model this build targets (248320 f32). On a 16 MB/s link it costs ~59 ms per generated token,
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
//
// It is per sequence *and* per frame. llama-server is one context
// with N slots, and two requests have two samplers, so the request names the seq_id it belongs to;
// the frame then flags only that sequence's sampling row with ROW_FLAG_CANDS and carries its own
// cand_req, bias list and history.
LLAMA_API bool llama_context_split_has_remote(const struct llama_context * ctx);

// This context's --split-verify-rows, i.e. the most result rows one decode frame
// may answer. It is a chain-wide number (both ends compare it through the identity handshake), so a
// caller that has to bound a draft block uses the context's value instead of a constant -- that is
// what makes "one sequence with k = 16" and "four sequences with k = 3" both work on the same build.
//
// Answered for every context, chain or not: an ordinary whole-model context returns the default,
// because nothing on that path is bounded by a frame.
LLAMA_API uint32_t llama_context_split_verify_rows(const struct llama_context * ctx);

// Ask the next remote decode for candidates. `req` describes the steps the tail must run and
// their parameters (qlh_split::cand_req), `biases` are the logit biases that belong to those
// steps (the tail has no common_params to read them from) and `hist` is the recent token window
// the penalties step needs, oldest first.
//
// Ignored (and the next decode answers with the row) when this context has no remote stage, so a
// caller does not have to distinguish the two chain kinds. The request covers exactly one decode of
// one sequence; `seq_id` is that sequence, and `req == nullptr` disarms it (only it).
LLAMA_API void llama_split_request_candidates(
                struct llama_context * ctx,
                llama_seq_id seq_id,
        const struct qlh_split::cand_req * req,
        const struct llama_logit_bias * biases, uint32_t n_bias,
        const llama_token * hist, uint32_t n_hist);

// Candidates of the last remote decode for batch row `row`. Returns their count and,
// through `cands`, the array (sorted by descending logit); 0 means the tail answered that row with
// the whole row instead (or nothing was decoded yet, or the row was not one of the answered ones).
//
// The row is an *input*, not an output. One reply can carry several
// rows' candidate sets -- two sequences with two top_k values -- so "the candidates" is not a
// single array and the caller must name the row it is about to sample from. That is the same index
// get_logits_ith() takes, so the two paths cannot drift on which row they describe.
struct llama_split_token_cand {
    int32_t id;
    float   logit;
};

LLAMA_API uint32_t llama_split_take_candidates(
        const struct llama_context * ctx,
        int32_t row,
        const struct llama_split_token_cand ** cands);

//
// The per-row verify channel
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
// Arming is per sequence. Only the rows of `seq_id` carry
// ROW_FLAG_VERIFY; every other row of the same frame is answered the way *its* sequence asked (the
// whole logits row, or candidates). A frame with more rows than the chain's --split-verify-rows
// (a runtime value, see llama_context_split_verify_rows) still falls back before anything is sent,
// per row: those rows are downgraded to ROW_FLAG_LOGITS and the caller reads them through the row
// path. A caller that must not fall back caps its batch instead -- llama-server caps the draft
// block, per sequence, so the sum of every sequence's block fits one frame.
LLAMA_API void llama_split_request_verify_rows(struct llama_context * ctx, llama_seq_id seq_id, bool on);

// The token the tail chose for batch row `row` of the last remote decode, or LLAMA_TOKEN_NULL when
// that row was not one of the verified ones (or no frame used the channel).
//
// `row` is the batch row of the frame that was decoded -- the same
// index llama_get_logits_ith() takes -- and the reply's row table is what maps it into the answer.
// A frame whose verify rows do not start at row 0 (sequence A verifying while sequence B decodes
// ahead of it) is the case this exists for; reading the answer positionally would hand A another
// row's token, i.e. a wrong token with no error anywhere.
//
// No synchronization: like llama_get_logits_ith(), call it on the thread that drives the context
// after the decode returned.
LLAMA_API llama_token llama_split_verified_token(const struct llama_context * ctx, int32_t row);

//
// The remote MTP draft
//
// The MTP head is a full transformer block appended after the trunk (blk.<n_layer>.nextn.*), it has
// no lm_head and no embedding of its own, and its input is t_h_nextn = RMSNorm(last trunk layer
// output). In a split chain that last layer runs in the tail, so the head process cannot compute
// the head's input at all -- which is why a remote chain can only draft with ngram without the MTP
// draft channel.
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
//  - k is the caller's own draft block (--spec-draft-n-max) bounded by the chain's
//    --split-verify-rows minus one, because a round has to fit one verify frame. There is no
//    protocol constant: the per-sequence k is what the configured draft block says,
//    and what has to fit is n_seq_max * (k + 1) rows, which the head checks at attach time.
LLAMA_API bool llama_context_split_mtp_ready(const struct llama_context * ctx);

// Arms the next remote decode to answer with the h_nextn rows of this sequence's rows (see above).
// Ignored by a local chain and by a whole model, so a caller does not have to distinguish the chain
// kinds.
//
// Per sequence, because the rows are. A frame may carry sequence A's
// verify block (which wants all of its k + 1 rows) and sequence B's plain decode (which wants its
// one row); arming the whole frame because some slot wanted a row is the frame-level shape this
// replaces.
LLAMA_API void llama_split_request_nextn_rows(struct llama_context * ctx, llama_seq_id seq_id, bool on);

// The h_nextn rows of the last remote decode: `n_rows` rows of `n_embd` floats each, and `row_ids`
// names the batch row each of them belongs to, in the same order. Returns 0 (and nulls the out
// parameters) when that decode carried none -- which is the answer for every chain that cannot
// draft, and also for a decode nobody armed.
//
// A table, not a single anchor row. The rows of one frame can belong to several sequences, and
// the caller has to give each
// row to the draft context of *its own* sequence -- one `row0` could only describe a contiguous
// block of one sequence, which is exactly what a mixed frame is not.
//
// Call it on the thread that drives the context, after the decode returned, for the same reason
// llama_get_logits_ith() has to be called there.
LLAMA_API uint32_t llama_split_take_nextn_rows(
        const struct llama_context * ctx,
        const float ** rows,
        const int32_t ** row_ids,
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
// The layer-input tap channel
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

//
// ==== end: K-Llama split-chain additions ====
//

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
