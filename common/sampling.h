#pragma once

#include "llama.h"

#include "common.h"

#include <string>
#include <vector>

// common_sampler extends llama_sampler with additional functionality:
//
//  - grammar support
//  - custom sampler logic based on the parameters
//  - history of the last accepted tokens
//  - performance metrics
//
// This goal is to have a common implementation of the sampling logic shared across the examples.
// For example, depending on the temperature, the sampling chain can be very simple (greedy) or more
// complex (top-k, top-p, etc).
//
// Another example is related to the grammar. In general, the grammar constraints applied on the full
// vocabulary can be very taxing. To improve performance, the grammar can be applied only to the sampled
// token in order to verify if it fits the grammar. And only if the token doesn't fit the grammar, the
// grammar constraints are applied to the full vocabulary and the token is resampled.
//
// The common_sampler also maintains a container with the last accepted tokens. In the future, this can
// be moved into the core llama library.
//
// For convenience, the common_sampler also maintains a container with the current candidate tokens.
// This can be used to access the probabilities of the rest of the non-sampled tokens.
//
// TODO: measure grammar performance
//

struct common_sampler;

// llama_sampler API overloads

// note: can mutate params in some cases
struct common_sampler * common_sampler_init(
        const struct llama_model * model,
        struct common_params_sampling & params);

void common_sampler_free(struct common_sampler * gsmpl);

// if is_generated is true, the token is accepted by the sampling chain, the reasoning budget sampler, and the grammar sampler
void                    common_sampler_accept(struct common_sampler * gsmpl, llama_token token, bool is_generated);
void                    common_sampler_reset (struct common_sampler * gsmpl);
struct common_sampler * common_sampler_clone (struct common_sampler * gsmpl);
void                    common_sampler_copy  (const struct common_sampler * src, struct common_sampler * dst);

// arguments can be nullptr to skip printing
void common_perf_print(const struct llama_context * ctx, const struct common_sampler * gsmpl);

// get the underlying llama_sampler_chain
struct llama_sampler * common_sampler_get(const struct common_sampler * gsmpl);

// extended sampling implementation:
//
// - set logits
// - apply the configured sampler chain
// - check if the token fits the grammar (if any)
// - if not: resample by first applying the grammar constraints and then sampling again (slower path)
//
// if grammar_first is true, the grammar is applied before the samplers (slower)
// useful in cases where all the resulting candidates (not just the sampled one) must fit the grammar
//
// [QLH-SPLIT] v7/v8/v9: on a frame answered through the per-row verify channel (see
// common_sampler_verify_rows_check) there is no logits row, so a token the grammar or the
// reasoning budget rejects cannot be resampled. The call answers LLAMA_TOKEN_NULL instead: "this
// row cannot be used". That is a signal, not a token -- the caller must not accept it and must not
// treat it as an id.
llama_token common_sampler_sample(struct common_sampler * gsmpl, struct llama_context * ctx, int idx, bool grammar_first = false);

// generalized version of common_sampler_sample
//
// will cross-reference the sampled tokens with a batch of draft tokens and accept those that match
// if the sampler disagrees at some point, we stop and return the accepted tokens up to now
//
//      common_sampler_sample_n(gsmpl, ctx, { idx }, {});
//
// is equivalent to
//
//      common_sampler_sample(gsmpl, ctx, idx);
//      common_sampler_accept(gsmpl, token, true);
//
// requires: idxs.size() == draft.size() + 1
//
// returns 1 .. idxs.size() tokens on the row path.
//
// [QLH-SPLIT] v8/v9: returns *nothing at all* (empty) when a row of the frame answers
// LLAMA_TOKEN_NULL, i.e. when the grammar or the reasoning budget rejected that row's token. The
// rows accepted before it have already been accepted by the sampler (and so advanced the grammar
// and spent budget), so the caller must roll the sampler and the KV back (it holds a checkpoint
// for exactly this) and decode that position through the row path. The frame is abandoned whole
// rather than shortened because a shortened frame would be a replay whose last row is the rejected
// position, which the next round would reject again.
std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, bool grammar_first = false);

// as above, but verifies by rejection sampling; draft_q holds the draft's candidates per token
std::vector<llama_token> common_sampler_sample_and_accept_n_rejection(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, const std::vector<std::vector<llama_token_data>> & draft_q, bool grammar_first = false);

// assume idxs == [ 0, 1, 2, ..., draft.size() ]
std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const llama_tokens & draft, bool grammar_first = false);

uint32_t common_sampler_get_seed(const struct common_sampler * gsmpl);

// force the reasoning budget sampler (if any) to begin forcing its end sequence now.
bool common_sampler_reasoning_budget_force(struct common_sampler * gsmpl);

// [QLH-SPLIT] v5: the remote candidate channel, driving side.
//
// A remote tail can answer a decode with its top-k candidates instead of the whole logits row
// (993 KB -> 320 B per token for the model this fork ships; see src/llama-ext.h). The point of
// this block is to make that a *pure* optimization: the candidate path must produce exactly the
// token the row path would have produced, or not be used at all.
//
// Why it can be exact: common_sampler's chain truncates the vocabulary in its `top_k` step, and
// every step after top_k only ever looks at the entries left in cur_p. So the tail runs the
// steps *before* top_k (logit_bias, penalties, optional top_n_sigma), answers with the k entries
// top_k would have kept, and this side runs the remaining steps on exactly that array. The
// steps after top_k cannot tell the difference, because there is nothing left for them to see.
// [QLH-SPLIT] v5: the steps a split-stage can run before top_k. The numbers are the wire values
// of qlh_split::cand_step (src/llama-split-proto.h); they are repeated here instead of including
// that header, because common/ must not grow a dependency on the split protocol -- the static
// assertion that keeps them in sync lives next to the code that fills the frame
// (common/sampling.cpp).
enum common_cand_step : uint8_t {
    COMMON_CAND_STEP_PENALTIES   = 1,
    COMMON_CAND_STEP_TOP_N_SIGMA = 2,
    COMMON_CAND_STEP_TOP_K       = 3,
};

struct common_sampler_candidates {
    int32_t  top_k           = 0;
    float    top_n_sigma     = -1.0f;      // < 0: the chain has no top_n_sigma step to run
    int32_t  penalty_last_n  = 0;
    float    penalty_repeat  = 1.0f;
    float    penalty_freq    = 0.0f;
    float    penalty_present = 0.0f;
    uint32_t n_steps         = 0;
    uint8_t  steps[3]        = {0, 0, 0}; // qlh_split::cand_step values, in chain order

    // The merged logit-bias list the chain applies before anything else (user biases, EOG
    // suppression, the vocabulary's own suppress tokens). The tail needs it verbatim: those
    // entries are -INFINITY and have to be applied before top_k ranks, otherwise a suppressed
    // token takes a candidate slot that the row path would have given to a usable one. Owned by
    // the sampler; valid while it is alive.
    const std::vector<llama_logit_bias> * biases = nullptr;
};

// [QLH-SPLIT] v5: can this request be sampled from its own top-k candidates?
//
// One function, one place, on purpose: the fallback is the highest-risk part of the feature, and
// a caller that invented its own subset of the checks would silently sample from a truncated
// distribution. Returns false and fills `why` with the reason the request must use the whole row;
// the caller is expected to log `why` once per request, because otherwise an operator cannot tell
// a slow request from a fast one.
//
// Conditions, and what each one protects:
//  1. `remote_chain` -- a local chain and a whole model never pay the wire cost.
//  2. no grammar -- GBNF legalizes tokens that are not in the top-k set (the `{` of a JSON
//     grammar is frequently outside the top 40), so the k set is not a superset of the legal one.
//  3. no reasoning-budget sampler -- it masks the whole vocabulary outside the chain, so top_k
//     does not truncate it.
//  4. no pre-sampling logprobs -- they are a softmax over the whole row, and there is no row.
//  5. no speculative decoding -- it samples several batch rows, each from its own chain, and the
//     tail answers one row.
//  6. enough history for penalties -- the window comes from this sampler's own last-token ring
//     (n_prev entries); a shorter ring would shorten the penalty history and change the logits.
//  7. the chain actually splits at top_k -- analysed once in common_sampler_init(); this includes
//     top_k > 0 and top_k < n_vocab (at k = n_vocab the "candidates" are bigger than the row).
bool common_sampler_candidates_check(
        const struct common_sampler * gsmpl,
        bool remote_chain,
        bool pre_sampling_logprobs,
        bool speculative,
        common_sampler_candidates & out,
        std::string & why);

// [QLH-SPLIT] v5: the last `penalty_last_n` accepted tokens, oldest first -- the window the tail
// replays into its own penalties sampler. Bounded by the ring the sampler keeps, so this never
// claims a history the sampler itself no longer has. Returns the number written.
uint32_t common_sampler_candidates_history(
        const struct common_sampler * gsmpl,
        std::vector<llama_token> & out);

// [QLH-SPLIT] v7: may this request verify its drafts through the per-row channel?
//
// The channel answers one token id per batch row (see llama_split_request_verify_rows in
// src/llama-ext.h), which is exact only when the token of a row *is* its argmax -- the same
// property, for the same reason, as in common_sampler_candidates_check: a fallback discovered
// after the frame is answered is a wrong token, not a slower one. One function, one place:
//
//  1. `remote_chain` -- a local chain and a whole model hold the rows already.
//  2. no probability request -- pre-sampling probs are a softmax over the row, post-sampling
//     ones are the sampler's candidate array; neither exists without the row.
//  3. the chain samples the argmax -- proven once in common_sampler_init()
//     (verify_plan_chain): a greedy tail and no step before it that can move the maximum.
//
// A grammar does NOT block the channel (v8). It is applied to each row's token *after* the row
// answers (common_sampler_grammar_rejects), and a row it rejects makes common_sampler_sample()
// return LLAMA_TOKEN_NULL: that row is not used, the round ends, and the caller falls back to a
// plain decode, which runs the row path's own grammar resampling. The cost is a lost round, never
// a token the row path would not have produced.
//
// A reasoning-budget sampler does not block it either (v9), on the same terms: its mask is asked
// about the row's token (common_sampler_rbudget_rejects) *after* the row answers, and the question
// is read-only -- the budget is spent only by accept(), which the caller rolls back with the rest
// of the sampler when a row of the frame is refused. The two filters are asked in the order the
// row path applies them: reasoning budget first, then grammar. They do not interact: while the
// reasoning budget is FORCING, a lazy grammar is not applied at all (grammar_should_apply), and a
// non-lazy grammar can only reject a token the mask would have dropped anyway.
bool common_sampler_verify_rows_check(
        const struct common_sampler * gsmpl,
        bool remote_chain,
        bool want_probs,
        std::string & why);

// [QLH-SPLIT] v8: does this sampler's grammar reject `token` from its current state?
//
// Read-only: llama_grammar_apply_impl() only masks, so this never advances the grammar (unlike
// common_sampler_accept(), which throws on a token the grammar cannot advance). A lazy grammar
// whose trigger has not fired masks nothing and answers false, exactly like the row path.
//
// Used by common_sampler_sample(), on both paths: the row path asks it about the token it just
// sampled (and resamples from the row when the answer is yes), the verified-row path about the
// argmax it was handed (and answers LLAMA_TOKEN_NULL instead, having no row).
//
// Not a draft pre-filter: a draft the grammar rejects cannot be *accepted*, but the frame that
// carries it still answers the target's own argmax for its first row, so truncating the draft at
// the first illegal token saves tail compute and loses whatever the later rows would have
// matched -- it does not turn a wasted frame into a used one.
bool common_sampler_grammar_rejects(struct common_sampler * gsmpl, llama_token token);

// [QLH-SPLIT] v8: is a grammar sampler attached to this sampler at all?
bool common_sampler_has_grammar(const struct common_sampler * gsmpl);

// [QLH-SPLIT] v9: does the reasoning-budget sampler reject `token` from its current state?
//
// The one-token form of the mask the row path applies before the chain
// (llama_sampler_apply(gsmpl->rbudget, &cur_p)). Read-only: common_reasoning_budget_rejects only
// reads (state, force_pos, the forced token), so asking it per row neither spends budget nor moves
// the forcing position. Only common_sampler_accept() advances the state machine, which is what
// makes "ask, then accept" and "ask, then roll back" both exact.
//
// Answers false when no reasoning-budget sampler is attached, when the sampler is passing the
// vocabulary through (IDLE / COUNTING / WAITING_UTF8 / DONE), and for the one forced token while
// it is FORCING.
bool common_sampler_rbudget_rejects(struct common_sampler * gsmpl, llama_token token);

// [QLH-SPLIT] v9: is a reasoning-budget sampler attached to this sampler at all?
bool common_sampler_has_reasoning_budget(const struct common_sampler * gsmpl);

// [QLH-SPLIT] v5: sample the next token from a candidate array (sorted by descending logit,
// exactly as the tail produced it) instead of from the logits row.
//
// Does NOT accept the token: the caller does that with common_sampler_accept(), exactly as with
// common_sampler_sample(), so the two paths stay interchangeable at the call site.
// Requires common_sampler_candidates_check() to have returned true for this request.
llama_token common_sampler_sample_candidates(
        struct common_sampler * gsmpl,
        const llama_token * ids,
        const float * logits,
        uint32_t n);

// Helpers

// access the internal list of current candidate tokens
// if do_sort == true, the candidates are guaranteed to be sorted afterwards (in descending order of probability)
// the .sorted flag of the result indicates whether the returned candidates are sorted
llama_token_data_array * common_sampler_get_candidates(struct common_sampler * gsmpl, bool do_sort);

// get the last accepted token
llama_token common_sampler_last(const struct common_sampler * gsmpl);

// print the sampler chain into a string
std::string common_sampler_print(const struct common_sampler * gsmpl);

// get a string representation of the last accepted tokens
std::string common_sampler_prev_str(common_sampler * gsmpl, llama_context * ctx, int n);

char        common_sampler_type_to_chr(enum common_sampler_type cnstr);
std::string common_sampler_type_to_str(enum common_sampler_type cnstr);

std::vector<enum common_sampler_type> common_sampler_types_from_names(const std::vector<std::string> & names);
std::vector<enum common_sampler_type> common_sampler_types_from_chars(const std::string & chars);

// add the strings that are a single token in the vocab to the preserved tokens
void common_sampling_add_preserved_tokens(common_params_sampling & sampling, const llama_vocab * vocab, const std::vector<std::string> & tokens);

// add grammar triggers, a trigger word that is a single token becomes a token trigger and must be a preserved token
void common_sampling_add_grammar_triggers(common_params_sampling & sampling, const llama_vocab * vocab, std::vector<common_grammar_trigger> triggers);

llama_sampler * llama_sampler_init_llg(const llama_vocab * vocab,
                const char * grammar_kind, const char * grammar_data);

struct common_sampler_deleter {
    void operator()(common_sampler * s) { common_sampler_free(s); }
};

typedef std::unique_ptr<common_sampler, common_sampler_deleter> common_sampler_ptr;
