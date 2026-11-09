#include "sampling.h"

#include "../src/llama-ext.h"        // [QLH-SPLIT] v7: llama_split_verified_token (per-row verify channel)
#include "../src/llama-split-proto.h" // [QLH-SPLIT] v5: the candidate step numbers (static_assert only)

#include "common.h"
#include "fit.h"
#include "log.h"
#include "reasoning-budget.h"

#include "ggml.h"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstring>
#include <random>
#include <unordered_map>
#include <vector>

// the ring buffer works similarly to std::deque, but with a fixed capacity
// TODO: deduplicate with llama-impl.h
template<typename T>
struct ring_buffer {
    ring_buffer(size_t cap) : capacity(cap), data(cap) {}

    T & front() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[first];
    }

    const T & front() const {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[first];
    }

    T & back() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[pos];
    }

    const T & back() const {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[pos];
    }

    void push_back(const T & value) {
        if (sz == capacity) {
            // advance the start when buffer is full
            first = (first + 1) % capacity;
        } else {
            sz++;
        }
        data[pos] = value;
        pos = (pos + 1) % capacity;
    }

    T pop_front() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        T value = data[first];
        first = (first + 1) % capacity;
        sz--;
        return value;
    }

    const T & rat(size_t i) const {
        if (i >= sz) {
            throw std::runtime_error("ring buffer: index out of bounds");
        }
        return data[(first + sz - i - 1) % capacity];
    }

    std::vector<T> to_vector() const {
        std::vector<T> result;
        result.reserve(sz);
        for (size_t i = 0; i < sz; i++) {
            result.push_back(data[(first + i) % capacity]);
        }
        return result;
    }

    void clear() {
        // here only reset the status of the buffer
        sz = 0;
        first = 0;
        pos = 0;
    }

    bool empty() const {
        return sz == 0;
    }

    size_t size() const {
        return sz;
    }

    size_t capacity = 0;
    size_t sz = 0;
    size_t first = 0;
    size_t pos = 0;
    std::vector<T> data;
};

struct common_sampler {
    common_params_sampling params;

    struct llama_sampler * grmr;
    struct llama_sampler * rbudget;
    struct llama_sampler * chain;

    ring_buffer<llama_token> prev;

    std::vector<llama_token_data> cur;

    llama_token_data_array cur_p;

    // for rejection sampling; independent of the draft, or the target distribution is not preserved
    std::mt19937 rng;

    // [QLH-SPLIT] v5: remote candidate channel (see sampling.h for the whole story).
    //
    // merged_bias is the exact list the chain's logit_bias step applies. common_sampler_init
    // builds it locally today; keeping it here is what lets the tail apply the same list without
    // common/ or the params.
    std::vector<llama_logit_bias> merged_bias;

    // The analysis of the chain's split point, computed once in common_sampler_init:
    //   cand_first > 0  -- index of the first step to run on candidates (top_k + 1)
    //   cand_full       -- the parameters the tail needs for the steps before it
    //   cand_why        -- why the split is not usable, for the log line
    // A -1 cand_first means "never use the candidate channel with this chain".
    int32_t                   cand_first = -1;
    common_sampler_candidates cand_full;
    std::string               cand_why;

    // [QLH-SPLIT] v7: is this chain "the argmax of the row"?
    //
    // The per-row verify channel answers a token id per row and nothing else, so the caller can
    // only use it when sampling a row *is* taking its argmax: every step before the final greedy
    // selection must leave the maximum where it is (top_k/top_p/min_p/top_n_sigma prune the tail
    // of the distribution, temperature > 0 is monotone, a no-op penalty/DRY step does nothing),
    // and no step may move it (logit_bias, an enabled penalties or DRY step, XTC's random
    // exclusion of the top choice, typical/mirostat/adaptive sampling). Computed once, next to
    // the candidate analysis, so the log line and the decision cannot drift.
    bool        verify_ok = false;
    std::string verify_why;

    void reset() {
        prev.clear();

        llama_sampler_reset(chain);
    }

    void set_logits(struct llama_context * ctx, int idx) {
        const float *       sampled_probs  = llama_get_sampled_probs_ith     (ctx, idx);
        const float *       sampled_logits = llama_get_sampled_logits_ith    (ctx, idx);
        const llama_token * sampled_ids    = llama_get_sampled_candidates_ith(ctx, idx);

        const llama_model * model = llama_get_model(ctx);
        const llama_vocab * vocab = llama_model_get_vocab(model);

        const int n_vocab = llama_vocab_n_tokens(vocab);

        if (sampled_probs) {
            const uint32_t sampled_probs_count = llama_get_sampled_probs_count_ith(ctx, idx);
            cur.resize(sampled_probs_count);
            for (uint32_t i = 0; i < sampled_probs_count; ++i) {
                cur[i] = llama_token_data{sampled_ids[i], sampled_logits[i], sampled_probs[i]};
            }
        } else if (sampled_logits) {
            const uint32_t sampled_logits_count = llama_get_sampled_logits_count_ith(ctx, idx);
            cur.resize(sampled_logits_count);
            for (uint32_t i = 0; i < sampled_logits_count; i++) {
                cur[i] = llama_token_data{sampled_ids[i], sampled_logits[i], 0.0f};
            }
        } else {
            const auto * logits = llama_get_logits_ith(ctx, idx);
            GGML_ASSERT(logits != nullptr);
            cur.resize(n_vocab);
            for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                cur[token_id] = llama_token_data{token_id, logits[token_id], 0.0f};
            }
        }

        cur_p = { cur.data(), cur.size(), -1, false };
    }

    common_time_meas tm() {
        return common_time_meas(t_total_us, params.no_perf);
    }

    mutable int64_t t_total_us = 0;
};

// [QLH-SPLIT] v5: is a penalties step a no-op? Mirrors llama_sampler_penalties::is_disabled()
// exactly, because the decision here is "does the tail have to run it", and a disagreement with
// the library would either drop a real penalty or send parameters nobody uses.
static bool cand_penalties_enabled(const common_params_sampling & p) {
    return p.penalty_last_n != 0 &&
           !(p.penalty_repeat == 1.0f && p.penalty_freq == 0.0f && p.penalty_present == 0.0f);
}

// Same for DRY (llama_sampler_dry_apply's early-out).
static bool cand_dry_enabled(const common_params_sampling & p) {
    return p.dry_multiplier != 0.0f && p.dry_base >= 1.0f && p.dry_penalty_last_n != 0;
}

// [QLH-SPLIT] v5: where can this chain be cut?
//
// The cut is after top_k, and it is exact only if:
//  - there is a top_k step at all, with k > 0 (otherwise nothing truncates the vocabulary and the
//    k candidates are a lossy sample of it -- measured: T=0.8 / top_p=0.95 needs 763 tokens, and
//    at k=64 the tail of the distribution carries 0.185);
//  - every step before it can be reproduced by a stage that only has llama.h (logit_bias is
//    handled outside this list, penalties and top_n_sigma are the two candidates, DRY is only
//    allowed when it is a no-op);
//  - every step after it is fine by construction: they all read cur_p, which holds exactly the k
//    candidates.
//
// Anything else keeps the whole-row path. The alternative -- approximating -- would be a sampling
// bug that no output ever shows.
static bool cand_plan_chain(
        const std::vector<common_sampler_type> & types,
        const common_params_sampling & params,
        int32_t n_vocab,
        int32_t & first,
        common_sampler_candidates & out,
        std::string & why) {
    int32_t ik = -1;
    for (size_t i = 0; i < types.size(); ++i) {
        if (types[i] == COMMON_SAMPLER_TYPE_TOP_K) {
            ik = (int32_t) i;
            break;
        }
    }
    if (ik < 0) {
        why = "the sampler chain has no top_k step (mirostat, or a custom --samplers list without "
              "top-k), so nothing truncates the vocabulary";
        return false;
    }
    if (params.top_k <= 0) {
        why = "top_k <= 0: the chain does not truncate the vocabulary";
        return false;
    }
    if (params.top_k >= n_vocab) {
        // k candidates would be n_vocab * 8 bytes, i.e. twice the row they replace. Still exact,
        // but the whole point is the bytes, so take the row.
        why = "top_k (" + std::to_string(params.top_k) + ") covers the whole vocabulary (" +
              std::to_string(n_vocab) + "): the candidate array would be larger than the row";
        return false;
    }
    if (params.backend_sampling) {
        // The chain lives on the backend as well and already produces candidates there; driving
        // this channel too would have two sources of truth for the same array.
        why = "backend sampling is enabled for this context";
        return false;
    }

    out = common_sampler_candidates{};
    out.n_steps = 0;

    // The wire frame has room for 3 steps (cand_req::steps), and top_k is always the last one, so
    // at most two steps may precede it. A custom --samplers list with more (say penalties twice
    // plus top_n_sigma) is not described by this frame; refuse rather than truncate it, because a
    // truncated chain would produce candidates the row path never would.
    const auto add_step = [&](uint8_t step) {
        if (out.n_steps + 2 > (uint32_t) sizeof(out.steps)) {
            return false;
        }
        out.steps[out.n_steps++] = step;
        return true;
    };

    for (int32_t i = 0; i < ik; ++i) {
        switch (types[i]) {
            case COMMON_SAMPLER_TYPE_PENALTIES:
                if (cand_penalties_enabled(params) && !add_step(COMMON_CAND_STEP_PENALTIES)) {
                    why = "the chain has more steps before top_k than a stage frame can describe";
                    return false;
                }
                break;
            case COMMON_SAMPLER_TYPE_DRY:
                if (cand_dry_enabled(params)) {
                    // DRY needs the vocabulary's token pieces and the configured sequence
                    // breakers; a stage has the model but not the driving side's parameters, and
                    // shipping them per frame would be a second, drifting copy of the sampler.
                    why = "the DRY repetition penalty is enabled and runs before top_k";
                    return false;
                }
                break;
            case COMMON_SAMPLER_TYPE_TOP_N_SIGMA:
                if (params.top_n_sigma > 0.0f) {
                    out.top_n_sigma = params.top_n_sigma;
                    if (!add_step(COMMON_CAND_STEP_TOP_N_SIGMA)) {
                        why = "the chain has more steps before top_k than a stage frame can describe";
                        return false;
                    }
                }
                break;
            default:
                why = std::string("the chain has a step before top_k that a split-stage cannot run: ") +
                      common_sampler_type_to_str(types[i]);
                return false;
        }
    }

    out.steps[out.n_steps++] = COMMON_CAND_STEP_TOP_K;
    out.top_k           = params.top_k;
    out.penalty_last_n  = params.penalty_last_n;
    out.penalty_repeat  = params.penalty_repeat;
    out.penalty_freq    = params.penalty_freq;
    out.penalty_present = params.penalty_present;

    first = ik + 1;
    return true;
}

// [QLH-SPLIT] v7: is this chain "the argmax of the row"?
//
// The per-row verify channel sends one token id per batch row and nothing else, so it is only
// exact when the token a row produces *is* its argmax. That is a property of the whole chain, not
// of the temperature alone:
//
//  - the final selection must be the greedy sampler. common_sampler_init picks it exactly when
//    n_probs == 0 (probs need a distribution) and the chain ends in temp 0 with no dynatemp, or
//    in top_k == 1 -- and not when the adaptive-p alternative was selected, which replaces
//    greedy/dist at the end of the chain and does not take the argmax either. Anything else --
//    dist, mirostat, adaptive_p -- draws, so the token is not a function of the row's maximum.
//  - every step before that must leave the maximum where it is. top_k / top_p / min_p /
//    top_n_sigma only remove the low-probability tail, and a positive temperature is a monotone
//    rescaling, so the argmax survives all of them. A no-op penalties or DRY step changes
//    nothing. This is deliberately a list of *proven* steps: XTC can drop the top choice on
//    purpose, typical inherits a different ranking, and mirostat/adaptive_p replace the sampling
//    rule, so a chain containing them keeps the whole-row path.
//  - no logit_bias step. It is always the first step and its entries are -INFINITY (model
//    suppress tokens) or a user bias; either can move the maximum away from the raw row's.
//
// The check is one function because the fallback is the highest-risk part of the feature: a
// caller with its own copy of these conditions would silently sample from a channel it cannot
// undo. `why` is written to be logged once per request (common/sampling.h).
static bool verify_plan_chain(
        const std::vector<common_sampler_type> & types,
        const common_params_sampling & params,
        bool greedy,
        bool has_bias,
        std::string & why) {
    if (!greedy) {
        why = "the sampler chain does not end in the greedy step (temp 0 with no dynatemp, or "
              "top_k = 1, and no probability request), so a row's token is not its argmax";
        return false;
    }
    if (has_bias) {
        why = "the chain applies a logit_bias step (user biases or the model's suppress tokens), "
              "which can move the argmax away from the raw row's";
        return false;
    }
    for (const auto type : types) {
        switch (type) {
            case COMMON_SAMPLER_TYPE_PENALTIES:
                if (!cand_penalties_enabled(params)) {
                    break; // a no-op step: same logits in, same logits out
                }
                why = "the repetition penalties are enabled, and they run before the argmax";
                return false;
            case COMMON_SAMPLER_TYPE_DRY:
                if (!cand_dry_enabled(params)) {
                    break; // no-op as well
                }
                why = "the DRY repetition penalty is enabled, and it runs before the argmax";
                return false;
            case COMMON_SAMPLER_TYPE_TOP_K:
            case COMMON_SAMPLER_TYPE_TOP_P:
            case COMMON_SAMPLER_TYPE_MIN_P:
            case COMMON_SAMPLER_TYPE_TOP_N_SIGMA:
            case COMMON_SAMPLER_TYPE_TEMPERATURE:
                // Prune the tail, or rescale monotonically; the maximum stays the maximum.
                break;
            case COMMON_SAMPLER_TYPE_TYPICAL_P:
                // llama_sampler_typical_apply returns immediately at p >= 1, which is the default.
                // Below that it keeps the tokens whose log-probability is close to the entropy --
                // a ranking that can leave the maximum out, so it must not be used here.
                if (params.typ_p >= 1.0f) {
                    break;
                }
                why = "typical sampling is enabled (typ_p < 1), and it can drop the argmax";
                return false;
            case COMMON_SAMPLER_TYPE_XTC:
                // llama_sample_xtc_apply is a no-op at probability <= 0 (the default) and at
                // threshold > 0.5. Above that it removes the top choices on purpose, which is
                // exactly the token this channel assumes survives.
                if (params.xtc_probability <= 0.0f || params.xtc_threshold > 0.5f) {
                    break;
                }
                why = "XTC is enabled, and it removes the top choices on purpose";
                return false;
            default:
                why = std::string("the chain has a step that can move the argmax: ") +
                      common_sampler_type_to_str(type);
                return false;
        }
    }
    return true;
}

std::string common_params_sampling::print() const {    char result[1024];

    snprintf(result, sizeof(result),
            "\trepeat_last_n = %d, repeat_penalty = %.3f, frequency_penalty = %.3f, presence_penalty = %.3f\n"
            "\tdry_multiplier = %.3f, dry_base = %.3f, dry_allowed_length = %d, dry_penalty_last_n = %d\n"
            "\ttop_k = %d, top_p = %.3f, min_p = %.3f, xtc_probability = %.3f, xtc_threshold = %.3f, typical_p = %.3f, top_n_sigma = %.3f, temp = %.3f\n"
            "\tmirostat = %d, mirostat_lr = %.3f, mirostat_ent = %.3f, adaptive_target = %.3f, adaptive_decay = %.3f",
            penalty_last_n, penalty_repeat, penalty_freq, penalty_present,
            dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n,
            top_k, top_p, min_p, xtc_probability, xtc_threshold, typ_p, top_n_sigma, temp,
            mirostat, mirostat_eta, mirostat_tau, adaptive_target, adaptive_decay);

    return std::string(result);
}

struct common_sampler * common_sampler_init(
        const struct llama_model * model,
        struct common_params_sampling & params) {
    if (!std::isfinite(params.penalty_repeat) ||
        params.penalty_repeat <= 0.0f ||
        !std::isfinite(1.0f/params.penalty_repeat)) {
        throw std::invalid_argument("penalty_repeat must be finite and greater than 0");
    }
    if (!std::isfinite(params.penalty_freq)) {
        throw std::invalid_argument("penalty_freq must be finite");
    }
    if (!std::isfinite(params.penalty_present)) {
        throw std::invalid_argument("penalty_present must be finite");
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_sampler_chain_params lparams = llama_sampler_chain_default_params();

    lparams.no_perf = params.no_perf;

    llama_sampler * grmr = nullptr;
    llama_sampler * rbudget = nullptr;
    llama_sampler * chain = llama_sampler_chain_init(lparams);

    std::vector<llama_sampler *> samplers;

    // [QLH-SPLIT] v7: how the chain ends, hoisted out of the branch below because the per-row
    // verify analysis needs exactly what the chain was built from.
    //
    // `greedy` is the condition common_sampler_init uses to pick llama_sampler_init_greedy(), and
    // `adaptive_p` says the chain ends in the adaptive-p sampler instead *even when greedy is
    // true* (the two are alternatives in the same if/else, and adaptive-p is registered after the
    // user's sampler list, so it does not appear in s_types at all). Both are needed: a chain
    // ending in adaptive-p does not take the argmax, no matter what `greedy` says.
    bool greedy         = false;
    bool use_adaptive_p = false;

    // [QLH-SPLIT] v5: the type of every entry pushed into `samplers`, parallel to it. The chain
    // itself cannot be asked for this: it exposes lambdas, not names, and a name string is not
    // something a fallback decision should be parsed out of. logit_bias is not in here because it
    // is always first and has no type in this enum; see has_bias below.
    std::vector<common_sampler_type> s_types;
    bool has_bias = false;

    const std::string & grammar_str = common_grammar_value(params.grammar);
    if (grammar_str.compare(0, 11, "%llguidance") == 0) {
#ifdef LLAMA_USE_LLGUIDANCE
        grmr = llama_sampler_init_llg(vocab, "lark", grammar_str.c_str());
#else
        throw std::runtime_error("failed to parse grammar: llguidance is not enabled");
#endif // LLAMA_USE_LLGUIDANCE
    } else {
        std::vector<std::string> trigger_patterns;
        std::vector<llama_token> trigger_tokens;
        for (const auto & trigger : params.grammar_triggers) {
            switch (trigger.type) {
                case COMMON_GRAMMAR_TRIGGER_TYPE_WORD:
                {
                    const auto & word = trigger.value;
                    trigger_patterns.push_back(regex_escape(word));
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN:
                {
                    trigger_patterns.push_back(trigger.value);
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL:
                {
                    const auto & pattern = trigger.value;
                    std::string anchored = "^$";
                    if (!pattern.empty()) {
                        anchored = (pattern.front() != '^' ? "^" : "")
                            + pattern
                            + (pattern.back() != '$' ? "$" : "");
                    }
                    trigger_patterns.push_back(anchored);
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN:
                {
                    const auto token = trigger.token;
                    trigger_tokens.push_back(token);
                    break;
                }
                default:
                    GGML_ASSERT(false && "unknown trigger type");
            }
        }

        std::vector<const char *> trigger_patterns_c;
        trigger_patterns_c.reserve(trigger_patterns.size());
        for (const auto & regex : trigger_patterns) {
            trigger_patterns_c.push_back(regex.c_str());
        }

        if (!grammar_str.empty()) {
             if (params.grammar_lazy) {
                 grmr = llama_sampler_init_grammar_lazy_patterns(vocab, grammar_str.c_str(), "root",
                         trigger_patterns_c.data(), trigger_patterns_c.size(),
                         trigger_tokens.data(), trigger_tokens.size());
             } else {
                 grmr = llama_sampler_init_grammar(vocab, grammar_str.c_str(), "root");
             }
        }
    }
    if (!grmr && !grammar_str.empty()) {
        throw std::runtime_error("failed to parse grammar");
    }

    // Compute prefill tokens from the generation prompt
    std::vector<llama_token> prefill_tokens;
    if (!params.generation_prompt.empty()) {
        GGML_ASSERT(vocab != nullptr);
        auto tokens = common_tokenize(vocab, params.generation_prompt, false, true);
        for (size_t i = 0; i < tokens.size(); i++) {
            std::string piece = common_token_to_piece(vocab, tokens[i], true);
            if (i == 0 && std::isspace(piece[0]) && !std::isspace(params.generation_prompt[0])) {
                // Some tokenizers will add a space before the first special token, need to exclude
                continue;
            }
            LOG_DBG("%s: prefill token: %d = %s\n", __func__, tokens[i], piece.c_str());
            prefill_tokens.push_back(tokens[i]);
        }
    }

    // Feed generation prompt tokens to the grammar sampler so it advances past
    // tokens the template already placed in the prompt.
    // Only applies to output-format and tool-call grammars; user-supplied grammars must not be prefilled.
    if (grmr && !params.grammar_lazy && common_grammar_needs_prefill(params.grammar)) {
        try {
            for (const auto & token : prefill_tokens) {
                llama_sampler_accept(grmr, token);
                LOG_DBG("%s: grammar accepted prefill token (%d)\n", __func__, token);
            }
        } catch (std::exception &e) {
            LOG_ERR("%s: error initializing grammar sampler for grammar:\n%s\n\nGeneration prompt:\n'%s'\n", __func__,
                common_grammar_value(params.grammar).c_str(), params.generation_prompt.c_str());
            throw e;
        }
    }

    // reasoning budget sampler (skip when budget is unlimited unless a lazy grammar is active, which needs rbudget for thinking-block suppression)
    if (!params.reasoning_budget_start.empty() && !params.reasoning_budget_end.empty() && (params.grammar_lazy || params.reasoning_budget_tokens >= 0 || params.reasoning_control)) {
        rbudget = common_reasoning_budget_init(
            vocab,
            {params.reasoning_budget_start},
            params.reasoning_budget_end,
            params.reasoning_budget_forced,
            params.reasoning_budget_tokens < 0 ? INT_MAX : params.reasoning_budget_tokens);

        for (const auto & token : prefill_tokens) {
            llama_sampler_accept(rbudget, token);
            LOG_DBG("%s: reasoning-budget accepted prefill token (%d)\n", __func__, token);
        }
    }

    // logit bias: user biases + model suppress tokens (-INFINITY)
    //
    // [QLH-SPLIT] v5: `merged_bias` is kept (rather than staying local) because the tail has to
    // apply the very same list before its top_k step. These entries are all -INFINITY, and their
    // position before top_k is what makes them matter: applied later, a suppressed token would
    // already have taken a candidate slot the column path would have given to a usable token.
    std::vector<llama_logit_bias> merged_bias = params.logit_bias;
    {
        int32_t n_suppress = 0;
        const llama_token * suppress = llama_vocab_get_suppress_tokens(vocab, &n_suppress);
        for (int32_t i = 0; i < n_suppress; ++i) {
            merged_bias.push_back({ suppress[i], -INFINITY });
        }

        if (!merged_bias.empty()) {
            samplers.push_back(llama_sampler_init_logit_bias(llama_vocab_n_tokens(vocab), merged_bias.size(), merged_bias.data()));
            has_bias = true;
        }
    }

    if (params.mirostat == 0) {

        // [QLH-SPLIT] v7: hoisted out of the branch below because the per-row verify analysis
        // needs it too, and it must be the same boolean the chain itself was built from.
        greedy = params.n_probs == 0 && !params.samplers.empty() &&
            ((params.samplers.back() == COMMON_SAMPLER_TYPE_TEMPERATURE && params.temp == 0.0f && params.dynatemp_range == 0.0f) ||
             (params.samplers.back() == COMMON_SAMPLER_TYPE_TOP_K && params.top_k == 1));

        for (const auto & cnstr : params.samplers) {
            // [QLH-SPLIT] v5: record which step each push is, so the candidate split can be
            // analysed without parsing sampler names. Every case below adds at most one sampler.
            const size_t n_before = samplers.size();

            switch (cnstr) {
                case COMMON_SAMPLER_TYPE_DRY:
                    {
                        std::vector<const char *> c_breakers;
                        c_breakers.reserve(params.dry_sequence_breakers.size());
                        for (const auto & str : params.dry_sequence_breakers) {
                            c_breakers.push_back(str.c_str());
                        }
                        samplers.push_back(llama_sampler_init_dry(vocab, params.dry_multiplier, params.dry_base, params.dry_allowed_length, params.dry_penalty_last_n, c_breakers.data(), c_breakers.size()));
                    }
                    break;
                case COMMON_SAMPLER_TYPE_TOP_K:
                    samplers.push_back(llama_sampler_init_top_k(params.top_k));
                    break;
                case COMMON_SAMPLER_TYPE_TOP_P:
                    samplers.push_back(llama_sampler_init_top_p(params.top_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_TOP_N_SIGMA:
                    samplers.push_back(llama_sampler_init_top_n_sigma(params.top_n_sigma));
                    break;
                case COMMON_SAMPLER_TYPE_MIN_P:
                    samplers.push_back(llama_sampler_init_min_p(params.min_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_XTC:
                    samplers.push_back(llama_sampler_init_xtc(params.xtc_probability, params.xtc_threshold, params.min_keep, params.seed));
                    break;
                case COMMON_SAMPLER_TYPE_TYPICAL_P:
                    samplers.push_back(llama_sampler_init_typical(params.typ_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_TEMPERATURE:
                    samplers.push_back(llama_sampler_init_temp_ext(params.temp, params.dynatemp_range, params.dynatemp_exponent));
                    break;
                case COMMON_SAMPLER_TYPE_INFILL:
                    samplers.push_back(llama_sampler_init_infill(vocab));
                    break;
                case COMMON_SAMPLER_TYPE_PENALTIES:
                    samplers.push_back(llama_sampler_init_penalties(llama_vocab_n_tokens(vocab), params.penalty_last_n, params.penalty_repeat, params.penalty_freq, params.penalty_present));
                    break;
                case COMMON_SAMPLER_TYPE_ADAPTIVE_P:
                    // the `adaptive-p` sampler is like `dist` and `mirostat` in that it selects
                    // a single token, so we will add `dist` at the end of the chain by default,
                    // unless the user specifically included `adaptive-p`. we set this flag here
                    // so we know to add the sampler at the very end.
                    use_adaptive_p = true;
                    break;
                default:
                    GGML_ASSERT(false && "unknown sampler type");
            }

            for (size_t i = n_before; i < samplers.size(); ++i) {
                s_types.push_back(cnstr);
            }
        }
        if (use_adaptive_p) {
            // only if user explicitly included adaptive-p sampler
            samplers.push_back(llama_sampler_init_adaptive_p(params.adaptive_target, params.adaptive_decay, params.seed));
        } else {
            // Keep distribution sampling when callers request probabilities.
            samplers.push_back(greedy ? llama_sampler_init_greedy() : llama_sampler_init_dist(params.seed));
        }
    } else if (params.mirostat == 1) {
        samplers.push_back(llama_sampler_init_temp(params.temp));
        samplers.push_back(llama_sampler_init_mirostat(llama_vocab_n_tokens(vocab), params.seed, params.mirostat_tau, params.mirostat_eta, 100));
    } else if (params.mirostat == 2) {
        samplers.push_back(llama_sampler_init_temp(params.temp));
        samplers.push_back(llama_sampler_init_mirostat_v2(params.seed, params.mirostat_tau, params.mirostat_eta));
    } else {
        GGML_ASSERT(false && "unknown mirostat version");
    }

    for (auto * smpl : samplers) {
        llama_sampler_chain_add(chain, smpl);
    }

    // [QLH-SPLIT] v5: analyse the split point once. A chain that cannot be split keeps the
    // whole-row path for its whole life; the reason is remembered rather than rebuilt per request
    // so the log line and the decision cannot drift.
    int32_t cand_first = -1;
    common_sampler_candidates cand_full;
    std::string cand_why;
    if (cand_plan_chain(s_types, params, llama_vocab_n_tokens(vocab), cand_first, cand_full, cand_why)) {
        // s_types covers only the samplers built by the loop above; the logit_bias step, when
        // present, sits at the very front of the chain and is not in it.
        cand_first += has_bias ? 1 : 0;
    } else {
        cand_first = -1;
    }

    // [QLH-SPLIT] v7: the same analysis for the per-row verify channel. Independent of the
    // candidate split on purpose: the two answer different questions, and a chain can be usable
    // for one and not the other (temp 0 has no top_k requirement here, and top_k > 0 with a
    // non-greedy tail is usable there and not here).
    std::string verify_why;
    bool        verify_ok = verify_plan_chain(s_types, params, greedy && !use_adaptive_p,
                                               has_bias, verify_why);
    if (verify_ok && params.backend_sampling) {
        // The backend sampler would produce the token itself, on a context whose rows this side
        // does not have; and the channel's whole contract is that the value read per row is the
        // argmax of that row.
        verify_ok  = false;
        verify_why = "backend sampling is enabled for this request";
    }

    if (grmr && params.backend_sampling) {
        LOG_WRN("%s: backend sampling is not compatible with grammar, disabling\n", __func__);

        params.backend_sampling = false;
    }

    if (rbudget && params.backend_sampling) {
        LOG_WRN("%s: backend sampling is not compatible with reasoning budget, disabling\n", __func__);

        params.backend_sampling = false;
    }

    auto * result = new common_sampler {
        /* .params  = */ params,
        /* .grmr    = */ grmr,
        /* .rbudget = */ rbudget,
        /* .chain   = */ chain,
        /* .prev    = */ ring_buffer<llama_token>(std::max(32, params.n_prev)),
        /* .cur     = */ {},
        /* .cur_p   = */ {},
        // mix it, the chain and the draft are seeded from this one too
        /* .rng     = */ std::mt19937(llama_sampler_get_seed(chain) ^ 0x9e3779b9u),
        /* .merged_bias = */ std::move(merged_bias),
        /* .cand_first  = */ cand_first,
        /* .cand_full   = */ cand_full,
        /* .cand_why    = */ std::move(cand_why),
        /* .verify_ok   = */ verify_ok,
        /* .verify_why  = */ std::move(verify_why),
    };

    return result;
}

void common_sampler_free(struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return;
    }

    llama_sampler_free(gsmpl->grmr);
    llama_sampler_free(gsmpl->rbudget);
    llama_sampler_free(gsmpl->chain);

    delete gsmpl;
}

static bool grammar_should_apply(struct common_sampler * gsmpl) {
    if (!gsmpl->grmr) {
        return false;
    }
    if (!gsmpl->rbudget) {
        return true;
    }
    if (gsmpl->params.grammar_lazy) {
        // if grammar is lazy, only apply when reasoning budget is not active
        const auto state = common_reasoning_budget_get_state(gsmpl->rbudget);
        return state == REASONING_BUDGET_IDLE || state == REASONING_BUDGET_DONE;
    }
    return true;
}

// [QLH-SPLIT] v8: does this sampler's grammar reject `token` from its *current* state?
//
// The single-token form of the rejection check that common_sampler_sample() does on the row path
// (mask a one-element array, look for -INFINITY), factored out because three places now ask the
// same question of the same state: the verified-row path (per row, below), the row path itself,
// and the server's draft pre-filter.
//
// Two properties make this safe to call at any time, and both matter:
//  - it never advances the grammar. llama_grammar_apply_impl() only writes -INFINITY into the
//    candidate array, so the state is untouched -- unlike accept, which throws on a token the
//    grammar cannot advance (llama_grammar_accept_token, src/llama-grammar.cpp).
//  - a lazy grammar whose trigger has not fired yet masks nothing, so every token "passes". That
//    is exactly what the row path concludes too: until the trigger fires the grammar is not
//    constraining anything, and the trigger can only fire on a token that was accepted.
bool common_sampler_grammar_rejects(struct common_sampler * gsmpl, llama_token token) {
    if (gsmpl == nullptr || !grammar_should_apply(gsmpl)) {
        return false;
    }

    llama_token_data       single_token_data       = { token, 1.0f, 0.0f };
    llama_token_data_array single_token_data_array = { &single_token_data, 1, -1, false };

    llama_sampler_apply(gsmpl->grmr, &single_token_data_array);

    return single_token_data_array.data[0].logit == -INFINITY;
}

bool common_sampler_has_grammar(const struct common_sampler * gsmpl) {
    return gsmpl != nullptr && gsmpl->grmr != nullptr;
}

// [QLH-SPLIT] v9: the same one-token question for the reasoning-budget sampler.
//
// The row path applies this sampler to the whole row before the chain
// (llama_sampler_apply(gsmpl->rbudget, &cur_p) in common_sampler_sample), so the question "would
// the row path have been able to select this token" has exactly one honest form: does the mask of
// the *current* state drop it. common_reasoning_budget_rejects answers that without touching the
// state machine.
//
// Why not apply() on a one-element array, as the grammar check does: apply() is a pure mask today,
// so both would answer the same, but rbudget's state machine lives in accept() (remaining--,
// FORCING -> DONE, the UTF-8 wait) and the check would silently depend on apply() never growing a
// side effect. The explicit predicate cannot drift that way.
//
// Note what this function deliberately does *not* do: it does not spend budget and does not move
// force_pos. A row that passes is accepted by common_sampler_sample_and_accept_n(), and that call
// is the only place the state machine advances -- so a rolled-back frame costs nothing and a frame
// that is fully accepted is charged exactly once per row.
bool common_sampler_rbudget_rejects(struct common_sampler * gsmpl, llama_token token) {
    if (gsmpl == nullptr || gsmpl->rbudget == nullptr) {
        return false;
    }

    return common_reasoning_budget_rejects(gsmpl->rbudget, token);
}

bool common_sampler_has_reasoning_budget(const struct common_sampler * gsmpl) {
    return gsmpl != nullptr && gsmpl->rbudget != nullptr;
}

void common_sampler_accept(struct common_sampler * gsmpl, llama_token token, bool is_generated) {
    if (!gsmpl) {
        return;
    }

    const auto tm = gsmpl->tm();

    // grammar_should_apply() checks the reasoning budget state, so calculate this before we accept
    const auto accept_grammar = is_generated && grammar_should_apply(gsmpl);

    if (gsmpl->rbudget && is_generated) {
        llama_sampler_accept(gsmpl->rbudget, token);

        // if done, replay end sequence which may contain a grammar trigger
        const bool is_done = common_reasoning_budget_get_state(gsmpl->rbudget) == REASONING_BUDGET_DONE;
        if (gsmpl->grmr && !accept_grammar && is_done) {
            const llama_tokens * end_seq = common_reasoning_budget_get_end_match(gsmpl->rbudget);
            if (end_seq) {
                for (const llama_token end_token : *end_seq) {
                    llama_sampler_accept(gsmpl->grmr, end_token);
                }
            }
        }
    }

    if (gsmpl->grmr && accept_grammar) {
        llama_sampler_accept(gsmpl->grmr, token);
    }

    llama_sampler_accept(gsmpl->chain, token);

    gsmpl->prev.push_back(token);
}

void common_sampler_reset(struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return;
    }

    gsmpl->reset();
}

struct common_sampler * common_sampler_clone(common_sampler * gsmpl) {
    return new common_sampler {
        /* .params  = */ gsmpl->params,
        /* .grmr    = */ llama_sampler_clone(gsmpl->grmr),
        /* .rbudget = */ llama_sampler_clone(gsmpl->rbudget),
        /* .chain   = */ llama_sampler_clone(gsmpl->chain),
        /* .prev    = */ gsmpl->prev,
        /* .cur     = */ gsmpl->cur,
        /* .cur_p   = */ gsmpl->cur_p,
        /* .rng     = */ gsmpl->rng,
        /* .merged_bias = */ gsmpl->merged_bias,
        /* .cand_first  = */ gsmpl->cand_first,
        /* .cand_full   = */ gsmpl->cand_full,
        /* .cand_why    = */ gsmpl->cand_why,
        /* .verify_ok   = */ gsmpl->verify_ok,
        /* .verify_why  = */ gsmpl->verify_why,
    };
}

void common_sampler_copy(const common_sampler * src, common_sampler * dst) {
    if (!src || !dst || src == dst) {
        return;
    }

    GGML_ASSERT((src->grmr == nullptr) == (dst->grmr == nullptr));
    GGML_ASSERT((src->rbudget == nullptr) == (dst->rbudget == nullptr));

    llama_sampler_copy(src->grmr,    dst->grmr);
    llama_sampler_copy(src->rbudget, dst->rbudget);
    llama_sampler_copy(src->chain,   dst->chain);

    dst->params     = src->params;
    dst->prev       = src->prev;
    dst->cur        = src->cur;
    dst->cur_p      = src->cur_p;
    dst->cur_p.data = src->cur_p.data ? dst->cur.data() : nullptr; // re-point to dst's buffer
    dst->rng        = src->rng;
    dst->t_total_us = src->t_total_us;

    // [QLH-SPLIT] v5: the candidate plan. The biases pointer mirrors merged_bias, so it is never
    // copied -- it is re-pointed below, or a checkpoint restore would hand the caller the other
    // sampler's list.
    dst->merged_bias = src->merged_bias;
    dst->cand_first  = src->cand_first;
    dst->cand_full   = src->cand_full;
    dst->cand_why    = src->cand_why;
    dst->cand_full.biases = nullptr;

    // [QLH-SPLIT] v7: the verify analysis is a pure function of the chain's construction, so a
    // copy agrees with the source by definition; copying it keeps a restored sampler's plan
    // identical to the one the request started with.
    dst->verify_ok  = src->verify_ok;
    dst->verify_why = src->verify_why;
}

void common_perf_print(const struct llama_context * ctx, const struct common_sampler * gsmpl) {
    // TODO: measure grammar performance

    const double t_sampling_ms = gsmpl ? 1e-3*gsmpl->t_total_us : 0;

    llama_perf_sampler_data data_smpl;
    llama_perf_context_data data_ctx;

    memset(&data_smpl, 0, sizeof(data_smpl));
    memset(&data_ctx,  0, sizeof(data_ctx));

    if (gsmpl) {
        auto & data = data_smpl;

        data = llama_perf_sampler(gsmpl->chain);

        // note: the sampling time includes the samplers time + extra time spent in common/sampling
        LOG_INF("%s:    sampling time = %10.2f ms\n", __func__, t_sampling_ms);
        LOG_INF("%s:    samplers time = %10.2f ms / %5d tokens\n", __func__, data.t_sample_ms, data.n_sample);
    }

    if (ctx) {
        auto & data = data_ctx;

        data = llama_perf_context(ctx);

        const double t_end_ms = 1e-3 * ggml_time_us();

        const double t_total_ms = t_end_ms - data.t_start_ms;
        const double t_unacc_ms = t_total_ms - (t_sampling_ms + data.t_p_eval_ms + data.t_eval_ms);
        const double t_unacc_pc = 100.0 * t_unacc_ms /  t_total_ms;

        LOG_INF("%s:        load time = %10.2f ms\n", __func__, data.t_load_ms);
        LOG_INF("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                __func__, data.t_p_eval_ms, data.n_p_eval, data.t_p_eval_ms / data.n_p_eval, 1e3 / data.t_p_eval_ms * data.n_p_eval);
        LOG_INF("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
                __func__, data.t_eval_ms, data.n_eval, data.t_eval_ms / data.n_eval, 1e3 / data.t_eval_ms * data.n_eval);
        LOG_INF("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (t_end_ms - data.t_start_ms), (data.n_p_eval + data.n_eval));
        LOG_INF("%s: unaccounted time = %10.2f ms / %5.1f %%      (total - sampling - prompt eval - eval) / (total)\n", __func__, t_unacc_ms, t_unacc_pc);
        LOG_INF("%s:    graphs reused = %10d\n", __func__, data.n_reused);
        LOG_INF("%s:   graph reallocs = %10d    / %5d        (same size) / (changed size)\n", __func__, data.n_reallocs_same, data.n_reallocs_diff);

        common_memory_breakdown_print(ctx);
    }
}

struct llama_sampler * common_sampler_get(const struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return nullptr;
    }

    return gsmpl->chain;
}

// [QLH-SPLIT] v5: the step numbers are a wire format (qlh_split::cand_step). Keeping the
// assertion here, next to the code that fills the frame, is what makes a renumbering on one side
// a build error instead of a stage that runs the wrong sampler.
static_assert(COMMON_CAND_STEP_PENALTIES   == qlh_split::CAND_STEP_PENALTIES,   "cand step drift");
static_assert(COMMON_CAND_STEP_TOP_N_SIGMA == qlh_split::CAND_STEP_TOP_N_SIGMA, "cand step drift");
static_assert(COMMON_CAND_STEP_TOP_K       == qlh_split::CAND_STEP_TOP_K,       "cand step drift");

bool common_sampler_candidates_check(
        const struct common_sampler * gsmpl,
        bool remote_chain,
        bool pre_sampling_logprobs,
        bool speculative,
        common_sampler_candidates & out,
        std::string & why) {
    why.clear();
    if (gsmpl == nullptr) {
        why = "no sampler";
        return false;
    }
    if (!remote_chain) {
        // Not a fallback in the slow sense: a local chain and a whole model have the row in
        // memory, so the candidate channel would only add a hop through the wire.
        why = "not a remote chain";
        return false;
    }
    if (gsmpl->grmr != nullptr) {
        why = "grammar is set: GBNF can legalize tokens outside the top-k set";
        return false;
    }
    if (gsmpl->rbudget != nullptr) {
        why = "the reasoning-budget sampler masks the whole vocabulary outside the chain";
        return false;
    }
    if (pre_sampling_logprobs) {
        why = "pre-sampling logprobs were requested: they are a softmax over the whole row";
        return false;
    }
    if (speculative) {
        why = "speculative decoding samples several batch rows per step";
        return false;
    }
    if (cand_penalties_enabled(gsmpl->params) &&
        gsmpl->params.penalty_last_n > gsmpl->params.n_prev) {
        // The window the tail replays comes from this sampler's own last-token ring, which holds
        // n_prev tokens. A shorter window would penalize fewer tokens than the row path does.
        why = "penalty_last_n (" + std::to_string(gsmpl->params.penalty_last_n) +
              ") exceeds n_prev (" + std::to_string(gsmpl->params.n_prev) +
              "): the history window would be truncated";
        return false;
    }
    if (gsmpl->cand_first <= 0) {
        why = gsmpl->cand_why.empty() ? "this sampler chain cannot be split at top_k" : gsmpl->cand_why;
        return false;
    }

    out = gsmpl->cand_full;
    out.biases = &gsmpl->merged_bias;
    return true;
}

bool common_sampler_verify_rows_check(
        const struct common_sampler * gsmpl,
        bool remote_chain,
        bool want_probs,
        std::string & why) {
    why.clear();
    if (gsmpl == nullptr) {
        why = "no sampler";
        return false;
    }
    if (!remote_chain) {
        // Nothing to save: the rows are already in this process's memory.
        why = "not a remote chain";
        return false;
    }
    if (want_probs) {
        // Both kinds of probability output read the row: pre-sampling logprobs are a softmax over
        // the whole vocabulary, post-sampling ones are the sampler's own cur_p, which the
        // token-id path does not build (there is no array to build it from). Either way the
        // request cannot be answered from one argmax per row.
        why = "probabilities were requested: they need the whole row (pre-sampling) or the "
              "sampler's candidate array (post-sampling)";
        return false;
    }
    // [QLH-SPLIT] v8: a grammar no longer blocks this channel, and it is worth stating why the
    // argument that used to be here no longer holds. A grammar can reject the argmax and, on the
    // row path, resample from the row -- which a verified row cannot do. But it does not have to:
    // the verified path asks the grammar about the one token it was handed
    // (common_sampler_grammar_rejects) and answers "this row cannot be used" when the grammar says
    // no (LLAMA_TOKEN_NULL from common_sampler_sample). The caller then restores its checkpoint
    // and decodes that position through the ordinary row path, which is the same resampling --
    // late, and one round more expensive, but never a different token. A lazy grammar needs no
    // special case either: until its trigger fires it masks nothing, and a draft that fires the
    // trigger is accepted and then constrains the next row exactly as on the row path.
    //
    // [QLH-SPLIT] v9: the reasoning-budget sampler does not block it either, for the same reason
    // and with one extra fact that had to be established first: its mask is a pure function of
    // (state, force_pos) and only accept() advances the state machine, so a check can ask it per
    // row without consuming budget (common_reasoning_budget_rejects, common/reasoning-budget.cpp).
    // Both filters are therefore asked about the row's token by common_sampler_sample(), in the
    // order the row path applies them (reasoning budget first, then the grammar), and a row either
    // one rejects makes the frame unusable. See common_sampler_rbudget_rejects for the read-only
    // argument; the budget itself is only spent by common_sampler_accept(), which the accept loop
    // calls per accepted row and the rollback path undoes through common_sampler_copy().
    if (!gsmpl->verify_ok) {
        why = gsmpl->verify_why.empty()
                ? "this sampler chain can move the argmax, so a row's token is not a function of it"
                : gsmpl->verify_why;
        return false;
    }
    return true;
}

uint32_t common_sampler_candidates_history(
        const struct common_sampler * gsmpl,
        std::vector<llama_token> & out) {    out.clear();
    if (gsmpl == nullptr) {
        return 0;
    }
    const int32_t want = std::max(0, gsmpl->params.penalty_last_n);
    const int32_t have = (int32_t) gsmpl->prev.size();
    const int32_t n    = std::min(want, have);

    out.reserve((size_t) n);
    // prev.rat(0) is the newest token; the tail replays oldest-first, exactly like a sequence of
    // llama_sampler_accept() calls would have arrived.
    for (int32_t i = n - 1; i >= 0; --i) {
        out.push_back(gsmpl->prev.rat((size_t) i));
    }
    return (uint32_t) out.size();
}

llama_token common_sampler_sample_candidates(
        struct common_sampler * gsmpl,
        const llama_token * ids,
        const float * logits,
        uint32_t n) {
    GGML_ASSERT(gsmpl != nullptr);
    GGML_ASSERT(gsmpl->cand_first > 0 && "candidates require common_sampler_candidates_check()");
    GGML_ASSERT(ids != nullptr && logits != nullptr && n > 0);

    const auto tm = gsmpl->tm();

    // The candidate array becomes cur_p exactly as the tail built it: descending logit, original
    // f32 values, p unset. The row path leaves the same thing behind its top_k step, which is why
    // the remaining steps below cannot tell the two apart.
    auto & cur = gsmpl->cur;
    cur.resize(n);
    for (uint32_t i = 0; i < n; ++i) {
        cur[i] = llama_token_data{ids[i], logits[i], 0.0f};
    }
    gsmpl->cur_p = { cur.data(), cur.size(), -1, true };

    // Every step from the one after top_k onwards, applied to the candidate array. The samplers
    // this skips are the ones the tail already ran: penalties, top_n_sigma, top_k itself. Running
    // penalties again here would apply the same history twice -- the single most likely way to
    // make the two paths disagree, which is why the split point is computed from the chain rather
    // than assumed to be "penalties first".
    auto & chain = gsmpl->chain;
    const int32_t n_chain = llama_sampler_chain_n(chain);
    for (int32_t i = gsmpl->cand_first; i < n_chain; ++i) {
        struct llama_sampler * s = llama_sampler_chain_get(chain, i);
        GGML_ASSERT(s != nullptr);
        llama_sampler_apply(s, &gsmpl->cur_p);
    }

    GGML_ASSERT(gsmpl->cur_p.selected >= 0 && gsmpl->cur_p.selected < (int32_t) gsmpl->cur_p.size);

    return gsmpl->cur_p.data[gsmpl->cur_p.selected].id;
}

llama_token common_sampler_sample(struct common_sampler * gsmpl, struct llama_context * ctx, int idx, bool grammar_first) {
    llama_synchronize(ctx);

    // start measuring sampling time after the llama_context synchronization in order to not measure any ongoing async operations
    const auto tm = gsmpl->tm();

    llama_token id = LLAMA_TOKEN_NULL;

    // [QLH-SPLIT] v7: the per-row verify channel.
    //
    // When the frame was answered with one argmax token per row instead of logits, there is
    // nothing to sample *from*: for a chain that samples the argmax (verify_ok, proven at init and
    // gated per request by common_sampler_verify_rows_check) the tail already produced exactly
    // the token this call would have returned. Taking it here and not in
    // common_sampler_sample_and_accept_n is what makes every caller of a verified frame correct
    // by construction: the speculative accept loop, the replay path and the synthetic path all
    // reach the sampler through this one function, and none of them needs to know the channel
    // exists.
    //
    // What is *not* done here is deliberate: no set_logits (there is no row), no chain apply (a
    // greedy chain's apply is the argmax the tail already took), and no accept (the caller owns
    // that, exactly as for the row path). cur_p keeps the previous call's contents; every reader
    // of it -- post-sampling probs, rejection sampling -- is excluded by the same check that
    // armed this channel.
    {
        const llama_token verified = llama_split_verified_token(ctx, idx);
        if (verified != LLAMA_TOKEN_NULL) {
            GGML_ASSERT(gsmpl->verify_ok &&
                    "a per-row verified token is only valid for a chain that samples the argmax");

            // [QLH-SPLIT] v8: the grammar is the last filter before a token becomes output, and
            // the row path below applies it after sampling: a token the grammar rejects is not
            // returned, it is resampled from the row with the grammar applied first. This path
            // has no row, so the same question has a different answer -- "this row cannot be
            // used" -- and LLAMA_TOKEN_NULL is how this function says so. It is NOT a token:
            // the caller must not accept it (llama_grammar_accept_token throws on a token the
            // grammar cannot advance) and must not resample from a row it does not have.
            //
            // [QLH-SPLIT] v9: the reasoning-budget sampler is asked first, because that is the
            // order the row path applies them (rbudget, then the grammar) and the question is the
            // same shape: the row path masks the row with rbudget before the chain runs, so a
            // masked token can never be selected. Asking it here does not consume budget or move
            // the forcing position -- common_reasoning_budget_rejects is a pure read -- and when
            // this sampler is FORCING the grammar is not applied at all (grammar_should_apply),
            // so grammar_rejects answers false here exactly as the row path would.
            //
            // The state the question is asked in is the frame's: rows are accepted in order by
            // common_sampler_sample_and_accept_n(), so row i is checked with the grammar and the
            // reasoning budget already advanced by rows 0..i-1. That is also why a rejected row
            // ends the whole round instead of shortening it -- see the comment in
            // sample_and_accept_n. And that is why the caller's rollback must restore rbudget
            // too: the rows accepted before the rejection spent budget and moved force_pos.
            if (common_sampler_rbudget_rejects(gsmpl, verified)) {
                return LLAMA_TOKEN_NULL;
            }

            if (common_sampler_grammar_rejects(gsmpl, verified)) {
                return LLAMA_TOKEN_NULL;
            }

            return verified;
        }
    }

    auto & grmr  = gsmpl->grmr;
    auto & rbudget = gsmpl->rbudget;
    auto & chain = gsmpl->chain;
    auto & cur_p = gsmpl->cur_p; // initialized by set_logits

    gsmpl->set_logits(ctx, idx);

    // Check if a backend sampler has already sampled a token in which case we
    // return that token id directly.
    {
        id = llama_get_sampled_token_ith(ctx, idx);

        if (id != LLAMA_TOKEN_NULL) {
            LOG_DBG("%s: Backend sampler selected token: '%d'. Will not run any CPU samplers\n", __func__, id);

            GGML_ASSERT(!gsmpl->grmr    && "using grammar in combination with backend sampling is not supported");
            GGML_ASSERT(!gsmpl->rbudget && "using reasoning budget in combination with backend sampling is not supported");

            for (size_t i = 0; i < cur_p.size; ++i) {
                if (cur_p.data[i].id == id) {
                    cur_p.selected = i;
                    break;
                }
            }

            return id;
        }
    }

    // apply reasoning budget first
    llama_sampler_apply(rbudget, &cur_p);

    if (grammar_first && grammar_should_apply(gsmpl)) {
        llama_sampler_apply(grmr, &cur_p);
    }

    llama_sampler_apply(chain, &cur_p);

    id = cur_p.data[cur_p.selected].id;

    if (grammar_first || !grammar_should_apply(gsmpl)) {
        return id;
    }

    // check if it the sampled token fits the grammar (grammar-based rejection sampling)
    if (!common_sampler_grammar_rejects(gsmpl, id)) {
        return id;
    }

    // resampling:
    // if the token is not valid, sample again, but first apply the grammar sampler and then the sampling chain
    gsmpl->set_logits(ctx, idx);

    llama_sampler_apply(rbudget,  &cur_p);

    if (grammar_should_apply(gsmpl)) {
        llama_sampler_apply(grmr,  &cur_p);
    }

    llama_sampler_apply(chain, &cur_p);

    GGML_ASSERT(cur_p.selected != -1 && "no selected token during sampling - check your sampling configuration");

    id = cur_p.data[cur_p.selected].id;

    return id;
}

std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, bool grammar_first) {
    GGML_ASSERT(idxs.size() == draft.size() + 1 && "idxs.size() must be draft.size() + 1");

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));

    size_t i = 0;
    for (; i < draft.size(); i++) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        if (id == LLAMA_TOKEN_NULL) {
            // [QLH-SPLIT] v8/v9: the grammar or the reasoning budget rejected this row's argmax,
            // and a verified row has no row to resample from. The frame is abandoned whole rather
            // than shortened, and nothing is accepted for it:
            //  - accepting the row's token would make llama_grammar_accept_token throw, and would
            //    charge the reasoning budget for a token the request never emitted;
            //  - returning the rows that did match would hand the caller a replay frame whose
            //    last row is this same rejected position, so every retry would reject it again
            //    (a spin that never emits a token) -- the round has to end *before* this
            //    position, which only the caller can arrange (restore its checkpoint, then
            //    decode the position plainly so the row path's own grammar resampling runs).
            // An empty result is the signal for that: "no token of this frame is usable".
            //
            // The rows already accepted above advanced the grammar and spent budget; the caller's
            // rollback restores both (common_sampler_copy), so the state after this frame is the
            // state before it -- the frame is as if it had never been generated.
            result.clear();
            return result;
        }

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);

        // do not accept draft tokens after an EOG - they are not output but would stay in the context
        // on replay the last token is from the target and can be EOG, so a trailing EOG is still accepted
        if (draft[i] != id || (llama_vocab_is_eog(vocab, id) && i + 1 < draft.size())) {
            break;
        }
    }

    if (i == draft.size()) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        if (id == LLAMA_TOKEN_NULL) {
            // same rejection, on the bonus row: it is the one row of the frame that is not a
            // draft, and it is still a verified row with no logits behind it.
            result.clear();
            return result;
        }

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);
    }

    return result;
}

static float prob_of(const llama_token_data * data, size_t n, llama_token id) {
    for (size_t k = 0; k < n; ++k) {
        if (data[k].id == id) {
            return data[k].p;
        }
    }
    return 0.0f;
}

// Accept a drafted token with probability min(1, p/q), else draw from norm(max(0, p - q)).
// Preserves the target distribution exactly, and accepts more often than matching does when the
// draft samples instead of taking its argmax.
std::vector<llama_token> common_sampler_sample_and_accept_n_rejection(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, const std::vector<std::vector<llama_token_data>> & draft_q, bool grammar_first) {
    GGML_ASSERT(idxs.size()    == draft.size() + 1 && "idxs.size() must be draft.size() + 1");
    GGML_ASSERT(draft_q.size() == draft.size() && "draft_q must have one entry per draft token");

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    // draws come from the sampler's own stream, so they stay independent of what was drafted
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    std::vector<llama_token_data> residual;

    std::vector<llama_token_data> cand; // candidate array masked by the grammar, if there is one

    size_t i = 0;
    for (; i < draft.size(); i++) {
        // leaves the target distribution in the candidate array
        const llama_token id_tgt = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        if (id_tgt == LLAMA_TOKEN_NULL) {
            // [QLH-SPLIT] v8/v9: same "this frame is unusable" signal as the plain version above.
            // Not reachable today (a request whose grammar or reasoning budget could reject a
            // verified row is greedy, and the rejection path needs temp > 0 — and the server
            // blocks exactly this combination in plan_remote_verify_rows), but the return value is
            // a contract and quietly accepting an unverified draft token here is the one mistake
            // that ends in a throw or in a budget charged twice instead of a wrong token.
            result.clear();
            return result;
        }

        const auto * cur_p = common_sampler_get_candidates(gsmpl, true);
        const auto & q     = draft_q[i];

        const bool masked = !grammar_first && grammar_should_apply(gsmpl);
        if (masked) {
            cand.assign(cur_p->data, cur_p->data + cur_p->size);
            llama_token_data_array arr = { cand.data(), cand.size(), -1, false };
            llama_sampler_apply(gsmpl->grmr, &arr);
        }

        // a candidate the grammar rejects carries no probability, whatever the target thinks
        auto p_raw = [&](size_t k) {
            return masked && cand[k].logit == -INFINITY ? 0.0f : cur_p->data[k].p;
        };

        // masking drops probability mass, so rescale what is left or the residual is over-weighted
        float p_sum = 0.0f;
        if (masked) {
            for (size_t k = 0; k < cur_p->size; ++k) {
                p_sum += p_raw(k);
            }
        }

        const float p_norm = masked && p_sum > 0.0f ? 1.0f/p_sum : 1.0f;

        auto p_of = [&](size_t k) {
            return p_raw(k)*p_norm;
        };

        // q_x is never 0 for a token the draft produced, but guard the divide
        const float q_x = prob_of(q.data(), q.size(), draft[i]);

        float p_x = 0.0f;
        for (size_t k = 0; k < cur_p->size; ++k) {
            if (cur_p->data[k].id == draft[i]) {
                p_x = p_of(k);
                break;
            }
        }

        if (q_x > 0.0f && (p_x >= q_x || uni(gsmpl->rng) < p_x / q_x)) {
            common_sampler_accept(gsmpl, draft[i], true);
            result.push_back(draft[i]);
            continue;
        }

        // rejected: tokens outside q's support keep all of p
        residual.clear();
        float sum = 0.0f;
        for (size_t k = 0; k < cur_p->size; ++k) {
            const float r = p_of(k) - prob_of(q.data(), q.size(), cur_p->data[k].id);
            if (r > 0.0f) {
                residual.push_back({ cur_p->data[k].id, 0.0f, r });
                sum += r;
            }
        }

        llama_token id = id_tgt;
        if (sum > 0.0f) {
            float u = uni(gsmpl->rng) * sum;
            id = residual.back().id;
            for (const auto & e : residual) {
                u -= e.p;
                if (u <= 0.0f) {
                    id = e.id;
                    break;
                }
            }
        }

        common_sampler_accept(gsmpl, id, true);
        result.push_back(id);

        break;
    }

    if (i == draft.size()) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        if (id == LLAMA_TOKEN_NULL) {
            result.clear();
            return result;
        }

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);
    }

    return result;
}

std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const llama_tokens & draft, bool grammar_first) {
    std::vector<int> idxs(draft.size() + 1);
    for (size_t i = 0; i < idxs.size(); ++i) {
        idxs[i] = i;
    }

    return common_sampler_sample_and_accept_n(gsmpl, ctx, idxs, draft, grammar_first);
}

uint32_t common_sampler_get_seed(const struct common_sampler * gsmpl) {
    return llama_sampler_get_seed(gsmpl->chain);
}

bool common_sampler_reasoning_budget_force(struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return false;
    }

    return common_reasoning_budget_force(gsmpl->rbudget);
}

// helpers

llama_token_data_array * common_sampler_get_candidates(struct common_sampler * gsmpl, bool do_sort) {
    const auto tm = gsmpl->tm();

    auto * res = &gsmpl->cur_p;

    if (do_sort && !res->sorted) {
        // remember the selected token before sorting
        const llama_token id = res->data[res->selected].id;

        std::sort(res->data, res->data + res->size, [](const llama_token_data & a, const llama_token_data & b) {
            return a.p > b.p;
        });

        // restore the selected token after sorting
        for (size_t i = 0; i < res->size; ++i) {
            if (res->data[i].id == id) {
                res->selected = i;
                break;
            }
        }

        res->sorted = true;
    }

    return res;
}

llama_token common_sampler_last(const struct common_sampler * gsmpl) {
    return gsmpl->prev.rat(0);
}

std::string common_sampler_print(const struct common_sampler * gsmpl) {
    std::string result = "logits ";

    for (int i = 0; i < llama_sampler_chain_n(gsmpl->chain); i++) {
        const auto * smpl = llama_sampler_chain_get(gsmpl->chain, i);
        result += std::string("-> ");
        result += std::string(llama_sampler_name(smpl)) + " ";
    }

    return result;
}

std::string common_sampler_prev_str(common_sampler * gsmpl, llama_context * ctx_main, int n) {
    n = std::min(n, (int) gsmpl->prev.size());

    if (n <= 0) {
        return "";
    }

    std::string result;
    result.reserve(8*n); // 8 is the average length of a token [citation needed], TODO: compute this from the vocab

    for (int i = n - 1; i >= 0; i--) {
        const llama_token id = gsmpl->prev.rat(i);

        GGML_ASSERT(id != LLAMA_TOKEN_NULL && "null token in the sampling history - should not happen");

        result += common_token_to_piece(ctx_main, id);
    }

    return result;
}

char common_sampler_type_to_chr(enum common_sampler_type cnstr) {
    switch (cnstr) {
        case COMMON_SAMPLER_TYPE_DRY:         return 'd';
        case COMMON_SAMPLER_TYPE_TOP_K:       return 'k';
        case COMMON_SAMPLER_TYPE_TYPICAL_P:   return 'y';
        case COMMON_SAMPLER_TYPE_TOP_P:       return 'p';
        case COMMON_SAMPLER_TYPE_TOP_N_SIGMA: return 's';
        case COMMON_SAMPLER_TYPE_MIN_P:       return 'm';
        case COMMON_SAMPLER_TYPE_TEMPERATURE: return 't';
        case COMMON_SAMPLER_TYPE_XTC:         return 'x';
        case COMMON_SAMPLER_TYPE_INFILL:      return 'i';
        case COMMON_SAMPLER_TYPE_PENALTIES:   return 'e';
        case COMMON_SAMPLER_TYPE_ADAPTIVE_P:  return 'a';
        default : return '?';
    }
}

std::string common_sampler_type_to_str(enum common_sampler_type cnstr) {
    switch (cnstr) {
        case COMMON_SAMPLER_TYPE_DRY:         return "dry";
        case COMMON_SAMPLER_TYPE_TOP_K:       return "top_k";
        case COMMON_SAMPLER_TYPE_TYPICAL_P:   return "typ_p";
        case COMMON_SAMPLER_TYPE_TOP_P:       return "top_p";
        case COMMON_SAMPLER_TYPE_TOP_N_SIGMA: return "top_n_sigma";
        case COMMON_SAMPLER_TYPE_MIN_P:       return "min_p";
        case COMMON_SAMPLER_TYPE_TEMPERATURE: return "temperature";
        case COMMON_SAMPLER_TYPE_XTC:         return "xtc";
        case COMMON_SAMPLER_TYPE_INFILL:      return "infill";
        case COMMON_SAMPLER_TYPE_PENALTIES:   return "penalties";
        case COMMON_SAMPLER_TYPE_ADAPTIVE_P:  return "adaptive_p";
        default : return "";
    }
}

std::vector<common_sampler_type> common_sampler_types_from_names(const std::vector<std::string> & names) {
    // sampler names can be written multiple ways; generate aliases from canonical names
    static const auto sampler_name_map = []{
        // canonical sampler name mapping
        std::unordered_map<std::string, common_sampler_type> canonical_name_map {
            { "dry",         COMMON_SAMPLER_TYPE_DRY         },
            { "top_k",       COMMON_SAMPLER_TYPE_TOP_K       },
            { "top_p",       COMMON_SAMPLER_TYPE_TOP_P       },
            { "top_n_sigma", COMMON_SAMPLER_TYPE_TOP_N_SIGMA },
            { "typ_p",       COMMON_SAMPLER_TYPE_TYPICAL_P   },
            { "min_p",       COMMON_SAMPLER_TYPE_MIN_P       },
            { "temperature", COMMON_SAMPLER_TYPE_TEMPERATURE },
            { "xtc",         COMMON_SAMPLER_TYPE_XTC         },
            { "infill",      COMMON_SAMPLER_TYPE_INFILL      },
            { "penalties",   COMMON_SAMPLER_TYPE_PENALTIES   },
            { "adaptive_p",  COMMON_SAMPLER_TYPE_ADAPTIVE_P  }
        };
        std::unordered_map<std::string, common_sampler_type> alias_name_map;
        for (const auto & entry : canonical_name_map) {
            const std::string & canonical = entry.first;
            if (canonical.find('_') == std::string::npos) {
                continue;
            }
            // kebab-case: "top-k", "min-p", etc.
            {
                std::string kebab_case = canonical;
                std::replace(kebab_case.begin(), kebab_case.end(), '_', '-');
                alias_name_map.insert({kebab_case, entry.second});
            }
            // no dash: "topk", "minp", etc.
            {
                std::string no_dash = canonical;
                no_dash.erase(std::remove(no_dash.begin(), no_dash.end(), '_'), no_dash.end());
                alias_name_map.insert({no_dash, entry.second});
            }
        }
        // misc. aliases
        alias_name_map.insert({"nucleus", COMMON_SAMPLER_TYPE_TOP_P});
        alias_name_map.insert({"temp",    COMMON_SAMPLER_TYPE_TEMPERATURE});
        alias_name_map.insert({"typ",     COMMON_SAMPLER_TYPE_TYPICAL_P});
        // include aliases + canonical names in the complete mapping
        alias_name_map.merge(canonical_name_map);
        return alias_name_map;
    }();

    std::vector<common_sampler_type> samplers;
    samplers.reserve(names.size());

    for (const auto & name : names) {
        std::string name_lower = name;
        std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);
        auto sampler = sampler_name_map.find(name_lower);
        if (sampler != sampler_name_map.end()) {
            samplers.push_back(sampler->second);
            continue;
        }
        LOG_WRN("%s: unable to match sampler by name '%s'\n", __func__, name_lower.c_str());
    }

    return samplers;
}

std::vector<common_sampler_type> common_sampler_types_from_chars(const std::string & chars) {
    std::unordered_map<char, common_sampler_type> sampler_name_map = {
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_DRY),         COMMON_SAMPLER_TYPE_DRY },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TOP_K),       COMMON_SAMPLER_TYPE_TOP_K },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TYPICAL_P),   COMMON_SAMPLER_TYPE_TYPICAL_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TOP_P),       COMMON_SAMPLER_TYPE_TOP_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TOP_N_SIGMA), COMMON_SAMPLER_TYPE_TOP_N_SIGMA },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_MIN_P),       COMMON_SAMPLER_TYPE_MIN_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TEMPERATURE), COMMON_SAMPLER_TYPE_TEMPERATURE },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_XTC),         COMMON_SAMPLER_TYPE_XTC },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_INFILL),      COMMON_SAMPLER_TYPE_INFILL },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_PENALTIES),   COMMON_SAMPLER_TYPE_PENALTIES },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_ADAPTIVE_P),  COMMON_SAMPLER_TYPE_ADAPTIVE_P },
    };

    std::vector<common_sampler_type> samplers;
    samplers.reserve(chars.size());

    for (const auto & c : chars) {
        const auto sampler = sampler_name_map.find(c);
        if (sampler != sampler_name_map.end()) {
            samplers.push_back(sampler->second);
        } else {
            LOG_WRN("%s: unable to match sampler by char '%c'\n", __func__, c);
        }
    }

    return samplers;
}

void common_sampling_add_preserved_tokens(common_params_sampling & sampling, const llama_vocab * vocab, const std::vector<std::string> & tokens) {
    GGML_ASSERT(vocab != nullptr);
    for (const auto & t : tokens) {
        auto ids = common_tokenize(vocab, t, false, true);
        if (ids.size() == 1) {
            sampling.preserved_tokens.insert(ids[0]);
        }
    }
}

void common_sampling_add_grammar_triggers(common_params_sampling & sampling, const llama_vocab * vocab, std::vector<common_grammar_trigger> triggers) {
    GGML_ASSERT(vocab != nullptr);
    for (auto & trigger : triggers) {
        if (trigger.type == COMMON_GRAMMAR_TRIGGER_TYPE_WORD) {
            const auto & word = trigger.value;
            auto ids = common_tokenize(vocab, word, false, true);
            if (ids.size() == 1) {
                auto token = ids[0];
                if (std::find(sampling.preserved_tokens.begin(), sampling.preserved_tokens.end(), (llama_token) token) == sampling.preserved_tokens.end()) {
                    throw std::runtime_error("Grammar trigger word should be marked as preserved token: " + word);
                }
                common_grammar_trigger token_trigger;
                token_trigger.type  = COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN;
                token_trigger.value = word;
                token_trigger.token = token;
                sampling.grammar_triggers.push_back(std::move(token_trigger));
            } else {
                sampling.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, word});
            }
        } else {
            sampling.grammar_triggers.push_back(std::move(trigger));
        }
    }
    if (sampling.grammar_lazy && sampling.grammar_triggers.empty()) {
        throw std::runtime_error("Error: no triggers set for lazy grammar!");
    }
}
