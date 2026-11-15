#pragma once

// Which of the model's two globals a layer range reads -- answered in one place.
//
// Why this header exists. "Does this range reach the source model's last trunk layer, and does it
// start at layer 0" is the same question as "does this range cover the output head / the token
// embedding", and the answer decides six different things:
//
//   src/llama-model-loader.cpp  get_weight(): whether output.weight / token_embd.weight are read
//   src/llama-context.cpp       the graph's logits node, and which stage is a relay vs the tail
//   src/llama-context.cpp       split_check_head(): a head must NOT reach the output
//   common/common.cpp           whether llama-server refuses a truncated model
//   src/qlh-vram-budget.h       whether the VRAM estimate charges the output tensors
//   tools/split-stage/main.cpp  whether a stage may call itself a relay or a tail
//
// Six hand-written copies of `hi >= n_layer` are six chances for one of them to drift, and one
// did: the attach check spelled out `layer_hi >= 0` next to a comment claiming it was the loader's
// predicate, while the loader's own test is `layer_hi > 0`. The two disagree for `layer_hi == 0`,
// and the comment was the only thing holding them together.
//
// The definitions live here rather than in llama_model_loader because the question is asked both
// before a model exists (the VRAM estimate, from the raw -ngl range read out of `--layer-range`)
// and after it is loaded (the relay check, from llama_hparams). Both the loader and llama_hparams
// delegate to these three functions, and llama_model_has_output() / llama_model_has_tok_embd()
// expose the hparams answer to code outside src/ (common/common.cpp, tools/split-stage). Changing
// the rule means editing this header and nothing else.
//
// Dependency-free on purpose: <cstdint> and nothing else. src/qlh-vram-budget.h is header-only and
// is included by tools/split-stage, which links `llama` and nothing else; pulling a llama internal
// header in here to share three integers would trade one coupling for another. That is also why
// this is a qlh-* header next to qlh-layer-placement.h and qlh-split-defaults.h.

#include <cstdint>

namespace qlh_layer_range {

// A whole-model load has no range: the loader's sentinel is hi < 0 (llama-model-loader.h's
// layer_hi = -1), and llama_hparams carries the same two numbers after the load. The extra
// `lo >= 0` / `hi > lo` terms give a caller that builds the numbers from arguments (the VRAM
// estimate) the same answer as the loader: `hi > 0` on its own would call `lo = -1, hi = 5` a
// range, and nothing can load one.
inline bool active(int32_t lo, int32_t hi) {
    return lo >= 0 && hi > lo;
}

// The token embedding is the input layer of the source model, so it belongs to the range that
// starts at layer 0 -- and to a whole-model load.
inline bool needs_tok_embd(int32_t lo, int32_t hi) {
    return !active(lo, hi) || lo == 0;
}

// The output head (output_norm + output) belongs to the range that reaches the source model's last
// trunk layer, i.e. hi >= n_layer_src. n_layer_src is the *trunk* depth
// (block_count - nextn_predict_layers), never block_count: the MTP block travels with the tail
// range too, so comparing hi against block_count would call every tail range a relay and leave the
// last stage without logits.
inline bool needs_output(int32_t lo, int32_t hi, int32_t n_layer_src) {
    return !active(lo, hi) || hi >= n_layer_src;
}

} // namespace qlh_layer_range
