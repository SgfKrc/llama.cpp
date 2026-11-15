#pragma once

// Which layers of a segment go to the GPU: one place that answers it.
//
// Why this header exists. `llama_model::load_tensors()` (src/llama-model.cpp) decides placement
// from three numbers -- i_gpu_start, act_gpu_layers, offload_output -- and src/qlh-vram-budget.h
// has to predict the *same* placement from the GGUF index alone, before the model is loaded. That
// estimate exists for one purpose: to catch a silent VRAM spill. A spill is silent by
// construction: the loader still prints "offloaded N/M layers", Windows (WDDM) backs the excess
// with system RAM, and the only symptom is a 27x slower decode. A second copy of the placement
// rules is therefore not redundancy -- it is a detector that quietly stops detecting the moment
// the rules change in only one of the two places.
//
// So both sides call this header:
//
//   src/llama-model.cpp    load_tensors(): the authoritative assignment
//   src/qlh-vram-budget.h  estimate_want(): the prediction, accumulated as tensor bytes
//
// and src/qlh-vram-budget.h keeps only the "add up the bytes of the tensors this places here"
// half. Changing a placement rule means editing compute() below and nothing else.
//
// Dependency-free on purpose: <algorithm> and nothing else. src/qlh-vram-budget.h is header-only
// and is included by tools/split-stage, which links `llama` and nothing else; pulling ggml or a
// llama internal header in here to share three integers would trade one coupling for another.
// That is also why this is a qlh-* header next to qlh-weight-buft.h and qlh-split-defaults.h
// rather than a member of llama_hparams: placement is not a property of the model (it is a
// property of -ngl, of whether this process loads a range, and of whether any device exists), and
// llama-hparams.h is the model-only header. Both callers can see this one.
//
// The rules are upstream's, unchanged -- with one lift described in compute() below. Keep them in
// sync with upstream by moving rules *here*, never by adding a third copy at a call site.

#include <algorithm>

namespace qlh_placement {

// `n_gpu_layers` as llama_model::n_gpu_layers() reports it: a negative request means "every local
// block, plus the output slot". Callers holding the raw -ngl value resolve it here so the sentinel
// has exactly one meaning.
inline int resolve_n_gpu_layers(int requested, int n_layer_all) {
    return requested >= 0 ? requested : n_layer_all + 1;
}

// The placement of one segment's layers.
//
// n_layer_all is the number of blocks *this process* loads. A layer-range load has already
// replaced the model's block count with the range's local count before placement runs (see
// llama-model-loader.cpp), so this is the segment's own count in both modes.
struct layer_placement {
    int  n_layer_all    = 0;
    int  i_gpu_start    = 0;     // first local block that goes to the GPU
    int  act_gpu_layers = 0;     // how many slots the request covers (the output slot included)
    bool offload_output = false; // the output slot's own decision inside a range
    bool has_devices    = false;

    // il < n_layer_all is a repeating block; il == n_layer_all is the output slot. This is the
    // predicate load_tensors() assigns devices with, in full.
    bool on_gpu(int il) const {
        if (!has_devices) {
            return false;
        }
        return il < n_layer_all
            ? (il >= i_gpu_start && il - i_gpu_start < act_gpu_layers)
            : offload_output;
    }

    // How many *repeating* blocks land on the GPU. This is what the loader reports as
    // "offloading N repeating layers to GPU" and what the VRAM estimate reports as N/M blocks;
    // the output slot is reported separately because it is not a block.
    int n_repeating_on_gpu() const {
        if (!has_devices) {
            return 0;
        }
        return std::max(0, std::min(act_gpu_layers, n_layer_all - i_gpu_start));
    }
};

// n_gpu_layers must already be resolved (see resolve_n_gpu_layers); layer_range is
// llama_model_loader::layer_range_active(); has_devices is !devices.empty().
//
// `-ngl N` counts inside the segment when this is a layer-range load:
//
//   whole model : i_gpu_start = n_layer_all + 1 - N   (N includes the output slot)
//   range load  : i_gpu_start = n_layer_all     - N   (N is this range's blocks only)
//
// Upstream counts the output layer in n_gpu_layers, which is right for a whole model: the logits
// layer is reachable only after every block, so it is the natural "last layer". In a range it is
// not: a head/relay has no output tensor at all, so reserving a slot for it would make "-ngl 8"
// offload 7 blocks and leave "give this segment 8 blocks of GPU" unexpressible. A tail *does*
// carry the output weight, and it gets it when N reaches the local block count (N == n_layer_all
// also offloads the output slot, see the on_gpu() test above), so "-ngl all" still means "this
// whole segment, logits included".
//
// The whole-model path is bit-for-bit upstream: n_layer_all + 1 - N for a non-range load.
//
// The output slot is decided separately inside a range: upstream's single i_gpu_start sweep puts
// it on the GPU for any N > 0, which is what a whole model wants. A range wants it only when the
// *whole* segment is on the GPU (N >= local blocks); "-ngl 3" on a 9-block tail must not drag the
// logits layer along just because it is the last slot.
inline layer_placement compute(int n_gpu_layers, int n_layer_all, bool layer_range, bool has_devices) {
    layer_placement p;
    p.n_layer_all    = n_layer_all;
    p.has_devices    = has_devices;
    p.i_gpu_start    = std::max((layer_range ? n_layer_all : n_layer_all + 1) - n_gpu_layers, 0);
    p.act_gpu_layers = has_devices ? std::min(n_gpu_layers, n_layer_all + 1) : 0;
    p.offload_output = layer_range ? (n_gpu_layers >= n_layer_all) : (n_gpu_layers > 0);
    return p;
}

} // namespace qlh_placement
