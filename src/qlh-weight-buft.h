#pragma once

// Where model weights live when the scheduler would place them in a device's
// *host* buffer type.
//
// Background, and why this is a policy rather than a switch.
//
// `ggml_backend_vk_host_malloc()` asks for HOST_VISIBLE|HOST_COHERENT|HOST_CACHED. On a
// single-heap integrated GPU every memory type is also DEVICE_LOCAL, so a "host" buffer is
// not a separate pool at all: it is allocated out of the same heap the weights use
// (upstream #31538). A partial offload therefore pays device memory for the layers it
// deliberately left on the CPU, and the KV cache then fails to allocate with an error that
// names neither the host buffer nor the cause.
//
// `llama-model-loader.cpp` has answered that with one line: when the chosen
// buffer type is a device host buffer, it is replaced by the CPU buffer type on an iGPU, and
// on any device when mmap is in use. That fallback is what makes Vulkan_Host unreachable for
// weights -- and it is also the only reason the fallback exists, so it is not a bug to be
// deleted but a trade to be made explicit.
//
// The trade, measured on Intel Iris Xe: allowing the host buffer moved 9B dense pp512
// 74.04 -> 89.15 t/s (+20%) and 35B-A3B Q2_K pp64 13.95 -> 21.78 t/s (+56%), at the price of
// the whole weight set counting against the device heap (9B: 1.32 -> 5.61 GiB; 35B Q2_K still
// fit, 12.84 GiB of 15.82). Prefill gains, heap cost. Which side of that trade is right
// depends on the model, the context size and the machine, so the default cannot silently pick
// one.
//
// The host-buffer fix backs a single-heap iGPU's host buffer with portable_aligned_malloc
// (system memory), so both halves of that trade are gone:
//   - the price is gone: `host` leaves the device-heap peak unchanged at 5.360 GiB (9B) and
//     12.190 GiB (35B, -c 4096);
//   - the gain is gone too: the Vulkan device buffer type already accepts all but ~545 MiB
//     (9B) / ~515 MiB (35B-A3B) of the weights, so `host` moves that much and measured
//     pp512 93.18 -> 92.83 (9B) and pp64 23.68 -> 23.81 (35B) -- noise.
// What remains is the control itself: `host` and `device` are the two explicit ways to
// overrule the scheduler's answer, and `auto` keeps the default answer for anyone who does
// not ask.

//
// Hence:
//
//   auto    (default) The previous behaviour, byte for byte: on an iGPU (or with mmap)
//                     the host buffer becomes the CPU buffer type. No environment variable,
//                     no log line, no allocation change. This is the revertible default:
//                     unset QLH_WEIGHT_BUFT and the build behaves as it did before.
//   host              Keep the device host buffer type. Weights land in
//                     Vulkan_Host, which on a single-heap iGPU counts against the device heap.
//                     Highest measured prefill throughput, highest heap cost. This also makes
//                     an explicit `-ot ...=Vulkan_Host` actually stick (auto would replace it).
//   device            Never use a host buffer for weights: a tensor the scheduler placed in
//                     the device's host buffer goes to that device's own buffer type instead
//                     -- but only when the device buffer type supports the op. When it does
//                     not, the host buffer is kept and one warning names the tensor; a tensor
//                     whose op the device cannot run is not something to discover at compute
//                     time. Note that this also applies to layers the caller deliberately
//                     left on the CPU (--n-cpu-moe, -ot, a partial -ngl): they will be moved
//                     into device memory, which is exactly what the single-heap problem above
//                     is about, so `device` is a deliberate override, not a safe default.
//
// Why "policy" and not "allow": QLH_ALLOW_HOST_WEIGHTS=1 would describe what the code is
// permitted to do. What an operator actually needs to state is where the weights are, and
// `auto`/`host`/`device` is that statement. It also gives the third option -- force device
// memory -- which an allow-switch cannot express.
//
// Parsed once per process from QLH_WEIGHT_BUFT in a function-local static: initialisation is
// thread-safe (C++11 magic statics) and happens exactly once, so this is one getenv() for the
// whole load rather than one per tensor, and every caller sees the same answer.
//
// An invalid value throws. This tree has QLH_SPLIT_SKIP_IDENTITY as the standing counter-
// example of a switch that quietly did nothing when its value was wrong; a misspelled
// QLH_WEIGHT_BUFT must not silently leave the operator with `auto`. The loader's constructor
// resolves the policy before anything is allocated, so the throw surfaces as the ordinary
// "error loading model: ..." line and the load stops.
//
// Moving into llama_model_params: `settings` is deliberately a plain struct separate from
// parsing, so the future change is
//   - add `qlh_weight_buft::settings weight_buft;` to llama_model_params (llama.h),
//   - parse the environment in common/arg.cpp where every other flag is parsed, into that
//     field (and keep QLH_WEIGHT_BUFT as the fallback when the field is left at `auto`),
//   - have llama_model_loader store the field and drop from_env() from the constructor.
// No loader logic changes: create_tensor() already only reads `settings::value`.

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace qlh_weight_buft {

enum class policy {
    AUTO,
    HOST,
    DEVICE,
};

struct settings {
    policy       value          = policy::AUTO;
    bool         explicitly_set = false;  // came from the environment, not from the default
    const char * raw            = nullptr;  // the value as written, for the log line
};

inline const char * ENV = "QLH_WEIGHT_BUFT";

inline const char * policy_name(policy p) {
    switch (p) {
        case policy::AUTO:   return "auto";
        case policy::HOST:   return "host";
        case policy::DEVICE: return "device";
    }
    return "?";
}

// One line stating the consequence, not just the name: an operator who reads the log should
// not have to look up what "host" did to the heap.
inline const char * policy_effect(policy p) {
    switch (p) {
        case policy::AUTO:
            return "device host buffers are replaced by the CPU buffer type on an iGPU or with mmap (previous behaviour)";
        case policy::HOST:
            return "device host buffers are kept for weights; on a single-heap iGPU the weights count against the device heap (#31538)";
        case policy::DEVICE:
            return "device host buffers are not used for weights; such tensors go to the device buffer type when the device supports the op";
    }
    return "?";
}

inline settings parse(const char * raw) {
    settings s;
    if (raw == nullptr) {
        return s;  // unset: auto, silent, previous behaviour
    }
    if (std::strcmp(raw, "auto") == 0) {
        s.value = policy::AUTO;
    } else if (std::strcmp(raw, "host") == 0) {
        s.value = policy::HOST;
    } else if (std::strcmp(raw, "device") == 0) {
        s.value = policy::DEVICE;
    } else {
        // Not a fallback: a wrong policy is a wrong answer, and the operator has to see it.
        throw std::runtime_error(std::string(ENV) + ": invalid value '" + raw +
                "' (expected one of: auto, host, device)");
    }
    s.explicitly_set = true;
    s.raw            = raw;
    return s;
}

inline const settings & from_env() {
    static const settings s = parse(std::getenv(ENV));
    return s;
}

} // namespace qlh_weight_buft
