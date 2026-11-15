#pragma once

// One parser for one format: the `lo:hi` layer range, and the name of the environment variable that
// carries one.
//
// Why this header exists. The same string was parsed in four places, with three different attitudes
// to a malformed value:
//
//   src/llama-model-loader.cpp  LLAMA_LAYER_RANGE, sscanf("%d:%d"): anything malformed left lo/hi
//                               at their defaults and the WHOLE model was loaded instead
//   common/arg.cpp              --split-layer-range / --split-tail-range: throw (parse error)
//   common/arg.cpp              common_split_layer_range_in_effect(): "the loader's rule, mirrored"
//                               -- a fourth copy of the same fallback, kept in step by a comment
//   tools/split-stage/main.cpp  --layer-range: sscanf, refuse to start
//
// Changing the format (accepting a stride, say) meant finding all four, and a value that one copy
// accepted while another rejected could start a stage on a range nobody asked for -- exactly the
// failure the comments at each of those sites warn about. The argument parsers and the loader's
// env fallback now call parse_range() below, so they accept and reject the same set of values, and
// they name the variable from LAYER_RANGE_ENV instead of a fifth literal.
//
// Dependency-free on purpose (<cstdint> and <string_view> and nothing else): the caller is both
// src/ and the tools, and src/ may not include common/. That direction is also why the function is
// not named common_split_range_parse(): it has to live in src/ for the loader to see it, and a
// `common_` name in src/ would name a layer that cannot own it.

#include <cstdint>
#include <string_view>

namespace qlh_split {

// The env-var fallback a stage-per-process run uses instead of --split-layer-range. One name, one
// place: the loader reads it, and the argument parser has to answer "is a range in effect for this
// process at all" (common/arg.cpp, common_split_layer_range_in_effect) before a model exists.
inline constexpr const char * LAYER_RANGE_ENV = "LLAMA_LAYER_RANGE";

// Parse the one layer-range format: `lo:hi`, both ends unsigned decimal, nothing else -- no sign,
// no whitespace, no trailing characters ("0:20:30" is a typo, not a range with a stride) -- with
// 0 <= lo < hi.
//
// On success lo/hi are written; on failure neither is touched. That matters: sscanf writes lo
// before it fails on the second number, and the stage that then started on a half-parsed range was
// the bug this rule is written down for.
inline bool parse_range(std::string_view s, int32_t & lo, int32_t & hi) {
    const auto digits = [](std::string_view v, int32_t & out) {
        if (v.empty()) {
            return false;
        }
        int64_t acc = 0;
        for (const char c : v) {
            if (c < '0' || c > '9') {
                return false;
            }
            acc = acc * 10 + int64_t(c - '0');
            if (acc > int64_t(0x7fffffff)) {
                return false;
            }
        }
        out = int32_t(acc);
        return true;
    };

    const size_t colon = s.find(':');
    if (colon == std::string_view::npos || s.find(':', colon + 1) != std::string_view::npos) {
        return false;
    }

    int32_t l = 0;
    int32_t h = 0;
    if (!digits(s.substr(0, colon), l) || !digits(s.substr(colon + 1), h)) {
        return false;
    }
    if (h <= l) {
        return false;
    }

    lo = l;
    hi = h;
    return true;
}

} // namespace qlh_split
