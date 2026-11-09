#pragma once

// [QLH-SPLIT] v2.4: the defaults a *stage* of a split chain needs, in one place.
//
// Why this header exists: op_offload is upstream's switch that lets the GPU take over the big
// operators of tensors that live in host memory. It is ON upstream, and on a *range* load it is
// the difference between a positive and a negative head stage: with it on, the CPU-resident
// blocks' weights go into the pinned CUDA host buffer and the GPU runs their prefill matmuls,
// which costs them ~31% of decode speed. Measured on the segment probe (2026-11-09, head 0:32,
// same build): -ngl 0 tg 11.25 with it on vs 18.23 with it off; -ngl 8 tg 14.60 vs 22.25 (+36%).
//
// tools/split-stage has defaulted `no_op_offload = true` since v2.2. llama-server, which is the
// *head* of every chain, was left on the upstream default instead -- so the same head measured
// +36% on the segment probe and -10% through the server (E: 8.44 vs the all-CPU A 9.34), and no
// line of log said which switch was on. Both entry points now read the default from here, so the
// value cannot drift again; an explicit --op-offload / --no-op-offload still wins.
//
// Header-only and dependency-free on purpose: tools/split-stage links against `llama` and
// nothing else (its CMakeLists says so), and this constant needs no more than that.
static constexpr bool QLH_SPLIT_DEFAULT_NO_OP_OFFLOAD = true;
