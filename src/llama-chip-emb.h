#pragma once

// Programmatic rows of the token embedding table (the "chip tokens" of a k0v3 checkpoint).
//
// A checkpoint can store only the text rows of `token_embd.weight` and describe the circuit-token
// rows (394,164 of 642,496 in k0v3) with an 826-byte blob instead: every chip row is 42..58
// non-zero columns of {0, +-1, +-128, 2^40}, all exactly representable in bf16, so rebuilding the
// row bit-for-bit is a memset plus a few stores (~0.14 us/row). The rows are never materialised as
// a whole table: the decoder rebuilds the handful of rows a batch asks for, which is what removes
// those 3.38 GiB from RAM and VRAM instead of only from the file.
//
// The blob is the format `_m4d/gen_compact.py` emits and `_m4d/chip_emb_expand.cpp` expands
// (little endian):
//
//   off 0   'CHIPEMB1'
//   off 8   u32 vocab, v0, hidden, off, m, log2_ballast, code_amp, nfam, npay, n_ranges
//   then    n_ranges x { u32 base, u32 size, u16 nbits, u16 col_is, u16 cols[16] }
//           u16 tau, u16 mu
//           u16 code_cols[nfam+npay]
//           u16 digit_cols[8]
//           u16 qs_range, u16 qd_range
//           u32 n_controls
//           n_controls x { u32 id, u16 col_is }
//
// The layout is bound to the checkpoint that produced it (column numbers come from the overlay
// plan), so parse() checks the blob against the file it came from rather than trusting it:
// vocab/offset/hidden must agree with the tensor and the kv that were read next to it.

#include <cstdint>
#include <string>
#include <vector>

struct llama_chip_emb {
    struct range_t {
        uint32_t base  = 0;    // first token id of the family, in chip space (0 .. n_chip-1)
        uint32_t size  = 0;    // number of tokens in the family
        uint16_t nbits = 0;    // payload width
        uint16_t col_is = 0;   // column of the family's `is<FAM>` one-hot
        uint16_t cols[16] = {}; // column of each payload bit, 0xFFFF = no column (QS/HEX/FLT)
    };

    struct ctrl_t {
        uint32_t id = 0;
        uint16_t col_is = 0;
    };

    // Rows [0, n_stored) come from the file, [n_stored, n_stored+n_chip) from the blob,
    // then n_kernel rows copied out of the kv blob, then n_pad zero rows up to n_vocab.
    int64_t n_stored = 0;
    int64_t n_chip   = 0;
    int64_t n_kernel = 0;
    int64_t n_pad    = 0;
    int64_t n_vocab  = 0;
    int64_t hidden   = 0; // n_embd, i.e. row width in bf16 values

    // header of the blob
    uint32_t vocab = 0, v0 = 0, blob_hidden = 0, off = 0, m = 0;
    uint32_t log2_ballast = 0, code_amp = 0, nfam = 0, npay = 0;
    std::vector<range_t>  ranges;
    uint16_t tau = 0, mu = 0;
    std::vector<uint16_t> code_cols;
    std::vector<uint16_t> digit_cols;
    uint16_t qs_range = 0, qd_range = 0;
    std::vector<ctrl_t>   ctrls;

    // the 8 kernel virtual-token rows, raw bf16 bytes (n_kernel * hidden * 2)
    std::vector<uint8_t> kernel_rows;

    // Parse `blob` (+ optional kernel rows) for a table whose file holds `n_stored` rows of
    // `hidden` bf16 values out of a vocabulary of `n_vocab` tokens. Returns false and fills `err`
    // when the blob or the kv disagree with the file.
    bool parse(const std::vector<uint8_t> & blob,
               const std::vector<uint8_t> & kernel,
               int64_t n_stored, int64_t n_vocab, int64_t hidden, std::string & err);

    // True when token `id` is not stored in the file and has to be generated.
    bool needs_row(int64_t id) const { return id >= n_stored; }

    // Write the bf16 bit pattern of token `id` into `dst` (must hold `hidden` uint16_t values).
    // `id` must satisfy needs_row(); ids past the kernel and pad rows come out as zeros.
    void make_row(uint16_t * dst, uint32_t id) const;

private:
    bool parse_blob(const std::vector<uint8_t> & blob, std::string & err);
    void make_chip_row(uint16_t * dst, uint32_t t) const;
};
