#include "llama-chip-emb.h"

#include <cstring>

namespace {

// fp32 -> bf16 bits with round-to-nearest-even, matching the reference expander. Every value the
// blob describes (0, +-1, +-128, 2^40) is exactly representable, so the rounding never matters
// here, but keeping it identical means a future non-exact feature still matches `merge.py`.
inline uint16_t f2bf(float v) {
    uint32_t u;
    std::memcpy(&u, &v, 4);
    return (uint16_t) ((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

template<typename T>
bool read_at(const std::vector<uint8_t> & b, size_t & p, T & out) {
    if (p + sizeof(T) > b.size()) {
        return false;
    }
    std::memcpy(&out, b.data() + p, sizeof(T));
    p += sizeof(T);
    return true;
}

bool read_bytes(const std::vector<uint8_t> & b, size_t & p, void * out, size_t n) {
    if (p + n > b.size()) {
        return false;
    }
    std::memcpy(out, b.data() + p, n);
    p += n;
    return true;
}

} // namespace

bool llama_chip_emb::parse_blob(const std::vector<uint8_t> & blob, std::string & err) {
    if (blob.size() < 48 || std::memcmp(blob.data(), "CHIPEMB1", 8) != 0) {
        err = "chip embedding blob has a bad magic";
        return false;
    }

    size_t p = 8;
    uint32_t hdr[10];
    if (!read_bytes(blob, p, hdr, sizeof(hdr))) {
        err = "chip embedding blob is truncated (header)";
        return false;
    }
    vocab         = hdr[0];
    v0            = hdr[1];
    blob_hidden   = hdr[2];
    off           = hdr[3];
    m             = hdr[4];
    log2_ballast  = hdr[5];
    code_amp      = hdr[6];
    nfam          = hdr[7];
    npay          = hdr[8];
    const uint32_t n_ranges = hdr[9];

    if (blob_hidden != (uint32_t) hidden) {
        err = "chip embedding blob is for a " + std::to_string(blob_hidden) +
              "-wide embedding, the table is " + std::to_string(hidden);
        return false;
    }
    if (n_ranges == 0 || n_ranges > 64 || nfam > 32 || npay > 64) {
        err = "chip embedding blob has implausible family counts";
        return false;
    }

    ranges.resize(n_ranges);
    for (auto & r : ranges) {
        if (!read_at(blob, p, r.base) || !read_at(blob, p, r.size) ||
            !read_at(blob, p, r.nbits) || !read_at(blob, p, r.col_is) ||
            !read_bytes(blob, p, r.cols, sizeof(r.cols))) {
            err = "chip embedding blob is truncated (families)";
            return false;
        }
    }

    if (!read_at(blob, p, tau) || !read_at(blob, p, mu)) {
        err = "chip embedding blob is truncated (tau/mu)";
        return false;
    }

    code_cols.resize(nfam + npay);
    if (!read_bytes(blob, p, code_cols.data(), code_cols.size()*sizeof(uint16_t))) {
        err = "chip embedding blob is truncated (code columns)";
        return false;
    }

    digit_cols.resize(8);
    if (!read_bytes(blob, p, digit_cols.data(), digit_cols.size()*sizeof(uint16_t))) {
        err = "chip embedding blob is truncated (digit columns)";
        return false;
    }

    if (!read_at(blob, p, qs_range) || !read_at(blob, p, qd_range)) {
        err = "chip embedding blob is truncated (digit ranges)";
        return false;
    }

    uint32_t n_ctrl = 0;
    if (!read_at(blob, p, n_ctrl) || n_ctrl > vocab) {
        err = "chip embedding blob is truncated (controls)";
        return false;
    }
    ctrls.resize(n_ctrl);
    for (auto & c : ctrls) {
        if (!read_at(blob, p, c.id) || !read_at(blob, p, c.col_is)) {
            err = "chip embedding blob is truncated (control rows)";
            return false;
        }
    }

    return true;
}

bool llama_chip_emb::parse(const std::vector<uint8_t> & blob,
                           const std::vector<uint8_t> & kernel,
                           int64_t n_stored_, int64_t n_vocab_, int64_t hidden_, std::string & err) {
    n_stored = n_stored_;
    n_vocab  = n_vocab_;
    hidden   = hidden_;

    if (n_stored <= 0 || n_vocab <= 0 || hidden <= 0 || hidden > ((int64_t) 1 << 20)) {
        err = "chip embedding: implausible table geometry";
        return false;
    }
    if (n_stored > n_vocab) {
        err = "chip embedding: the file holds more rows than the vocabulary";
        return false;
    }

    if (!parse_blob(blob, err)) {
        return false;
    }

    // The blob describes rows [v0, v0+vocab) of the same table it was built for. Both numbers
    // have to agree with the file, otherwise the column numbers in the blob would be applied to
    // rows that belong to a different checkpoint (silently wrong logits, not a crash).
    if ((int64_t) v0 != n_stored) {
        err = "chip embedding blob starts at row " + std::to_string(v0) +
              ", the file stops after row " + std::to_string(n_stored);
        return false;
    }
    if ((int64_t) v0 + (int64_t) vocab > n_vocab) {
        err = "chip embedding blob describes rows past the vocabulary";
        return false;
    }

    n_chip = vocab;

    // Kernel virtual-token rows: a whole number of rows, or absent for a checkpoint without them.
    const int64_t kernel_row_bytes = hidden * 2;
    if (kernel.size() % (size_t) kernel_row_bytes != 0) {
        err = "chip embedding kernel blob is not a whole number of rows";
        return false;
    }
    n_kernel = (int64_t) (kernel.size() / (size_t) kernel_row_bytes);
    if (n_stored + n_chip + n_kernel > n_vocab) {
        err = "chip embedding kernel rows run past the vocabulary";
        return false;
    }
    n_pad = n_vocab - (n_stored + n_chip + n_kernel);
    kernel_rows = kernel;

    return true;
}

void llama_chip_emb::make_chip_row(uint16_t * r, uint32_t t) const {
    std::memset(r, 0, (size_t) hidden * sizeof(uint16_t));

    const uint16_t ballast = f2bf((float) (1ull << log2_ballast));
    for (uint32_t i = 0; i < m; ++i) {
        r[off + i] = ballast;
    }

    const uint16_t one = f2bf(1.0f), neg = f2bf(-1.0f);
    const uint16_t cp = f2bf((float) code_amp), cn = f2bf(-(float) code_amp);

    uint32_t famid = 0, p = 0;
    int ri = -1;
    for (uint32_t i = 0; i < (uint32_t) ranges.size(); ++i) {
        const range_t & R = ranges[i];
        if (t >= R.base && t < R.base + R.size) {
            ri = (int) i;
            p = t - R.base;
            break;
        }
    }
    if (ri < 0) {
        for (uint32_t j = 0; j < (uint32_t) ctrls.size(); ++j) {
            if (ctrls[j].id == t) {
                r[ctrls[j].col_is] = one;
                famid = (uint32_t) ranges.size() + j;
                break;
            }
        }
    } else {
        const range_t & R = ranges[ri];
        famid = (uint32_t) ri;
        r[R.col_is] = one;
        for (uint16_t b = 0; b < R.nbits; ++b) {
            // a payload family without per-bit columns (QS / HEX / FLT) stores 0xFFFF here
            if (R.cols[b] != 0xFFFFu && ((p >> b) & 1u)) {
                r[R.cols[b]] = one;
            }
        }
        if ((uint32_t) ri == qs_range || (uint32_t) ri == qd_range) {
            for (int k = 0; k < 8; ++k) {
                const bool hit = (uint32_t) ri == qs_range ? (p == (uint32_t) k)
                                                           : ((p >> 4) == (uint32_t) (k - 1));
                if (hit) {
                    r[digit_cols[k]] = one;
                    break;
                }
            }
        }
    }

    for (uint32_t i = 0; i < (uint32_t) code_cols.size(); ++i) {
        const uint32_t bit = i < nfam ? ((famid >> i) & 1u) : ((p >> (i - nfam)) & 1u);
        r[code_cols[i]] = bit ? cp : cn;
    }

    r[tau] = neg;
    r[mu]  = one;
}

void llama_chip_emb::make_row(uint16_t * dst, uint32_t id) const {
    const int64_t k = (int64_t) id - n_stored;
    if (k < 0) {
        // text rows live in the file; a caller that ends up here has no row for this id
        std::memset(dst, 0, (size_t) hidden * sizeof(uint16_t));
        return;
    }
    if (k < n_chip) {
        make_chip_row(dst, (uint32_t) k);
        return;
    }
    if (k < n_chip + n_kernel) {
        std::memcpy(dst, kernel_rows.data() + (size_t) (k - n_chip) * (size_t) hidden * 2,
                    (size_t) hidden * 2);
        return;
    }
    // trailing rows up to n_vocab are zeros in the source table
    std::memset(dst, 0, (size_t) hidden * sizeof(uint16_t));
}
