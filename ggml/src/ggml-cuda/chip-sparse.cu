// k0v4: the chip layers' sparse matvec on the GPU.
//
// One thread per output row (per token). The numbers are the point, not the throughput: the kernel
// reproduces the dense reduction tree with explicit round-to-nearest intrinsics, so a skipped zero
// is exactly a skipped `acc + 0*x`. See ggml-cpu/ops.cpp for the two trees and why the batch size
// picks one:
//
//   n_tokens == 1 -> 32 accumulators indexed by k%32, mul then add (ggml_vec_dot_bf16's AVX2 body)
//   n_tokens  > 1 -> 8 lanes indexed by k%8 accumulated with FMA (llamafile's tinyBLAS body)
//
// __fadd_rn / __fmul_rn / __fmaf_rn are single-rounding intrinsics, so this file does not depend on
// -fmad or on the build's fast-math settings.
//
// x is F32 like a mul_mat operand and is rounded to bf16 per element, which is what the dense path
// does to src1 before its dot product (GGML_FP32_TO_BF16). The store's row_ptr is global (all 78
// matrices concatenated), so mat_base[mat] is read on the device: a 4-byte D2H sync per matvec
// would cost more than the kernel itself.

#include "chip-sparse.cuh"

#include <cstdint>

__device__ __forceinline__ float chip_bf16_to_f32(uint16_t b) {
    return __uint_as_float((uint32_t) b << 16);
}

// Same rounding as ggml_compute_fp32_to_bf16 (round to nearest even, NaNs quietened).
__device__ __forceinline__ uint16_t chip_f32_to_bf16(float s) {
    const uint32_t u = __float_as_uint(s);
    if ((u & 0x7FFFFFFFu) > 0x7F800000u) {
        return (uint16_t) ((u >> 16) | 64u);
    }
    return (uint16_t) ((u + (0x7FFFu + ((u >> 16) & 1u))) >> 16);
}

__device__ __forceinline__ float chip_x_bf16(const float * x, uint32_t k) {
    return chip_bf16_to_f32(chip_f32_to_bf16(x[k]));
}

// ggml_vec_dot_bf16's horizontal fold over 32 accumulators.
__device__ __forceinline__ float chip_reduce32(const float * acc) {
    float c[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        c[j] = __fadd_rn(__fadd_rn(acc[j], acc[16 + j]), __fadd_rn(acc[8 + j], acc[24 + j]));
    }
    const float g0 = __fadd_rn(c[0], c[4]);
    const float g1 = __fadd_rn(c[1], c[5]);
    const float g2 = __fadd_rn(c[2], c[6]);
    const float g3 = __fadd_rn(c[3], c[7]);
    return __fadd_rn(__fadd_rn(g0, g2), __fadd_rn(g1, g3));
}

// tinyBLAS' hsum over 8 lanes.
__device__ __forceinline__ float chip_reduce8(const float * a) {
    const float g0 = __fadd_rn(a[0], a[4]);
    const float g1 = __fadd_rn(a[1], a[5]);
    const float g2 = __fadd_rn(a[2], a[6]);
    const float g3 = __fadd_rn(a[3], a[7]);
    return __fadd_rn(__fadd_rn(g0, g2), __fadd_rn(g1, g3));
}

template <bool FMA_TREE>
__device__ __forceinline__ float chip_dot_row(
        const uint32_t * __restrict__ rp,
        const uint16_t * __restrict__ col,
        const uint16_t * __restrict__ w,
        const float    * __restrict__ x,
        uint32_t                      r) {
    const uint32_t b = rp[r];
    const uint32_t e = rp[r + 1];

    if (FMA_TREE) {
        float acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
        for (uint32_t j = b; j < e; ++j) {
            const uint32_t k = col[j];
            const uint32_t l = k & 7u;
            acc[l] = __fmaf_rn(chip_bf16_to_f32(w[j]), chip_x_bf16(x, k), acc[l]);
        }
        return chip_reduce8(acc);
    }

    const uint32_t n = e - b;
    if (n == 0) {
        return 0.0f;
    }
    if (n == 1) {
        const uint32_t k = col[b];
        const float t = __fmul_rn(chip_bf16_to_f32(w[b]), chip_x_bf16(x, k));
        return __fadd_rn(t, 0.0f);
    }
    if (n == 2) {
        const uint32_t k0 = col[b];
        const uint32_t k1 = col[b + 1];
        const float t0 = __fmul_rn(chip_bf16_to_f32(w[b]),     chip_x_bf16(x, k0));
        const float t1 = __fmul_rn(chip_bf16_to_f32(w[b + 1]), chip_x_bf16(x, k1));
        return __fadd_rn(t0, t1);
    }

    float acc[32];
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        acc[i] = 0.0f;
    }
    for (uint32_t j = b; j < e; ++j) {
        const uint32_t k = col[j];
        const float t = __fmul_rn(chip_bf16_to_f32(w[j]), chip_x_bf16(x, k));
        const uint32_t l = k & 31u;
        acc[l] = __fadd_rn(acc[l], t);
    }
    return chip_reduce32(acc);
}

template <bool FMA_TREE>
__global__ void k_chip_sparse_mm(
        const uint32_t * __restrict__ mat_base,
        const uint32_t * __restrict__ row_ptr,
        const uint16_t * __restrict__ col,
        const uint16_t * __restrict__ w,
        const float    * __restrict__ x,
        float          * __restrict__ y,
        int32_t                       mat,
        uint32_t                      rows,
        uint32_t                      kdim) {
    const uint32_t t = blockIdx.y;
    const uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) {
        return;
    }

    const uint32_t * rp = row_ptr + mat_base[mat];
    const float    * x_row = x + (size_t) t * kdim;

    y[(size_t) t * rows + r] = chip_dot_row<FMA_TREE>(rp, col, w, x_row, r);
}

void ggml_cuda_chip_sparse_mm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * t_base = dst->src[0];
    const ggml_tensor * t_rp   = dst->src[1];
    const ggml_tensor * t_col  = dst->src[2];
    const ggml_tensor * t_w    = dst->src[3];
    const ggml_tensor * x      = dst->src[4];

    GGML_ASSERT(ggml_is_contiguous(t_base) && ggml_is_contiguous(t_rp) &&
                ggml_is_contiguous(t_col)  && ggml_is_contiguous(t_w) && ggml_is_contiguous(x));
    GGML_ASSERT(dst->nb[0] == sizeof(float));

    const int32_t mat = ggml_get_op_params_i32(dst, 0);

    const int64_t rows = dst->ne[0];
    const int64_t nt   = dst->ne[1];
    const int64_t K    = x->ne[0];

    if (rows == 0 || nt == 0) {
        return;
    }

    const int block = 256;
    const dim3 grid((unsigned) ((rows + block - 1) / block), (unsigned) nt, 1);

    cudaStream_t stream = ctx.stream();

    if (nt > 1) {
        k_chip_sparse_mm<true><<<grid, block, 0, stream>>>(
            (const uint32_t *) t_base->data, (const uint32_t *) t_rp->data,
            (const uint16_t *) t_col->data,  (const uint16_t *) t_w->data,
            (const float    *) x->data,      (float *) dst->data,
            mat, (uint32_t) rows, (uint32_t) K);
    } else {
        k_chip_sparse_mm<false><<<grid, block, 0, stream>>>(
            (const uint32_t *) t_base->data, (const uint32_t *) t_rp->data,
            (const uint16_t *) t_col->data,  (const uint16_t *) t_w->data,
            (const float    *) x->data,      (float *) dst->data,
            mat, (uint32_t) rows, (uint32_t) K);
    }
}
