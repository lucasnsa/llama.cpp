#pragma once

#include "common.cuh"
#include "../ggml-turboquant.h"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

static __device__ void quantize_f32_turbo2_0_block(const float * x, block_turbo2_0 * y) {
    float norm;
    ggml_turbo2_encode(x, y->qs, &norm);
    y->d = __float2half(norm);
}

// Extra threads only take part in barriers in the split-KV combiner.
template <int nthreads = 128>
static __device__ __forceinline__ void ggml_cuda_turbo2_wht(float * buf) {
    static_assert(nthreads == 128 || nthreads == 256, "unsupported WHT block size");
    const int tid = threadIdx.y*WARP_SIZE + threadIdx.x;
    __syncthreads();
    for (int len = 1; len < QK_TURBO2; len *= 2) {
        if (nthreads == 128 || tid < 128) {
            const int lane = tid % (QK_TURBO2/2);
            const int i = (tid/(QK_TURBO2/2))*QK_TURBO2 + (lane/len)*(2*len) + lane%len;
            const float u = buf[i];
            const float v = buf[i + len];
            buf[i] = u + v;
            buf[i + len] = u - v;
        }
        __syncthreads();
    }
}

static __device__ void quantize_f32_tbq3_0_block(const float * x, block_tbq3_0 * y) {
    float norm, norm_qjl;
    ggml_tbq_encode(x, y->qs, 3, &norm, nullptr, &norm_qjl);
    y->d = __float2half(norm);
}

static __device__ void quantize_f32_tbq4_0_block(const float * x, block_tbq4_0 * y) {
    float norm, norm_qjl;
    ggml_tbq_encode(x, y->qs, 4, &norm, nullptr, &norm_qjl);
    y->d = __float2half(norm);
}

static __device__ void quantize_f32_tbqp3_0_block(const float * x, block_tbqp3_0 * y) {
    float norm, norm_qjl;
    ggml_tbq_encode(x, y->qs, 2, &norm, y->qjl, &norm_qjl);
    y->d = __float2half(norm);
    y->d_qjl = __float2half(norm_qjl);
}

// The first 128 threads update the shared buffer.
template <int nthreads = 128>
static __device__ __forceinline__ void ggml_cuda_tbq_wht(float * buf) {
    static_assert(nthreads == 128 || nthreads == 256, "unsupported WHT block size");
    const int tid = threadIdx.y*WARP_SIZE + threadIdx.x;
    __syncthreads();
    for (int len = 1; len < QK_TBQ; len *= 2) {
        if (nthreads == 128 || tid < 128) {
            const int i = (tid/len)*(2*len) + tid%len;
            const float u = buf[i];
            const float v = buf[i + len];
            buf[i] = u + v;
            buf[i + len] = u - v;
        }
        __syncthreads();
    }
}

#endif
