#pragma once

#include "common.cuh"
#include "../ggml-turboquant.h"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

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

// All 128 threads use the same shared buffer.
static __device__ __forceinline__ void ggml_cuda_tbq_wht(float * buf) {
    const int tid = threadIdx.y*WARP_SIZE + threadIdx.x;
    __syncthreads();
    for (int len = 1; len < QK_TBQ; len *= 2) {
        const int i = (tid/len)*(2*len) + tid%len;
        const float u = buf[i];
        const float v = buf[i + len];
        buf[i] = u + v;
        buf[i + len] = u - v;
        __syncthreads();
    }
}

#endif
