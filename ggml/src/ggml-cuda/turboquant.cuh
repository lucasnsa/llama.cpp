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

template <ggml_type type_K>
__launch_bounds__(128, 1)
static __global__ void k_turboquant_prepare_q(
        const char * src, float * dst, const int ne1, const int ne2,
        const int64_t nb1, const int64_t nb2, const int64_t nb3, const float scale) {
    ggml_cuda_pdl_lc();
    constexpr int D = 256;
    constexpr int planes = type_K == GGML_TYPE_TBQP3_0 ? 2 : 1;
    const int row = blockIdx.x;
    const int col = row%ne1;
    const int head = (row/ne1)%ne2;
    const int sequence = row/(ne1*ne2);
    const float * q = (const float *) (src + col*nb1 + head*nb2 + sequence*nb3);
    float * out = dst + int64_t(row)*D*planes;
    __shared__ float buf[D];
    ggml_cuda_pdl_sync();
    if constexpr (type_K == GGML_TYPE_TURBO2_0) {
        for (int i = threadIdx.x; i < D; i += 128) {
            buf[i] = q[i]*ggml_turbo2_sign(i%QK_TURBO2, false);
        }
        ggml_cuda_turbo2_wht(buf);
        for (int i = threadIdx.x; i < D; i += 128) {
            out[i] = buf[i]*(0.08838834764831845f*ggml_turbo2_sign(i%QK_TURBO2, true))*scale;
        }
    } else {
        for (int i = threadIdx.x; i < D; i += 128) {
            buf[i] = q[i]*ggml_tbq_sign(i, false);
        }
        ggml_cuda_tbq_wht(buf);
        for (int i = threadIdx.x; i < D; i += 128) {
            out[i] = buf[i]*scale/D;
        }
        if constexpr (type_K == GGML_TYPE_TBQP3_0) {
            for (int i = threadIdx.x; i < D; i += 128) {
                buf[i] *= ggml_tbq_sign(i, true);
            }
            ggml_cuda_tbq_wht(buf);
            const float sf = scale*1.2533f/(D*D);
            for (int i = threadIdx.x; i < D; i += 128) {
                out[D + i] = buf[i]*sf;
            }
        }
    }
}

template <ggml_type type_K>
static void ggml_cuda_turboquant_prepare_q(const ggml_tensor * q, float * dst, float scale, cudaStream_t stream) {
    GGML_ASSERT(q->ne[0] == 256);
    const dim3 blocks(ggml_nrows(q));
    const ggml_cuda_kernel_launch_params params = ggml_cuda_kernel_launch_params(blocks, dim3(128), 0, stream);
    ggml_cuda_kernel_launch(k_turboquant_prepare_q<type_K>, params,
        (const char *) q->data, dst, int(q->ne[1]), int(q->ne[2]),
        int64_t(q->nb[1]), int64_t(q->nb[2]), int64_t(q->nb[3]), scale);
    CUDA_CHECK(cudaGetLastError());
}

#endif
