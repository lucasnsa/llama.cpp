#pragma once

#include "ggml.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
static constexpr bool ggml_type_is_turboquant(enum ggml_type type) {
#else
static inline bool ggml_type_is_turboquant(enum ggml_type type) {
#endif
    return type == GGML_TYPE_TBQ3_0 || type == GGML_TYPE_TBQ4_0 || type == GGML_TYPE_TBQP3_0;
}

#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__MUSACC__)
#define GGML_TBQ_INLINE static __device__ __forceinline__
#else
#define GGML_TBQ_INLINE static inline
#endif

GGML_TBQ_INLINE float ggml_tbq_sign(int i, bool qjl) {
    const uint8_t signs[32] = {
        0xa7,0x3b,0x91,0xf4,0x6d,0xc2,0x58,0x0e,0xb3,0x7f,0x24,0xd6,0x89,0x45,0xea,0x1c,
        0x63,0xaf,0xd8,0x52,0x97,0x0b,0xe1,0x3d,0x76,0xc4,0x19,0xfe,0x4a,0x85,0x2c,0xdb,
    };
    const uint8_t signs_qjl[32] = {
        0xd3,0x4e,0xa8,0x17,0x9c,0x5b,0xe6,0x31,0x72,0xb9,0x0d,0xf5,0x43,0x8a,0x6e,0xc7,
        0x58,0x2f,0x94,0xe1,0xb6,0x3d,0x0a,0x7c,0xc5,0x61,0xd8,0x4f,0xa3,0x97,0x1e,0x85,
    };
    const uint8_t b = qjl ? signs_qjl[i >> 3] : signs[i >> 3];
    return ((b >> (i & 7)) & 1) ? -1.0f : 1.0f;
}

GGML_TBQ_INLINE float ggml_tbq_centroid(int bits, int index) {
    const float c2[4] = {-1.5104f,-0.4528f,0.4528f,1.5104f};
    const float c3[8] = {-2.1520f,-1.3440f,-0.7560f,-0.2451f,0.2451f,0.7560f,1.3440f,2.1520f};
    const float c4[16] = {
        -2.7326f,-2.0690f,-1.6180f,-1.2562f,-0.9424f,-0.6568f,-0.3881f,-0.1284f,
         0.1284f,0.3881f,0.6568f,0.9424f,1.2562f,1.6180f,2.0690f,2.7326f,
    };
    return bits == 2 ? c2[index] : bits == 3 ? c3[index] : c4[index];
}

GGML_TBQ_INLINE int ggml_tbq_index(float x, int bits) {
    const float b2[3] = {-0.9816f,0.0f,0.9816f};
    const float b3[7] = {-1.7480f,-1.0500f,-0.5006f,0.0f,0.5006f,1.0500f,1.7480f};
    const float b4[15] = {
        -2.4008f,-1.8435f,-1.4371f,-1.0993f,-0.7996f,-0.5225f,-0.2583f,0.0f,
         0.2583f,0.5225f,0.7996f,1.0993f,1.4371f,1.8435f,2.4008f,
    };
    const float * boundaries = bits == 2 ? b2 : bits == 3 ? b3 : b4;
    for (int i = 0; i < (1 << bits) - 1; ++i) {
        if (x < boundaries[i]) {
            return i;
        }
    }
    return (1 << bits) - 1;
}

GGML_TBQ_INLINE int ggml_tbq_unpack(const uint8_t * qs, int i, int bits) {
    const int byte = i*bits/8;
    const int shift = i*bits%8;
    unsigned value = qs[byte] >> shift;
    if (shift + bits > 8) {
        value |= (unsigned) qs[byte + 1] << (8 - shift);
    }
    return value & ((1 << bits) - 1);
}

GGML_TBQ_INLINE void ggml_tbq_wht(float * x) {
    for (int len = 1; len < 256; len *= 2) {
        for (int i = 0; i < 256; i += 2*len) {
            for (int j = 0; j < len; ++j) {
                const float u = x[i + j];
                const float v = x[i + j + len];
                x[i + j] = u + v;
                x[i + j + len] = u - v;
            }
        }
    }
}

GGML_TBQ_INLINE void ggml_tbq_encode(const float * x, uint8_t * qs, int bits, float * norm, uint8_t * qjl, float * norm_qjl) {
    float tmp[256];
    float sum = 0.0f;
    for (int i = 0; i < 256; ++i) {
        sum += x[i]*x[i];
    }
    *norm = sqrtf(sum);
    for (int i = 0; i < 256*bits/8; ++i) {
        qs[i] = 0;
    }
    if (qjl) {
        *norm_qjl = 0.0f;
        for (int i = 0; i < 32; ++i) {
            qjl[i] = 0;
        }
    }
    if (*norm < 1e-10f) {
        return;
    }
    const float inv_norm = 1.0f / *norm;
    for (int i = 0; i < 256; ++i) {
        tmp[i] = x[i]*inv_norm*ggml_tbq_sign(i, false);
    }
    ggml_tbq_wht(tmp);
    sum = 0.0f;
    for (int i = 0; i < 256; ++i) {
        const int index = ggml_tbq_index(tmp[i], bits);
        const int byte = i*bits/8;
        const int shift = i*bits%8;
        qs[byte] |= (uint8_t) (index << shift);
        if (shift + bits > 8) {
            qs[byte + 1] |= (uint8_t) (index >> (8 - shift));
        }
        if (qjl) {
            const float residual = tmp[i] - ggml_tbq_centroid(bits, index);
            sum += residual*residual;
            tmp[i] = residual*ggml_tbq_sign(i, true);
        }
    }
    if (qjl) {
        *norm_qjl = sqrtf(sum) * *norm;
        ggml_tbq_wht(tmp);
        for (int i = 0; i < 256; ++i) {
            if (tmp[i] >= 0.0f) {
                qjl[i/8] |= (uint8_t) (1 << (i%8));
            }
        }
    }
}

GGML_TBQ_INLINE void ggml_tbq_decode(const uint8_t * qs, int bits, float norm, const uint8_t * qjl, float norm_qjl, float * y) {
    if (qjl) {
        for (int i = 0; i < 256; ++i) {
            y[i] = ((qjl[i/8] >> (i%8)) & 1) ? 1.0f : -1.0f;
        }
        ggml_tbq_wht(y);
    }
    for (int i = 0; i < 256; ++i) {
        // The second inverse WHT and its 1/256 factor reconstruct the QJL estimator.
        const float correction = qjl ? y[i]*ggml_tbq_sign(i, true)*norm_qjl*(1.2533f/256.0f) : 0.0f;
        y[i] = norm*ggml_tbq_centroid(bits, ggml_tbq_unpack(qs, i, bits)) + correction;
    }
    ggml_tbq_wht(y);
    for (int i = 0; i < 256; ++i) {
        y[i] *= ggml_tbq_sign(i, false)/256.0f;
    }
}

#undef GGML_TBQ_INLINE
