#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <vector>

template <typename T>
void self_attention_(T *attn_val, const T *q, const T *k, const T *v, float scale,
                      size_t seqlen, size_t kvlen, size_t nhead, size_t nkvhead, size_t d, size_t dv) {
    // Self-attention: Y = softmax(Q @ K^T * scale) @ V
    // Input shapes: Q [seqlen, nhead, d], K [kvlen, nkvhead, d], V [kvlen, nkvhead, d]
    // Output shape: attn_val [seqlen, nhead, d]
    // For GQA: nhead > nkvhead, multiple query heads share each key-value head

    size_t heads_per_kv = nhead / nkvhead;

    // Process each query head
    for (size_t h = 0; h < nhead; h++) {
        size_t kv_h = h / heads_per_kv;  // Which KV head this query head uses

        // For each query position
        for (size_t i = 0; i < seqlen; i++) {
            std::vector<float> scores(kvlen);

            // Compute Q @ K^T
            // Q: [i, h, dim], K: [j, kv_h, dim]
            for (size_t j = 0; j < kvlen; j++) {
                float sum = 0.0f;
                for (size_t dim = 0; dim < d; dim++) {
                    float q_val = llaisys::utils::cast<float>(q[i * nhead * d + h * d + dim]);
                    float k_val = llaisys::utils::cast<float>(k[j * nkvhead * d + kv_h * d + dim]);
                    sum += q_val * k_val;
                }
                scores[j] = sum * scale;
            }

            // Causal mask + softmax
            // When kvlen > seqlen, can attend to positions up to i + (kvlen - seqlen)
            // When kvlen == seqlen, can only attend to positions <= i
            size_t max_j = i;
            if (kvlen > seqlen) {
                max_j = i + (kvlen - seqlen);
            }

            float max_score = scores[0];
            for (size_t j = 1; j <= max_j; j++) {
                if (scores[j] > max_score) max_score = scores[j];
            }

            float exp_sum = 0.0f;
            for (size_t j = 0; j <= max_j; j++) {
                scores[j] = std::exp(scores[j] - max_score);
                exp_sum += scores[j];
            }

            for (size_t j = 0; j <= max_j; j++) {
                scores[j] /= exp_sum;
            }

            // Compute weighted sum with V
            // V: [j, kv_h, dim] with shape [kvlen, nkvhead, dv]
            for (size_t dim = 0; dim < d; dim++) {
                float val = 0.0f;
                for (size_t j = 0; j <= max_j; j++) {
                    float v_val = llaisys::utils::cast<float>(v[j * nkvhead * dv + kv_h * dv + dim]);
                    val += scores[j] * v_val;
                }
                attn_val[i * nhead * d + h * d + dim] = llaisys::utils::cast<T>(val);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v, float scale,
                     llaisysDataType_t dtype, size_t seqlen, size_t kvlen, size_t nhead, size_t nkvhead, size_t d, size_t dv) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(attn_val), reinterpret_cast<const float *>(q),
                               reinterpret_cast<const float *>(k), reinterpret_cast<const float *>(v),
                               scale, seqlen, kvlen, nhead, nkvhead, d, dv);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t *>(attn_val), reinterpret_cast<const llaisys::bf16_t *>(q),
                               reinterpret_cast<const llaisys::bf16_t *>(k), reinterpret_cast<const llaisys::bf16_t *>(v),
                               scale, seqlen, kvlen, nhead, nkvhead, d, dv);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<llaisys::fp16_t *>(attn_val), reinterpret_cast<const llaisys::fp16_t *>(q),
                               reinterpret_cast<const llaisys::fp16_t *>(k), reinterpret_cast<const llaisys::fp16_t *>(v),
                               scale, seqlen, kvlen, nhead, nkvhead, d, dv);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
