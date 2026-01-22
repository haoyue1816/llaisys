#include "linear_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias,
             size_t batch_size, size_t in_features, size_t out_features, bool has_bias) {
    // Compute: out = in @ weight^T + bias
    // weight shape: [out_features, in_features]
    // in shape: [batch_size, in_features]
    // out shape: [batch_size, out_features]

    for (size_t b = 0; b < batch_size; b++) {
        for (size_t o = 0; o < out_features; o++) {
            float sum = 0.0f;
            for (size_t i = 0; i < in_features; i++) {
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    float in_val = llaisys::utils::cast<float>(in[b * in_features + i]);
                    float w_val = llaisys::utils::cast<float>(weight[o * in_features + i]);
                    sum += in_val * w_val;
                } else {
                    sum += in[b * in_features + i] * weight[o * in_features + i];
                }
            }
            if (has_bias) {
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    float bias_val = llaisys::utils::cast<float>(bias[o]);
                    sum += bias_val;
                } else {
                    sum += bias[o];
                }
            }
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                out[b * out_features + o] = llaisys::utils::cast<T>(sum);
            } else {
                out[b * out_features + o] = static_cast<T>(sum);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
             llaisysDataType_t dtype, size_t batch_size, size_t in_features, size_t out_features, bool has_bias) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                       reinterpret_cast<const float *>(weight), reinterpret_cast<const float *>(bias),
                       batch_size, in_features, out_features, has_bias);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                       reinterpret_cast<const llaisys::bf16_t *>(weight), reinterpret_cast<const llaisys::bf16_t *>(bias),
                       batch_size, in_features, out_features, has_bias);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                       reinterpret_cast<const llaisys::fp16_t *>(weight), reinterpret_cast<const llaisys::fp16_t *>(bias),
                       batch_size, in_features, out_features, has_bias);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
