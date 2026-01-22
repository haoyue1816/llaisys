#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, float eps,
                size_t batch_size, size_t hidden_size) {
    // For each row: Y = W * X / sqrt(mean(X^2) + eps)
    for (size_t b = 0; b < batch_size; b++) {
        const T *row_in = in + b * hidden_size;
        T *row_out = out + b * hidden_size;

        // Compute mean of squares
        float sum_sq = 0.0f;
        for (size_t i = 0; i < hidden_size; i++) {
            float val = llaisys::utils::cast<float>(row_in[i]);
            sum_sq += val * val;
        }
        float mean_sq = sum_sq / hidden_size;
        float rms = std::sqrt(mean_sq + eps);

        // Normalize and scale by weight
        for (size_t i = 0; i < hidden_size; i++) {
            float val = llaisys::utils::cast<float>(row_in[i]);
            float w = llaisys::utils::cast<float>(weight[i]);
            row_out[i] = llaisys::utils::cast<T>(w * val / rms);
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, float eps,
               llaisysDataType_t dtype, size_t batch_size, size_t hidden_size) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                         reinterpret_cast<const float *>(weight), eps, batch_size, hidden_size);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                         reinterpret_cast<const llaisys::bf16_t *>(weight), eps, batch_size, hidden_size);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                         reinterpret_cast<const llaisys::fp16_t *>(weight), eps, batch_size, hidden_size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
