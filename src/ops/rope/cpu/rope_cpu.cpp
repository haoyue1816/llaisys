#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids, float theta,
           size_t seqlen, size_t nhead, size_t d) {
    // RoPE: Rotary Position Embedding
    // For each vector [a, b] -> [a', b'] where a, b are d/2 dimensional
    size_t half_d = d / 2;

    for (size_t seq = 0; seq < seqlen; seq++) {
        int64_t pos = pos_ids[seq];

        for (size_t head = 0; head < nhead; head++) {
            size_t offset = (seq * nhead + head) * d;
            const T *vec = in + offset;
            T *out_vec = out + offset;

            for (size_t j = 0; j < half_d; j++) {
                // Compute angle: phi = pos / theta^(2j/d)
                float exponent = static_cast<float>(2 * j) / d;
                float phi = pos / std::pow(theta, exponent);
                float cos_phi = std::cos(phi);
                float sin_phi = std::sin(phi);

                float a = llaisys::utils::cast<float>(vec[j]);
                float b = llaisys::utils::cast<float>(vec[j + half_d]);

                out_vec[j] = llaisys::utils::cast<T>(a * cos_phi - b * sin_phi);
                out_vec[j + half_d] = llaisys::utils::cast<T>(b * cos_phi + a * sin_phi);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const int64_t *pos_ids, float theta,
           llaisysDataType_t dtype, size_t seqlen, size_t nhead, size_t d) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                     pos_ids, theta, seqlen, nhead, d);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                     pos_ids, theta, seqlen, nhead, d);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                     pos_ids, theta, seqlen, nhead, d);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
