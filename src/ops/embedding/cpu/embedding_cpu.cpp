#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cstring>

template <typename T, typename IndexT>
void embedding_(T *out, const IndexT *index, const T *weight, size_t num_indices, size_t emb_dim) {
    for (size_t i = 0; i < num_indices; i++) {
        IndexT idx = index[i];
        const T *src = weight + idx * emb_dim;
        T *dst = out + i * emb_dim;
        std::memcpy(dst, src, emb_dim * sizeof(T));
    }
}

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
                llaisysDataType_t dtype, size_t num_indices, size_t emb_dim) {
    // Index is always int64
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return embedding_(reinterpret_cast<float *>(out), reinterpret_cast<const int64_t *>(index),
                          reinterpret_cast<const float *>(weight), num_indices, emb_dim);
    case LLAISYS_DTYPE_BF16:
        return embedding_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const int64_t *>(index),
                          reinterpret_cast<const llaisys::bf16_t *>(weight), num_indices, emb_dim);
    case LLAISYS_DTYPE_F16:
        return embedding_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const int64_t *>(index),
                          reinterpret_cast<const llaisys::fp16_t *>(weight), num_indices, emb_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
