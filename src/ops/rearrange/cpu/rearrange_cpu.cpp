#include "rearrange_cpu.hpp"

#include "../../../utils.hpp"

#include <cstring>
#include <vector>
#include <numeric>

template <typename T>
void rearrange_(T *out, const T *in, size_t numel,
                 const ptrdiff_t *out_strides, const ptrdiff_t *in_strides,
                 const size_t *shape, size_t ndim) {
    // Rearrange data from input tensor to output tensor with different strides
    // Both tensors have the same shape but potentially different memory layouts

    if (ndim == 0) {
        // Scalar case
        if (numel == 1) {
            out[0] = in[0];
        }
        return;
    }

    // Compute total elements
    size_t total_elements = 1;
    for (size_t i = 0; i < ndim; i++) {
        total_elements *= shape[i];
    }

    if (total_elements != numel) {
        // Edge case: just copy element-by-element
        std::memcpy(out, in, numel * sizeof(T));
        return;
    }

    // For non-contiguous tensors, iterate through each element
    // using strides to compute memory offsets
    std::vector<size_t> out_indices(ndim, 0);
    std::vector<size_t> in_indices(ndim, 0);

    for (size_t elem = 0; elem < total_elements; elem++) {
        // Compute input offset
        size_t in_offset = 0;
        for (size_t dim = 0; dim < ndim; dim++) {
            in_offset += in_indices[dim] * static_cast<size_t>(in_strides[dim]);
        }

        // Compute output offset
        size_t out_offset = 0;
        for (size_t dim = 0; dim < ndim; dim++) {
            out_offset += out_indices[dim] * static_cast<size_t>(out_strides[dim]);
        }

        // Copy element
        out[out_offset] = in[in_offset];

        // Increment indices (like odometer)
        for (int dim = static_cast<int>(ndim) - 1; dim >= 0; dim--) {
            in_indices[dim]++;
            out_indices[dim]++;
            if (out_indices[dim] < shape[dim]) {
                break;
            }
            in_indices[dim] = 0;
            out_indices[dim] = 0;
        }
    }
}

namespace llaisys::ops::cpu {
void rearrange(std::byte *out, const std::byte *in,
                llaisysDataType_t dtype, size_t numel,
                const ptrdiff_t *out_strides, const ptrdiff_t *in_strides,
                const size_t *shape, size_t ndim) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rearrange_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                          numel, out_strides, in_strides, shape, ndim);
    case LLAISYS_DTYPE_BF16:
        return rearrange_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                          numel, out_strides, in_strides, shape, ndim);
    case LLAISYS_DTYPE_F16:
        return rearrange_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                          numel, out_strides, in_strides, shape, ndim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
