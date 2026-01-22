#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    if (bias) CHECK_SAME_DEVICE(out, bias);
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), "linear: input tensors must be contiguous.");
    ASSERT(in->ndim() == 2 && weight->ndim() == 2 && out->ndim() == 2, "linear: input, weight, and output must be 2D.");
    ASSERT(!bias || bias->ndim() == 1, "linear: bias must be 1D.");

    size_t batch_size = in->shape()[0];
    size_t in_features = in->shape()[1];
    size_t out_features = weight->shape()[0];
    bool has_bias = bias != nullptr;

    // CPU implementation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), weight->data(), has_bias ? bias->data() : nullptr,
                           out->dtype(), batch_size, in_features, out_features, has_bias);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(), has_bias ? bias->data() : nullptr,
                           out->dtype(), batch_size, in_features, out_features, has_bias);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
