#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <numeric>
#include <sstream>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

bool Tensor::isContiguous() const {
    size_t ndim_ = this->ndim();

    // 0维张量总是连续的
    if (ndim_ == 0) {
        return true;
    }

    // strides 是按元素个数计算的，不是字节数
    // 对于连续张量，最后一维的 stride 应该等于 1（一个元素）
    if (_meta.strides[ndim_ - 1] != 1) {
        return false;
    }

    // 检查其他维度：strides[i] 应该等于 strides[i+1] * shape[i+1]
    for (size_t i = 0; i < ndim_ - 1; i++) {
        ptrdiff_t expected_stride = _meta.strides[i + 1] * static_cast<ptrdiff_t>(_meta.shape[i + 1]);
        if (_meta.strides[i] != expected_stride) {
            return false;
        }
    }

    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    // 步骤1：验证 order 的有效性
    size_t ndim_ = this->ndim();

    // 检查 order 长度是否匹配维度数
    if (order.size() != ndim_) {
        throw std::runtime_error("permute() order length does not match tensor dimensions");
    }

    // 检查 order 是否包含所有维度 [0, 1, 2, ..., ndim-1]
    std::vector<bool> seen(ndim_, false);
    for (size_t i = 0; i < ndim_; i++) {
        if (order[i] >= ndim_) {
            throw std::runtime_error("permute() order contains invalid dimension");
        }
        seen[order[i]] = true;
    }

    for (size_t i = 0; i < ndim_; i++) {
        if (!seen[i]) {
            throw std::runtime_error("permute() order missing dimension");
        }
    }

    // 步骤2：根据 order 重新排列 shape
    std::vector<size_t> new_shape(ndim_);
    for (size_t i = 0; i < ndim_; i++) {
        new_shape[i] = _meta.shape[order[i]];
    }

    // 步骤3：根据 order 重新排列 strides
    std::vector<ptrdiff_t> new_strides(ndim_);
    for (size_t i = 0; i < ndim_; i++) {
        new_strides[i] = _meta.strides[order[i]];
    }

    // 步骤4：创建新的 TensorMeta
    TensorMeta new_meta;
    new_meta.dtype = _meta.dtype;
    new_meta.shape = new_shape;
    new_meta.strides = new_strides;

    // 步骤5：创建新张量，复用同一块 storage（不复制数据）
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    // 步骤1：计算新形状的元素总数
    size_t new_numel = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());

    // 步骤2：检查元素总数是否相同
    if (new_numel != this->numel()) {
        printf("ERROR: view() numel mismatch! Current shape: [");
        for (size_t i = 0; i < _meta.shape.size(); i++) {
            printf("%zu%s", _meta.shape[i], i < _meta.shape.size() - 1 ? ", " : "");
        }
        printf("] = %zu, Requested shape: [", this->numel());
        for (size_t i = 0; i < shape.size(); i++) {
            printf("%zu%s", shape[i], i < shape.size() - 1 ? ", " : "");
        }
        printf("] = %zu\n", new_numel);
        throw std::runtime_error("view() shape does not match total number of elements");
    }

    // 步骤3：计算新的 strides（从后往前）
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> new_strides(ndim_);

    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        new_strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }

    // 步骤4：创建新的 TensorMeta
    TensorMeta new_meta;
    new_meta.dtype = _meta.dtype;
    new_meta.shape = shape;
    new_meta.strides = new_strides;

    // 步骤5：创建新张量，复用同一块 storage（不复制数据）
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    // 步骤1：验证参数
    size_t ndim_ = this->ndim();

    // 检查 dim 是否有效
    if (dim >= ndim_) {
        throw std::runtime_error("slice() dim is out of bounds");
    }

    // 检查 start 是否小于 end
    if (start >= end) {
        throw std::runtime_error("slice() start must be less than end");
    }

    // 检查 end 是否超出范围
    if (end > _meta.shape[dim]) {
        throw std::runtime_error("slice() end is out of bounds");
    }

    // 步骤2：计算新的 shape
    std::vector<size_t> new_shape = _meta.shape;
    new_shape[dim] = end - start;

    // 步骤3：计算新的 offset（字节数）
    // strides 是按元素个数计算的，需要乘以 elementSize 转换为字节数
    size_t new_offset = _offset + start * _meta.strides[dim] * this->elementSize();

    // 步骤4：创建新的 TensorMeta
    // strides 保持不变
    TensorMeta new_meta;
    new_meta.dtype = _meta.dtype;
    new_meta.shape = new_shape;
    new_meta.strides = _meta.strides;

    // 步骤5：创建新张量，使用新的 offset
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, new_offset));
}

void Tensor::load(const void *src_) {
    // 设置设备上下文
    core::context().setDevice(this->deviceType(), this->deviceId());

    // 计算要复制的字节数
    size_t bytes = this->numel() * this->elementSize();

    // 从主机（CPU）复制数据到设备
    core::context().runtime().api()->memcpy_sync(
        this->data(),           // 目标：张量数据
        src_,                   // 源：CPU数据
        bytes,                  // 字节数
        LLAISYS_MEMCPY_H2D      // Host to Device
    );
}

tensor_t Tensor::contiguous() const {
    // 简化实现：如果已经是连续的，返回自己；否则创建连续副本
    if (this->isContiguous()) {
        return std::shared_ptr<Tensor>(new Tensor(_meta, _storage, _offset));
    }
    // TODO: 实现真正的连续化操作
    throw std::runtime_error("contiguous() not fully implemented");
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    // reshape 类似于 view，但可以处理非连续张量
    // 简化实现：只支持连续张量
    if (!this->isContiguous()) {
        throw std::runtime_error("reshape() only implemented for contiguous tensors");
    }
    return this->view(shape);
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    // TODO: 实现设备转换
    throw std::runtime_error("to() not implemented");
}

} // namespace llaisys
