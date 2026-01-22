#pragma once

#include "llaisys/models/qwen2.h"
#include "../../tensor/tensor.hpp"

// Include all operator headers
#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"
#include "../../ops/rearrange/op.hpp"

#include <vector>
#include <memory>

namespace llaisys::models {

class Qwen2Model {
public:
    // Constructor with metadata
    Qwen2Model(const LlaisysQwen2Meta &meta,
               llaisysDeviceType_t device_type,
               int device_id);

    // Destructor
    ~Qwen2Model();

    // Get weights structure pointer
    LlaisysQwen2Weights* getWeights() { return &weights_; }

    // Inference: forward pass and return next token
    // token_ids: input token sequence
    // ntoken: number of tokens in the sequence
    // Returns: predicted next token ID
    int64_t infer(const int64_t *token_ids, size_t ntoken);

    // Reset KV cache (for new generation)
    void resetCache();

    // Get metadata
    const LlaisysQwen2Meta& getMeta() const { return meta_; }

private:
    // Model metadata
    LlaisysQwen2Meta meta_;

    // Device
    llaisysDeviceType_t device_type_;
    int device_id_;

    // Weights structure
    LlaisysQwen2Weights weights_;

    // KV Cache: [nlayer * 2] tensors
    // Index layer * 2 + 0: K cache for that layer
    // Index layer * 2 + 1: V cache for that layer
    std::vector<tensor_t> kv_cache_;

    // Current cache length (number of tokens cached)
    size_t cache_len_;

    // Helper methods
    void allocateWeights();
    void freeWeights();
    void allocateCache();
    void freeCache();

    // Single transformer layer forward
    tensor_t transformerLayer(tensor_t input, size_t layer_idx, size_t seqlen);

    // Self-attention block
    tensor_t attentionBlock(tensor_t input, size_t layer_idx, size_t seqlen);

    // MLP block
    tensor_t mlpBlock(tensor_t input, size_t layer_idx);

    // Helper: 3D linear for [1, seqlen, in_features] -> [1, seqlen, out_features]
    tensor_t linear3d(tensor_t input, tensor_t weight, tensor_t bias);
    tensor_t linear3d(tensor_t input, tensor_t weight);
};

} // namespace llaisys::models
