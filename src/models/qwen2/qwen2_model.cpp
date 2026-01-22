#include "qwen2_model.hpp"
#include "../../utils.hpp"
#include "../../llaisys/llaisys_tensor.hpp"
#include <cmath>
#include <cstring>

namespace llaisys::models {

Qwen2Model::Qwen2Model(const LlaisysQwen2Meta &meta,
                       llaisysDeviceType_t device_type,
                       int device_id)
    : meta_(meta), device_type_(device_type), device_id_(device_id), cache_len_(0) {
    allocateWeights();
    allocateCache();
}

Qwen2Model::~Qwen2Model() {
    freeWeights();
    freeCache();
}

void Qwen2Model::allocateWeights() {
    // Allocate all weight tensors
    // Input embedding
    weights_.in_embed = new LlaisysTensor{
        Tensor::create({meta_.voc, meta_.hs}, meta_.dtype, device_type_, device_id_)
    };

    // Output embedding (lm_head)
    weights_.out_embed = new LlaisysTensor{
        Tensor::create({meta_.voc, meta_.hs}, meta_.dtype, device_type_, device_id_)
    };

    // Output norm
    weights_.out_norm_w = new LlaisysTensor{
        Tensor::create({meta_.hs}, meta_.dtype, device_type_, device_id_)
    };

    // Per-layer weights
    weights_.attn_norm_w = new llaisysTensor_t[meta_.nlayer];
    weights_.attn_q_w = new llaisysTensor_t[meta_.nlayer];
    weights_.attn_q_b = new llaisysTensor_t[meta_.nlayer];
    weights_.attn_k_w = new llaisysTensor_t[meta_.nlayer];
    weights_.attn_k_b = new llaisysTensor_t[meta_.nlayer];
    weights_.attn_v_w = new llaisysTensor_t[meta_.nlayer];
    weights_.attn_v_b = new llaisysTensor_t[meta_.nlayer];
    weights_.attn_o_w = new llaisysTensor_t[meta_.nlayer];

    weights_.mlp_norm_w = new llaisysTensor_t[meta_.nlayer];
    weights_.mlp_gate_w = new llaisysTensor_t[meta_.nlayer];
    weights_.mlp_up_w = new llaisysTensor_t[meta_.nlayer];
    weights_.mlp_down_w = new llaisysTensor_t[meta_.nlayer];

    for (size_t i = 0; i < meta_.nlayer; i++) {
        // Attention weights
        weights_.attn_norm_w[i] = new LlaisysTensor{
            Tensor::create({meta_.hs}, meta_.dtype, device_type_, device_id_)
        };

        // Q projection
        weights_.attn_q_w[i] = new LlaisysTensor{
            Tensor::create({meta_.hs, meta_.hs}, meta_.dtype, device_type_, device_id_)
        };
        weights_.attn_q_b[i] = new LlaisysTensor{
            Tensor::create({meta_.hs}, meta_.dtype, device_type_, device_id_)
        };

        // K projection: [hs_per_kv_head * nkvh, hs] for GQA
        // In Qwen2 with GQA: K and V have smaller output dimension
        // hs_per_kv_head = hs / nh = dh
        weights_.attn_k_w[i] = new LlaisysTensor{
            Tensor::create({meta_.dh * meta_.nkvh, meta_.hs}, meta_.dtype, device_type_, device_id_)
        };
        weights_.attn_k_b[i] = new LlaisysTensor{
            Tensor::create({meta_.dh * meta_.nkvh}, meta_.dtype, device_type_, device_id_)
        };

        // V projection: [hs_per_kv_head * nkvh, hs] for GQA
        weights_.attn_v_w[i] = new LlaisysTensor{
            Tensor::create({meta_.dh * meta_.nkvh, meta_.hs}, meta_.dtype, device_type_, device_id_)
        };
        weights_.attn_v_b[i] = new LlaisysTensor{
            Tensor::create({meta_.dh * meta_.nkvh}, meta_.dtype, device_type_, device_id_)
        };

        // O projection
        weights_.attn_o_w[i] = new LlaisysTensor{
            Tensor::create({meta_.hs, meta_.hs}, meta_.dtype, device_type_, device_id_)
        };

        // MLP weights
        weights_.mlp_norm_w[i] = new LlaisysTensor{
            Tensor::create({meta_.hs}, meta_.dtype, device_type_, device_id_)
        };

        // Gate projection
        weights_.mlp_gate_w[i] = new LlaisysTensor{
            Tensor::create({meta_.di, meta_.hs}, meta_.dtype, device_type_, device_id_)
        };

        // Up projection
        weights_.mlp_up_w[i] = new LlaisysTensor{
            Tensor::create({meta_.di, meta_.hs}, meta_.dtype, device_type_, device_id_)
        };

        // Down projection
        weights_.mlp_down_w[i] = new LlaisysTensor{
            Tensor::create({meta_.hs, meta_.di}, meta_.dtype, device_type_, device_id_)
        };
    }
}

void Qwen2Model::freeWeights() {
    if (weights_.in_embed) {
        delete weights_.in_embed;
        weights_.in_embed = nullptr;
    }
    if (weights_.out_embed) {
        delete weights_.out_embed;
        weights_.out_embed = nullptr;
    }
    if (weights_.out_norm_w) {
        delete weights_.out_norm_w;
        weights_.out_norm_w = nullptr;
    }

    if (weights_.attn_norm_w) {
        for (size_t i = 0; i < meta_.nlayer; i++) {
            if (weights_.attn_norm_w[i]) delete weights_.attn_norm_w[i];
            if (weights_.attn_q_w[i]) delete weights_.attn_q_w[i];
            if (weights_.attn_q_b[i]) delete weights_.attn_q_b[i];
            if (weights_.attn_k_w[i]) delete weights_.attn_k_w[i];
            if (weights_.attn_k_b[i]) delete weights_.attn_k_b[i];
            if (weights_.attn_v_w[i]) delete weights_.attn_v_w[i];
            if (weights_.attn_v_b[i]) delete weights_.attn_v_b[i];
            if (weights_.attn_o_w[i]) delete weights_.attn_o_w[i];
            if (weights_.mlp_norm_w[i]) delete weights_.mlp_norm_w[i];
            if (weights_.mlp_gate_w[i]) delete weights_.mlp_gate_w[i];
            if (weights_.mlp_up_w[i]) delete weights_.mlp_up_w[i];
            if (weights_.mlp_down_w[i]) delete weights_.mlp_down_w[i];
        }
        delete[] weights_.attn_norm_w;
        delete[] weights_.attn_q_w;
        delete[] weights_.attn_q_b;
        delete[] weights_.attn_k_w;
        delete[] weights_.attn_k_b;
        delete[] weights_.attn_v_w;
        delete[] weights_.attn_v_b;
        delete[] weights_.attn_o_w;
        delete[] weights_.mlp_norm_w;
        delete[] weights_.mlp_gate_w;
        delete[] weights_.mlp_up_w;
        delete[] weights_.mlp_down_w;

        weights_.attn_norm_w = nullptr;
        weights_.attn_q_w = nullptr;
        weights_.attn_q_b = nullptr;
        weights_.attn_k_w = nullptr;
        weights_.attn_k_b = nullptr;
        weights_.attn_v_w = nullptr;
        weights_.attn_v_b = nullptr;
        weights_.attn_o_w = nullptr;
        weights_.mlp_norm_w = nullptr;
        weights_.mlp_gate_w = nullptr;
        weights_.mlp_up_w = nullptr;
        weights_.mlp_down_w = nullptr;
    }
}

void Qwen2Model::allocateCache() {
    // KV Cache: [nlayer][2][nkvhead][maxseq][head_dim]
    kv_cache_.resize(meta_.nlayer * 2);  // nlayer × 2 (K and V)

    for (size_t layer = 0; layer < meta_.nlayer; layer++) {
        // K cache: [nkvhead, maxseq, head_dim]
        kv_cache_[layer * 2 + 0] = Tensor::create(
            {meta_.nkvh, meta_.maxseq, meta_.dh},
            meta_.dtype, device_type_, device_id_
        );

        // V cache: [nkvhead, maxseq, head_dim]
        kv_cache_[layer * 2 + 1] = Tensor::create(
            {meta_.nkvh, meta_.maxseq, meta_.dh},
            meta_.dtype, device_type_, device_id_
        );
    }
}

void Qwen2Model::freeCache() {
    kv_cache_.clear();
    cache_len_ = 0;
}

void Qwen2Model::resetCache() {
    cache_len_ = 0;
}

tensor_t Qwen2Model::attentionBlock(tensor_t input, size_t layer_idx, size_t seqlen) {
    // 1. RMS Norm
    // Reshape 3D [1, seqlen, hs] -> 2D [seqlen, hs] for rms_norm
    tensor_t input_2d = input->view({seqlen, meta_.hs});
    tensor_t normalized_2d = Tensor::create({seqlen, meta_.hs}, input->dtype(), device_type_, device_id_);
    ops::rms_norm(normalized_2d, input_2d, weights_.attn_norm_w[layer_idx]->tensor, meta_.epsilon);
    tensor_t normalized = normalized_2d->view({1, seqlen, meta_.hs});

    // 2. Q, K, V projections
    // Q, K, V: [1, seqlen, hs]
    tensor_t q_proj = linear3d(normalized,
                                weights_.attn_q_w[layer_idx]->tensor,
                                weights_.attn_q_b[layer_idx]->tensor);
    tensor_t k_proj = linear3d(normalized,
                                weights_.attn_k_w[layer_idx]->tensor,
                                weights_.attn_k_b[layer_idx]->tensor);
    tensor_t v_proj = linear3d(normalized,
                                weights_.attn_v_w[layer_idx]->tensor,
                                weights_.attn_v_b[layer_idx]->tensor);

    // 3. Reshape to multi-head
    // q: [seqlen, nhead, dh]
    // k, v: [seqlen, nkvhead, dh]
    tensor_t q = q_proj->view({seqlen, meta_.nh, meta_.dh});
    tensor_t k = k_proj->view({seqlen, meta_.nkvh, meta_.dh});
    tensor_t v = v_proj->view({seqlen, meta_.nkvh, meta_.dh});

    // 4. RoPE position encoding
    // Create position IDs tensor
    std::vector<int64_t> pos_ids(seqlen);
    for (size_t i = 0; i < seqlen; i++) {
        pos_ids[i] = static_cast<int64_t>(cache_len_ + i);
    }
    tensor_t positions = Tensor::create({seqlen}, llaisysDataType_t::LLAISYS_DTYPE_I64, device_type_, device_id_);
    positions->load(pos_ids.data());

    // Apply RoPE to Q (q has shape [seqlen, nh, dh])
    tensor_t q_rope = Tensor::create(q->shape(), q->dtype(), device_type_, device_id_);
    ops::rope(q_rope, q, positions, meta_.theta);

    // Apply RoPE to K (k has shape [seqlen, nkvh, dh])
    tensor_t k_rope = Tensor::create(k->shape(), k->dtype(), device_type_, device_id_);
    ops::rope(k_rope, k, positions, meta_.theta);

    // 5. Update KV cache
    // Copy new K (after RoPE), V to cache
    // Cache shape: [nkvhead, maxseq, dh]
    // New k, v shape: [seqlen, nkvhead, dh]

    // Permute k_rope and v to [nkvhead, seqlen, dh] for copying
    tensor_t k_perm = k_rope->permute({1, 0, 2});  // [seqlen, nkvhead, dh] -> [nkvhead, seqlen, dh]
    tensor_t v_perm = v->permute({1, 0, 2});

    // Slice cache and copy new data
    tensor_t k_cache_slice = kv_cache_[layer_idx * 2 + 0]->slice(1, cache_len_, cache_len_ + seqlen);
    tensor_t v_cache_slice = kv_cache_[layer_idx * 2 + 1]->slice(1, cache_len_, cache_len_ + seqlen);

    // Copy data (need to use rearrange or memcpy)
    ops::rearrange(k_cache_slice, k_perm);
    ops::rearrange(v_cache_slice, v_perm);

    // 6. Self-attention with cached K, V
    // kvlen = cache_len + seqlen
    size_t kvlen = cache_len_ + seqlen;

    // Slice cache for attention
    tensor_t k_cached = kv_cache_[layer_idx * 2 + 0]->slice(1, 0, kvlen);    // [nkvhead, kvlen, dh]
    tensor_t v_cached = kv_cache_[layer_idx * 2 + 1]->slice(1, 0, kvlen);    // [nkvhead, kvlen, dh]

    // Permute back to [kvlen, nkvhead, dh]
    tensor_t k_attn = k_cached->permute({1, 0, 2});
    tensor_t v_attn = v_cached->permute({1, 0, 2});

    // Make sure all tensors are contiguous for self_attention
    tensor_t q_rope_contig = q_rope;
    tensor_t k_attn_contig = k_attn;
    tensor_t v_attn_contig = v_attn;

    if (!q_rope->isContiguous()) {
        q_rope_contig = Tensor::create(q_rope->shape(), q_rope->dtype(), device_type_, device_id_);
        ops::rearrange(q_rope_contig, q_rope);
    }
    if (!k_attn->isContiguous()) {
        k_attn_contig = Tensor::create(k_attn->shape(), k_attn->dtype(), device_type_, device_id_);
        ops::rearrange(k_attn_contig, k_attn);
    }
    if (!v_attn->isContiguous()) {
        v_attn_contig = Tensor::create(v_attn->shape(), v_attn->dtype(), device_type_, device_id_);
        ops::rearrange(v_attn_contig, v_attn);
    }

    // Output tensor
    tensor_t attn_out = Tensor::create({seqlen, meta_.nh, meta_.dh}, meta_.dtype, device_type_, device_id_);

    // Self-attention
    float scale = 1.0f / std::sqrt(static_cast<float>(meta_.dh));
    ops::self_attention(attn_out, q_rope_contig, k_attn_contig, v_attn_contig, scale);

    // 7. Reshape and output projection
    // [seqlen, nh, dh] -> [1, seqlen, hs]
    // Debug: check shapes
    size_t attn_out_numel = attn_out->numel();
    size_t target_numel = seqlen * meta_.hs;
    if (attn_out_numel != target_numel) {
        printf("ERROR: attn_out numel mismatch! attn_out: [%zu, %zu, %zu] = %zu, target: [1, %zu, %zu] = %zu\n",
               attn_out->shape()[0], attn_out->shape()[1], attn_out->shape()[2], attn_out_numel,
               seqlen, meta_.hs, target_numel);
        printf("  nh=%zu, dh=%zu, hs=%zu\n", meta_.nh, meta_.dh, meta_.hs);
    }
    tensor_t attn_reshaped = attn_out->view({1, seqlen, meta_.hs});
    tensor_t output = linear3d(attn_reshaped,
                             weights_.attn_o_w[layer_idx]->tensor,
                             nullptr);  // No bias for output projection

    return output;
}

tensor_t Qwen2Model::mlpBlock(tensor_t input, size_t layer_idx) {
    size_t seqlen = input->shape()[1];

    // 1. RMS Norm
    // Reshape 3D [1, seqlen, hs] -> 2D [seqlen, hs] for rms_norm
    tensor_t input_2d = input->view({seqlen, meta_.hs});
    tensor_t normalized_2d = Tensor::create({seqlen, meta_.hs}, input->dtype(), device_type_, device_id_);
    ops::rms_norm(normalized_2d, input_2d, weights_.mlp_norm_w[layer_idx]->tensor, meta_.epsilon);
    tensor_t normalized = normalized_2d->view({1, seqlen, meta_.hs});

    // 2. Gate and Up projections
    tensor_t gate = linear3d(normalized,
                             weights_.mlp_gate_w[layer_idx]->tensor,
                             nullptr);
    tensor_t up = linear3d(normalized,
                           weights_.mlp_up_w[layer_idx]->tensor,
                           nullptr);

    // 3. SwiGLU activation
    tensor_t activated = Tensor::create({1, seqlen, meta_.di}, meta_.dtype, device_type_, device_id_);
    ops::swiglu(activated, gate, up);

    // 4. Down projection
    tensor_t output = linear3d(activated,
                                weights_.mlp_down_w[layer_idx]->tensor,
                                nullptr);

    return output;
}

tensor_t Qwen2Model::transformerLayer(tensor_t input, size_t layer_idx, size_t seqlen) {
    // Attention block + residual
    tensor_t attn_out = attentionBlock(input, layer_idx, seqlen);
    tensor_t residual1 = Tensor::create(input->shape(), input->dtype(), device_type_, device_id_);
    ops::add(residual1, input, attn_out);

    // MLP block + residual
    tensor_t mlp_out = mlpBlock(residual1, layer_idx);
    tensor_t output = Tensor::create(residual1->shape(), residual1->dtype(), device_type_, device_id_);
    ops::add(output, residual1, mlp_out);

    return output;
}

int64_t Qwen2Model::infer(const int64_t *token_ids, size_t ntoken) {
    // 1. Embedding
    // Convert int64 tokens to tensor
    tensor_t tokens = Tensor::create({ntoken}, llaisysDataType_t::LLAISYS_DTYPE_I64, device_type_, device_id_);
    tokens->load(token_ids);

    // Embedding output is 2D [ntoken, hs], then reshape to 3D [1, ntoken, hs]
    tensor_t x_embed = Tensor::create({ntoken, meta_.hs}, meta_.dtype, device_type_, device_id_);
    ops::embedding(x_embed, tokens, weights_.in_embed->tensor);
    tensor_t x = x_embed->view({1, ntoken, meta_.hs});

    // 2. Pass through transformer layers
    for (size_t layer = 0; layer < meta_.nlayer; layer++) {
        x = transformerLayer(x, layer, ntoken);
    }

    // 3. Final RMS norm
    // Reshape 3D [1, ntoken, hs] -> 2D [ntoken, hs] for rms_norm
    tensor_t x_2d = x->view({ntoken, meta_.hs});
    tensor_t normalized_2d = Tensor::create({ntoken, meta_.hs}, x->dtype(), device_type_, device_id_);
    ops::rms_norm(normalized_2d, x_2d, weights_.out_norm_w->tensor, meta_.epsilon);
    tensor_t normalized = normalized_2d->view({1, ntoken, meta_.hs});

    // 4. Output projection to logits
    // Get last token only: [1, ntoken, hs] -> [1, 1, hs]
    tensor_t last_token = normalized->slice(1, ntoken - 1, ntoken);

    // Project to vocab: [1, 1, hs] -> [1, 1, voc]
    tensor_t logits = linear3d(last_token, weights_.out_embed->tensor, nullptr);

    // 5. Argmax to get next token
    // logits shape: [1, 1, voc]
    // Slice creates [1, 1, voc] tensor - squeeze first dimension to get [1, voc], then view to [voc]
    tensor_t logits_2d = logits->view({1, meta_.voc});
    tensor_t last_logits = logits_2d->slice(0, 0, 1);  // [1, voc] -> [1, voc]
    last_logits = last_logits->view({meta_.voc});       // [1, voc] -> [voc]

    tensor_t argmax_out = Tensor::create({}, llaisysDataType_t::LLAISYS_DTYPE_I64, device_type_, device_id_);
    tensor_t max_val = Tensor::create({}, meta_.dtype, device_type_, device_id_);
    ops::argmax(argmax_out, max_val, last_logits);

    // Read result
    int64_t next_token = -1;
    if (device_type_ == LLAISYS_DEVICE_CPU) {
        std::memcpy(&next_token, argmax_out->data(), sizeof(int64_t));
    }

    // 6. Update cache length
    cache_len_ += ntoken;

    return next_token;
}

tensor_t Qwen2Model::linear3d(tensor_t input, tensor_t weight, tensor_t bias) {
    // Helper function for 3D linear: [1, seqlen, in_feat] -> [1, seqlen, out_feat]
    // Reshape 3D -> 2D, call linear, then reshape back
    size_t seqlen = input->shape()[1];
    size_t out_feat = weight->shape()[0];

    // If input is not contiguous, make it contiguous first
    tensor_t input_contig = input;
    if (!input->isContiguous()) {
        input_contig = Tensor::create(input->shape(), input->dtype(), device_type_, device_id_);
        ops::rearrange(input_contig, input);
    }

    // Reshape: [1, seqlen, in] -> [seqlen, in]
    tensor_t input_2d = input_contig->view({seqlen, input_contig->shape()[2]});
    tensor_t out_2d = Tensor::create({seqlen, out_feat}, input->dtype(), device_type_, device_id_);

    ops::linear(out_2d, input_2d, weight, bias);
    return out_2d->view({1, seqlen, out_feat});
}

tensor_t Qwen2Model::linear3d(tensor_t input, tensor_t weight) {
    return linear3d(input, weight, nullptr);
}

} // namespace llaisys::models
