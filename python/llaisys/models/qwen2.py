from typing import Sequence
import json
import numpy as np
from pathlib import Path
import safetensors

from ..libllaisys import LIB_LLAISYS, DeviceType, DataType
from ..libllaisys.qwen2_models import LlaisysQwen2Meta
from ..libllaisys.qwen2_models import LlaisysQwen2Model_p
from ..libllaisys.llaisys_types import llaisysDeviceType_t, llaisysDataType_t
from ctypes import c_int, c_int64, POINTER, byref


class Qwen2:

    def __init__(self, model_path, device=None):
        if device is None:
            device = DeviceType.CPU

        model_path = Path(model_path)

        # Load config.json
        config_path = model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            config = json.load(f)

        # Extract model metadata
        meta = LlaisysQwen2Meta()

        meta.dtype = llaisysDataType_t(DataType.BF16)
        meta.nlayer = config.get("num_hidden_layers", 28)
        meta.hs = config.get("hidden_size", 1536)
        meta.nh = config.get("num_attention_heads", 12)
        meta.nkvh = config.get("num_key_value_heads", 2)
        meta.dh = meta.hs // meta.nh  # head_dim = hidden_size / num_heads
        meta.di = config.get("intermediate_size", 8960)
        meta.maxseq = config.get("max_position_embeddings", 131072)
        # Use sliding_window if available, otherwise max_position_embeddings
        if "sliding_window" in config:
            meta.maxseq = config["sliding_window"]
        meta.voc = config.get("vocab_size", 151936)
        meta.epsilon = float(config.get("rms_norm_eps", 1e-6))
        meta.theta = float(config.get("rope_theta", 10000))
        meta.end_token = c_int64(config.get("eos_token_id", 151643))

        # Create C model
        device_type = llaisysDeviceType_t(device)
        device_id = c_int(0)

        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            byref(meta),
            device_type,
            byref(device_id),
            1
        )

        if not self._model:
            raise RuntimeError("Failed to create Qwen2 model")

        # Get weights structure
        weights_ptr = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model)

        # Load all safetensors files
        weight_map = {}
        safetensors_files = list(sorted(model_path.glob("*.safetensors")))

        for file in safetensors_files:
            # Load file metadata and get raw bytes
            with open(file, 'rb') as f:
                # Read safetensors header
                import struct
                header_len_bytes = f.read(8)
                header_len = struct.unpack('<Q', header_len_bytes)[0]
                header_json = f.read(header_len).decode('utf-8')
                header = json.loads(header_json)

                # Read each tensor
                for name, tensor_info in header.items():
                    # tensor_info contains shape, dtype, data_offsets
                    shape = tensor_info.get('shape', [])
                    data_offsets = tensor_info.get('data_offsets', [0, 0])
                    data_start = data_offsets[0]
                    data_end = data_offsets[1]

                    # Calculate position in file (after header)
                    f.seek(8 + header_len + data_start)

                    # Read raw bytes
                    tensor_bytes = f.read(data_end - data_start)

                    # Store as bytes - will be loaded directly into C++
                    weight_map[name] = tensor_bytes

        def load_tensor(name: str, tensor_ptr):
            """Helper to load a single tensor"""
            if name not in weight_map:
                raise ValueError(f"Weight not found: {name}")

            # Get the raw bytes directly
            tensor_bytes = weight_map[name]

            # Load into C tensor
            LIB_LLAISYS.tensorLoad(tensor_ptr, tensor_bytes)

        # Input embedding
        load_tensor("model.embed_tokens.weight", weights_ptr.contents.in_embed)

        # Output embedding (lm_head)
        load_tensor("lm_head.weight", weights_ptr.contents.out_embed)

        # Output norm
        load_tensor("model.norm.weight", weights_ptr.contents.out_norm_w)

        # Per-layer weights
        for i in range(meta.nlayer):
            layer_prefix = f"model.layers.{i}."

            # Attention norm
            load_tensor(layer_prefix + "input_layernorm.weight",
                       weights_ptr.contents.attn_norm_w[i])

            # Q projection
            load_tensor(layer_prefix + "self_attn.q_proj.weight",
                       weights_ptr.contents.attn_q_w[i])
            load_tensor(layer_prefix + "self_attn.q_proj.bias",
                       weights_ptr.contents.attn_q_b[i])

            # K projection
            load_tensor(layer_prefix + "self_attn.k_proj.weight",
                       weights_ptr.contents.attn_k_w[i])
            load_tensor(layer_prefix + "self_attn.k_proj.bias",
                       weights_ptr.contents.attn_k_b[i])

            # V projection
            load_tensor(layer_prefix + "self_attn.v_proj.weight",
                       weights_ptr.contents.attn_v_w[i])
            load_tensor(layer_prefix + "self_attn.v_proj.bias",
                       weights_ptr.contents.attn_v_b[i])

            # O projection
            load_tensor(layer_prefix + "self_attn.o_proj.weight",
                       weights_ptr.contents.attn_o_w[i])

            # MLP norm
            load_tensor(layer_prefix + "post_attention_layernorm.weight",
                       weights_ptr.contents.mlp_norm_w[i])

            # Gate projection
            load_tensor(layer_prefix + "mlp.gate_proj.weight",
                       weights_ptr.contents.mlp_gate_w[i])

            # Up projection
            load_tensor(layer_prefix + "mlp.up_proj.weight",
                       weights_ptr.contents.mlp_up_w[i])

            # Down projection
            load_tensor(layer_prefix + "mlp.down_proj.weight",
                       weights_ptr.contents.mlp_down_w[i])

        # Store metadata
        self._meta = meta
        self._device = device

    def __del__(self):
        if hasattr(self, "_model") and self._model is not None:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
            self._model = None

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        # Default max_new_tokens
        if max_new_tokens is None:
            max_new_tokens = 128

        # Convert inputs to int64 array
        tokens = list(inputs)
        n_tokens = len(tokens)

        # Store input tokens for final output
        input_tokens = list(tokens)

        # For now, implement simple greedy decoding (top_k=1)
        # TODO: implement top-k and top-p sampling

        output_tokens = []

        for step in range(max_new_tokens):
            # Prepare token array
            token_arr = (c_int64 * n_tokens)(*tokens)

            # Run inference
            try:
                next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
                    self._model,
                    token_arr,
                    n_tokens
                )
            except Exception as e:
                print(f"[ERROR] Exception in llaisysQwen2ModelInfer: {e}")
                raise

            # Check for EOS
            if next_token == self._meta.end_token:
                output_tokens.append(next_token)  # Include EOS token in output
                break

            output_tokens.append(next_token)

            # For next iteration, only pass the new token (KV-Cache handles history)
            tokens = [next_token]
            n_tokens = 1

            # Safety check
            if len(output_tokens) >= max_new_tokens:
                break

        # Return full sequence: input_tokens + output_tokens
        return input_tokens + output_tokens
