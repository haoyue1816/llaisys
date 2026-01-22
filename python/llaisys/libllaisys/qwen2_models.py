from ctypes import Structure, c_int, c_size_t, c_int64, c_float, c_void_p, POINTER
from .llaisys_types import llaisysDeviceType_t, llaisysDataType_t

# LlaisysQwen2Meta structure
class LlaisysQwen2Meta(Structure):
    _fields_ = [
        ("dtype", llaisysDataType_t),
        ("nlayer", c_size_t),
        ("hs", c_size_t),      # hidden_size
        ("nh", c_size_t),      # num_attention_heads
        ("nkvh", c_size_t),    # num_key_value_heads
        ("dh", c_size_t),      # head_dim
        ("di", c_size_t),      # intermediate_size
        ("maxseq", c_size_t),  # max_sequence_length
        ("voc", c_size_t),     # vocab_size
        ("epsilon", c_float),  # rms_norm_eps
        ("theta", c_float),    # rope_theta
        ("end_token", c_int64),
    ]

# Forward declaration for LlaisysTensor
from .tensor import llaisysTensor_t

# LlaisysQwen2Weights structure
class LlaisysQwen2Weights(Structure):
    pass

LlaisysQwen2Weights._fields_ = [
    ("in_embed", llaisysTensor_t),
    ("out_embed", llaisysTensor_t),
    ("out_norm_w", llaisysTensor_t),
    ("attn_norm_w", POINTER(llaisysTensor_t)),
    ("attn_q_w", POINTER(llaisysTensor_t)),
    ("attn_q_b", POINTER(llaisysTensor_t)),
    ("attn_k_w", POINTER(llaisysTensor_t)),
    ("attn_k_b", POINTER(llaisysTensor_t)),
    ("attn_v_w", POINTER(llaisysTensor_t)),
    ("attn_v_b", POINTER(llaisysTensor_t)),
    ("attn_o_w", POINTER(llaisysTensor_t)),
    ("mlp_norm_w", POINTER(llaisysTensor_t)),
    ("mlp_gate_w", POINTER(llaisysTensor_t)),
    ("mlp_up_w", POINTER(llaisysTensor_t)),
    ("mlp_down_w", POINTER(llaisysTensor_t)),
]

# Pointer types
LlaisysQwen2Model_p = c_void_p
LlaisysQwen2Weights_p = POINTER(LlaisysQwen2Weights)

def load_qwen2_models(lib):
    # llaisysQwen2ModelCreate
    lib.llaisysQwen2ModelCreate.argtypes = [
        POINTER(LlaisysQwen2Meta),  # meta
        llaisysDeviceType_t,        # device
        POINTER(c_int),             # device_ids
        c_int                       # ndevice
    ]
    lib.llaisysQwen2ModelCreate.restype = LlaisysQwen2Model_p

    # llaisysQwen2ModelDestroy
    lib.llaisysQwen2ModelDestroy.argtypes = [LlaisysQwen2Model_p]
    lib.llaisysQwen2ModelDestroy.restype = None

    # llaisysQwen2ModelWeights
    lib.llaisysQwen2ModelWeights.argtypes = [LlaisysQwen2Model_p]
    lib.llaisysQwen2ModelWeights.restype = LlaisysQwen2Weights_p

    # llaisysQwen2ModelInfer
    lib.llaisysQwen2ModelInfer.argtypes = [
        LlaisysQwen2Model_p,  # model
        POINTER(c_int64),     # token_ids
        c_size_t              # ntoken
    ]
    lib.llaisysQwen2ModelInfer.restype = c_int64
