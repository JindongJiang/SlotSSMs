import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Any
from dataclasses import dataclass, field

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import ModelOutput, PreTrainedModel
from mamba.mamba_ssm import Mamba, Mamba2
from flash_attn.modules.mha import MHA as FlashMHA

from src.utils import mprint
from module import MultiHeadAttention

@dataclass
class MambaCache:
    """Inference parameters that are passed to the main model in order
    to efficienly calculate and store the context during inference."""
    seqlen_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)

class SlotSSMConfig(PretrainedConfig):
    model_type = "slot_ssm"
    
    def __init__(
        self,
        num_slots: int = 32,
        num_blocks: int = 4,
        d_model: int = 256,
        use_cross_attn: bool = True,
        space_attn_num_heads: int = None,
        input_d_model: int = None,
        use_inverted_attention: bool = False,
        encoder_attn_num_heads: int = None,
        use_ffn: bool = False,
        mamba_version: str = "mamba2",
        attn_impl: str = "flash_attention_2",
        mamba_d_state: int = 128,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_headdim: int = 64,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_slots = num_slots
        self.num_blocks = num_blocks
        self.d_model = d_model
        self.use_cross_attn = use_cross_attn
        self.space_attn_num_heads = space_attn_num_heads or d_model // 64
        self.input_d_model = input_d_model
        self.use_inverted_attention = use_inverted_attention
        self.encoder_attn_num_heads = encoder_attn_num_heads or d_model // 64
        self.use_ffn = use_ffn
        self.mamba_version = mamba_version
        self.attn_impl = attn_impl
        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand
        self.mamba_headdim = mamba_headdim

@dataclass
class SlotSSMOutput(ModelOutput):
    slots: Optional[Any] = None
    attentions: Optional[Any] = None

class SlotSSM(PreTrainedModel):
    config_class = SlotSSMConfig
    
    def __init__(self, config: SlotSSMConfig):
        super().__init__(config)
        
        self.blocks = nn.ModuleList([
            SlotSSMBlock(
                d_model=config.d_model,
                space_attn_num_heads=config.space_attn_num_heads,
                input_d_model=config.input_d_model,
                use_inverted_attention=config.use_inverted_attention,
                encoder_attn_num_heads=config.encoder_attn_num_heads,
                use_ffn=config.use_ffn,
                use_cross_attn=config.use_cross_attn,
                mamba_version=config.mamba_version,
                attn_impl=config.attn_impl,
                mamba_d_state=config.mamba_d_state,
                mamba_d_conv=config.mamba_d_conv,
                mamba_expand=config.mamba_expand,
                mamba_headdim=config.mamba_headdim,
                layer_idx=idx
            )
            for idx in range(config.num_blocks)
        ])
        self.init_slots = nn.Parameter(torch.randn(1, 1, config.num_slots, config.d_model))

    def forward(self, ref=None, cache_params=None, output_attentions=False):
        """
        ref: B, T, L, D
        """
        B, T, L, D = ref.shape
        slots = self.init_slots.repeat(B, T, 1, 1)
        
        attentions = []
        for block in self.blocks:
            slots = block(slots, ref, cache_params=cache_params, output_attentions=output_attentions)
            if isinstance(slots, tuple):
                slots, attn = slots
                attentions.append(attn)
        
        if cache_params is not None:
            cache_params.seqlen_offset += T
            
        return SlotSSMOutput(slots=slots, attentions=attentions)

class SlotSSMBlock(nn.Module):
    def __init__(self, d_model, space_attn_num_heads, input_d_model=None, use_ffn=False,
                 encoder_attn_num_heads=None, use_cross_attn=True, use_inverted_attention=False,
                 mamba_version='mamba2', layer_idx=None, attn_impl="flash_attention_2",
                 mamba_d_state=128, mamba_d_conv=4, mamba_expand=2, mamba_headdim=64):
        super().__init__()
        assert mamba_version in ['mamba1', 'mamba2'], "Mamba version must be mamba1 or mamba2"
        assert attn_impl in ["flash_attention_2", "eager"], \
            "Attention implementation must be flash_attention_2 or eager"
        
        if use_cross_attn:
            self.cross_attn_input_norm = nn.LayerNorm(d_model)
            self.cross_attn_ref_norm = nn.LayerNorm(d_model)
            self.input_proj = nn.Linear(input_d_model, d_model) if input_d_model is not None else d_model
            if use_inverted_attention:
                if attn_impl == "flash_attention_2":
                    mprint("Inverted attention with flash attention 2 is not supported, using eager instead")
                self.cross_attn = MultiHeadAttention(
                    d_model=d_model,
                    num_heads=encoder_attn_num_heads,
                    inverted=True
                )
            else:
                if attn_impl == "flash_attention_2":
                    self.cross_attn = FlashMHA(
                        embed_dim=d_model,
                        num_heads=encoder_attn_num_heads,
                        cross_attn=True
                    )
                else:
                    self.cross_attn = MultiHeadAttention(
                        d_model=d_model,
                        num_heads=encoder_attn_num_heads,
                        inverted=False,
                        output_attentions=True
                    )

        self.time_mixer_norm = nn.LayerNorm(d_model)
        if mamba_version == 'mamba2':
            # need to check if d_model * expand / headdim = multiple of 8
            assert (d_model * mamba_expand / mamba_headdim) % 8 == 0, "d_model * expand must be a multiple of headdim"
            self.time_mixer = Mamba2(
                d_model=d_model,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                expand=mamba_expand,
                headdim=mamba_headdim,
                layer_idx=layer_idx
            )
        else:
            self.time_mixer = Mamba(
                d_model=d_model,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                expand=mamba_expand,
                layer_idx=layer_idx
            )
        
        self.space_attn_norm = nn.LayerNorm(d_model)
        if attn_impl == "flash_attention_2":
            self.space_attn = FlashMHA(
                embed_dim=d_model,
                num_heads=space_attn_num_heads
            )
        else:
            self.space_attn = MultiHeadAttention(
                d_model=d_model,
                num_heads=space_attn_num_heads
            )
        
        if use_ffn:
            self.ffn_norm = nn.LayerNorm(d_model)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model)
            )
            
    def forward(self, input, ref=None, cache_params=None, output_attentions=False):
        """
        input: B, T, N, D (slots)
        ref: B, T, L, D (optional reference input for cross attention)
        """
        B, T, N, D = input.shape
        # Cross attention if enabled and ref is provided
        output_attn = None
        if hasattr(self, 'cross_attn_input_norm'):
            assert ref is not None, "Reference input is required for cross attention"

            ref_reshape = rearrange(ref, 'b t n d -> (b t) n d')
            input_reshape = rearrange(input, 'b t n d -> (b t) n d')

            ref_proj = self.input_proj(ref_reshape)
            ref_proj = self.cross_attn_ref_norm(ref_proj)
            x = self.cross_attn_input_norm(input_reshape)

            if isinstance(self.cross_attn, FlashMHA):
                cross_attn_out = self.cross_attn(x=x, x_kv=ref_proj)
                input = input + rearrange(cross_attn_out, '(b t) n d -> b t n d', b=B, t=T)
            else:
                cross_attn_out = self.cross_attn(
                    x, ref_proj, ref_proj, output_attentions=output_attentions
                )
                if isinstance(cross_attn_out, tuple) and len(cross_attn_out) > 1:
                    output_attn = cross_attn_out[1]
                input = input + rearrange(cross_attn_out[0], '(b t) n d -> b t n d', b=B, t=T)
            
        else:
            assert ref is None, "Reference input is not supported without cross attention"

        # Time mixing (Mamba)
        input_reshape = rearrange(input, 'b t n d -> (b n) t d')
        x = self.time_mixer_norm(input_reshape)
        time_mixed = self.time_mixer(x, inference_params=cache_params)
        input = input + rearrange(time_mixed, '(b n) t d -> b t n d', b=B, n=N)

        # Space attention
        input_reshape = rearrange(input, 'b t n d -> (b t) n d')
        x = self.space_attn_norm(input_reshape)
        if isinstance(self.space_attn, FlashMHA):
            space_attn_out = self.space_attn(x)
        else:
            space_attn_out = self.space_attn(x, x, x)
        input = input + rearrange(space_attn_out, '(b t) n d -> b t n d', b=B, t=T)

        if hasattr(self, 'ffn_norm'):
            input_norm = self.ffn_norm(input)
            input = input + self.ffn(input_norm)

        return input if not output_attentions else (input, output_attn)
    
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    num_slots = 32
    d_model = 256
    ref = torch.randn(2, 10, 448, 1024).to(device)

    # cache_params = MambaCache()
    cache_params = None

    slotssm = SlotSSM(
        num_slots=num_slots, num_blocks=4, d_model=d_model, 
        space_attn_num_heads=d_model // 64, input_d_model=ref.shape[-1], 
        use_inverted_attention=False, encoder_attn_num_heads=d_model // 64
    )

    slotssm = slotssm.to(device)

    # Use automatic mixed precision
    with torch.amp.autocast(device_type=device, dtype=dtype):
        output = slotssm(ref, cache_params=cache_params)

    ref = torch.randn(2, 10, 448, 1024).to(device)

    cache_params = MambaCache()

    with torch.amp.autocast(device_type=device, dtype=dtype):
        output = slotssm(ref, cache_params=cache_params)
    
    ref_next = torch.randn(2, 1, 448, 1024).to(device)
    with torch.amp.autocast(device_type=device, dtype=dtype):
        for k, v in cache_params.key_value_memory_dict.items():
            cache_params.key_value_memory_dict[k] = (v[0].to(dtype), v[1].to(dtype))
        output_next = slotssm(ref_next, cache_params=cache_params)
    
    print(output_next.shape)