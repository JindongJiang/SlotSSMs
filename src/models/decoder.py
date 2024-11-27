import os
import torch
from torch import nn
from dataclasses import dataclass
from typing import Optional, Any

from transformers.modeling_utils import PreTrainedModel, ModelOutput
from transformers.configuration_utils import PretrainedConfig
from einops import repeat, rearrange, reduce

from .encoder import get_2d_sincos_pos_embed

class CartesianPositionalEmbedding(nn.Module):
    def __init__(self, channels, image_size):
        super().__init__()
        self.channels = channels
        pos_x = torch.linspace(-1, 1, image_size)
        pos_y = torch.linspace(-1, 1, image_size)
        grid_y, grid_x = torch.meshgrid(pos_y, pos_x, indexing='ij')
        self.register_buffer('grid_x', grid_x.unsqueeze(0))
        self.register_buffer('grid_y', grid_y.unsqueeze(0))
        
    def forward(self, x):
        b, c, h, w = x.shape
        pos_x = repeat(self.grid_x, '1 h w -> b c h w', b=b, c=c//2)
        pos_y = repeat(self.grid_y, '1 h w -> b c h w', b=b, c=c//2)
        return x + torch.cat([pos_x, pos_y], dim=1)

class SBDecoderConfig(PretrainedConfig):
    model_type = "spatial_broadcast_decoder"
    def __init__(self,
                 in_channels=192,
                 out_channels=3,
                 initial_size=8,
                 output_resolution=128,
                 hidden_size=192,
                 **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.initial_size = initial_size
        self.output_resolution = output_resolution
        self.hidden_size = hidden_size

@dataclass
class SBDecoderOutput(ModelOutput):
    recon: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None

class SBDecoder(PreTrainedModel):
    config_class = SBDecoderConfig
    
    def __init__(self, config: SBDecoderConfig):
        super().__init__(config)
        
        assert (config.output_resolution // config.initial_size) in [8, 16], "Currently only support upsample factor of 8 or 16"
        up_sample_factor = config.output_resolution // config.initial_size
        
        self.norm = nn.LayerNorm(config.in_channels)
        self.pos = CartesianPositionalEmbedding(channels=config.in_channels, image_size=config.initial_size)
        self.cnn_transpose = nn.Sequential(
            ConvTranspose2dBlock(config.in_channels, config.hidden_size, kernel_size=5, stride=2, padding=2, output_padding=1, bias=True),
            ConvTranspose2dBlock(config.hidden_size, config.hidden_size, kernel_size=5, stride=2, padding=2, output_padding=1, bias=True),
            ConvTranspose2dBlock(config.hidden_size, config.hidden_size, kernel_size=5, stride=2, padding=2, output_padding=1, bias=True),
            (nn.ConvTranspose2d(config.hidden_size, config.out_channels + 1, kernel_size=5, stride=2, padding=2, output_padding=1, bias=True) 
             if up_sample_factor == 16 else nn.Conv2d(config.hidden_size, config.out_channels + 1, kernel_size=5, stride=1, padding=2, bias=True))
        )

    def forward(self, slots: torch.Tensor):
        B, N, D = slots.shape
        slots = self.norm(slots)
        slots_broadcast = repeat(slots, 'b n d -> (b n) d h w', h=self.config.initial_size, w=self.config.initial_size)
        slots_broadcast = self.pos(slots_broadcast)
        output = self.cnn_transpose(slots_broadcast)
        output = rearrange(output, '(b n) c h w -> b n c h w', n=N)
        
        mask = output[:, :, -1:]
        mask = mask.softmax(dim=1)
        color = output[:, :, :-1]
        recon = reduce(color * mask, 'b n c h w -> b c h w', reduction='sum')

        return SBDecoderOutput(
            recon=recon,
            mask=mask
        )

class ConvTranspose2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=bias)
        self.norm = nn.GroupNorm(1, out_channels)
        self.activation = nn.GELU()
    
    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))

class VitDecoderConfig(PretrainedConfig):
    model_type = "vit_decoder"
    def __init__(self,
                 input_size=None,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=None,
                 hidden_act="relu",
                 hidden_dropout_prob=0.0,
                 image_size=224,
                 patch_size=16,
                 num_channels=3,
                 num_cls_tokens=1,
                 **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size if input_size is not None else hidden_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.hidden_act = hidden_act
        self.num_cls_tokens = num_cls_tokens
        self.num_patches = (image_size // patch_size) ** 2

@dataclass
class VitDecoderOutput(ModelOutput):
    recon: Optional[Any] = None
    attentions: Optional[Any] = None

class VitDecoder(PreTrainedModel):
    config_class = VitDecoderConfig
    
    def __init__(self, config: VitDecoderConfig):
        super().__init__(config)
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, config.num_cls_tokens + config.num_patches, config.hidden_size),
            requires_grad=False
        )
        self.input_proj = nn.Linear(config.input_size, config.hidden_size)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size or 4 * config.hidden_size,
            dropout=config.hidden_dropout_prob,
            activation=config.hidden_act,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_hidden_layers
        )
        
        self.pred = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, (config.patch_size ** 2) * config.num_channels, bias=True),
        )
        
        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(
            self.position_embeddings.shape[-1],
            int(self.config.num_patches**0.5),
            add_cls_token=self.config.num_cls_tokens > 0,
            num_cls_tokens=self.config.num_cls_tokens
        )
        self.position_embeddings.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def forward(self, slots):
        B, N, D = slots.shape
        mask_tokens = self.mask_token.repeat(B, self.config.num_patches, 1)
        
        slots = self.input_proj(slots)
        hidden_states = torch.cat([slots, mask_tokens], dim=1)
        hidden_states = hidden_states + self.position_embeddings
        
        last_hidden_state = self.encoder(hidden_states)
        last_hidden_state = last_hidden_state[:, self.config.num_cls_tokens:]
        
        logits = rearrange(
            self.pred(last_hidden_state),
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=int(self.config.num_patches**0.5),
            w=int(self.config.num_patches**0.5),
            p1=self.config.patch_size,
            p2=self.config.patch_size,
            c=self.config.num_channels
        )
        
        return VitDecoderOutput(
            recon=logits,
            attentions=None
        )

# for testing
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test parameters
    batch_size = 2
    num_slots = 4
    input_dim = 192

    # Test configurations
    vit_config = VitDecoderConfig(
        input_size=192,
        hidden_size=384,
        num_hidden_layers=6,
        num_attention_heads=6,
        image_size=128,
        patch_size=8,
        num_channels=3,
        num_cls_tokens=num_slots
    )
    
    sb_config = SBDecoderConfig(
        in_channels=192,
        out_channels=3,
        initial_size=8,
        output_resolution=128,
        hidden_size=384
    )
    
    
    print("Testing VitDecoder...")
    # Initialize VitDecoder model
    vit_model = VitDecoder(vit_config).to(device)
    
    # Create dummy input
    vit_slots = torch.randn(batch_size, num_slots, input_dim, device=device)
    
    # Forward pass
    vit_outputs = vit_model(vit_slots)
    
    # Print output shapes
    print("Input shape:", vit_slots.shape)
    print("Output shape:", vit_outputs.recon.shape)
    print("Expected output shape:", 
          f"[{batch_size}, {vit_config.num_channels}, {vit_config.image_size}, {vit_config.image_size}]")
    
    # Verify output dimensions
    vit_expected_shape = (batch_size, vit_config.num_channels, vit_config.image_size, vit_config.image_size)
    assert vit_outputs.recon.shape == vit_expected_shape, \
        f"VitDecoder output shape {vit_outputs.recon.shape} doesn't match expected shape {vit_expected_shape}"
    
    print("VitDecoder tests passed!")
    print("\nTesting SBDecoder...")
    
    # Initialize SBDecoder model
    sb_model = SBDecoder(sb_config).to(device)
    
    # Create dummy input
    sb_slots = torch.randn(batch_size, num_slots, input_dim, device=device)
    
    # Forward pass
    sb_outputs = sb_model(sb_slots)
    
    # Print output shapes
    print("Input shape:", sb_slots.shape)
    print("Output shape:", sb_outputs.recon.shape)
    print("Mask shape:", sb_outputs.mask.shape)
    print("Expected output shape:", 
          f"[{batch_size}, {sb_config.out_channels}, {sb_config.output_resolution}, {sb_config.output_resolution}]")
    print("Expected mask shape:", 
          f"[{batch_size}, {num_slots}, 1, {sb_config.output_resolution}, {sb_config.output_resolution}]")
    
    # Verify output dimensions
    sb_expected_shape = (batch_size, sb_config.out_channels, sb_config.output_resolution, sb_config.output_resolution)
    sb_mask_expected_shape = (batch_size, num_slots, 1, sb_config.output_resolution, sb_config.output_resolution)
    
    assert sb_outputs.recon.shape == sb_expected_shape, \
        f"SBDecoder output shape {sb_outputs.recon.shape} doesn't match expected shape {sb_expected_shape}"
    assert sb_outputs.mask.shape == sb_mask_expected_shape, \
        f"SBDecoder mask shape {sb_outputs.mask.shape} doesn't match expected shape {sb_mask_expected_shape}"
    
    # Verify mask sums to 1 across slots
    mask_sum = sb_outputs.mask.sum(dim=1)
    assert torch.allclose(mask_sum, torch.ones_like(mask_sum), atol=1e-6), \
        "SBDecoder mask does not sum to 1 across slots"
    
    print("SBDecoder tests passed!")
    print("\nAll tests passed successfully!")

