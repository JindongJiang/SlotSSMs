import torch
from torch import nn
import numpy as np
from einops import rearrange, repeat, reduce
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers.modeling_utils import PreTrainedModel, ModelOutput
from transformers.configuration_utils import PretrainedConfig
from dataclasses import dataclass

from transformers.models.vit_mae.modeling_vit_mae import get_2d_sincos_pos_embed_from_grid


def get_2d_sincos_pos_embed(embed_dim, grid_size, add_cls_token=False, num_cls_tokens=1):
    """
    Create 2D sin/cos positional embeddings.

    Args:
        embed_dim (`int`):
            Embedding dimension.
        grid_size (`int`):
            The grid height and width.
        add_cls_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add a classification (CLS) token.

    Returns:
        (`torch.FloatTensor` of shape (grid_size*grid_size, embed_dim) or (1+grid_size*grid_size, embed_dim): the
        position embeddings (with or without classification token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if add_cls_token:
        pos_embed = np.concatenate(
            [np.zeros([num_cls_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

# Configuration classes
class VitEncoderConfig(PretrainedConfig):
    model_type = "vit_encoder"
    
    def __init__(
        self,
        hidden_size=192,
        output_size=None,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=None,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        image_size=224,
        patch_size=16,
        num_channels=3,
        initializer_range=0.02,
        image_norm_mode="zero_one",
        num_cls_tokens=1,
        learn_initial_cls_tokens=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        self.hidden_size = hidden_size
        self.output_size = output_size if output_size is not None else hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.image_norm_mode = image_norm_mode
        self.num_cls_tokens = num_cls_tokens
        self.num_patches = (image_size // patch_size) ** 2
        self.learn_initial_cls_tokens = learn_initial_cls_tokens

# Model Output classes
@dataclass
class VitEncoderOutput(ModelOutput):
    cls_tokens: Optional[Any] = None
    last_hidden_state: Optional[Any] = None
    attentions: Optional[Any] = None

# Model classes
class VitEncoder(PreTrainedModel):
    config_class = VitEncoderConfig
    
    def __init__(self, config: VitEncoderConfig):
        super().__init__(config)
        
        # Patch embedding
        self.patch_embedding = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )
        
        # Position embeddings
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, config.num_cls_tokens + config.num_patches, config.hidden_size)
        )
        
        # CLS tokens
        if config.num_cls_tokens > 0:
            self.cls_tokens = nn.Parameter(
                torch.zeros(1, config.num_cls_tokens, config.hidden_size)
            )
            if not config.learn_initial_cls_tokens:
                self.cls_tokens_log_sigma = nn.Parameter(
                    torch.zeros(1, config.num_cls_tokens, config.hidden_size)
                )
        
        # Transformer encoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size if config.intermediate_size else 4*config.hidden_size,
                dropout=config.hidden_dropout_prob,
                activation=config.hidden_act,
                batch_first=True
            ),
            num_layers=config.num_hidden_layers
        )
        
        # Output projection
        self.output_proj = nn.Linear(config.hidden_size, config.output_size)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        # Initialize position embeddings
        pos_embed = get_2d_sincos_pos_embed(
            self.position_embeddings.shape[-1],
            int(self.config.num_patches**0.5),
            add_cls_token=self.config.num_cls_tokens > 0,
            num_cls_tokens=self.config.num_cls_tokens
        )
        self.position_embeddings.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
    
    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        
        # Patch embedding
        x = self.patch_embedding(pixel_values)
        x = rearrange(x, 'b c h w -> b (h w) c')
        
        # Add position embeddings
        if self.config.num_cls_tokens > 0:
            cls_tokens = self.cls_tokens.expand(batch_size, -1, -1)
            if not self.config.learn_initial_cls_tokens:
                cls_tokens = cls_tokens + torch.randn_like(cls_tokens) * torch.exp(self.cls_tokens_log_sigma)
            x = torch.cat([cls_tokens, x], dim=1)
        
        x = x + self.position_embeddings
        
        # Transformer encoder
        encoder_outputs = self.encoder(x)
        
        # Project outputs
        hidden_states = self.output_proj(encoder_outputs)
        
        # Split cls tokens and patch embeddings
        if self.config.num_cls_tokens > 0:
            cls_tokens = hidden_states[:, :self.config.num_cls_tokens]
            last_hidden_state = hidden_states[:, self.config.num_cls_tokens:]
        else:
            cls_tokens = None
            last_hidden_state = hidden_states
            
        return VitEncoderOutput(
            cls_tokens=cls_tokens,
            last_hidden_state=last_hidden_state,
            attentions=None  # Could add attention outputs if needed
        )

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        # Using GroupNorm with groups=in_channels (one group per channel)
        self.norm1 = nn.GroupNorm(num_groups=1, num_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.norm2 = nn.GroupNorm(num_groups=1, num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU(inplace=True)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = x
        
        out = self.norm1(x)
        out = self.conv1(out)
        out = self.relu(out)
        
        out = self.norm2(out)
        out = self.conv2(out)
        
        out += self.shortcut(residual)
        out = self.relu(out)
        
        return out
    

class CNNEncoderConfig(PretrainedConfig):
    model_type = "cnn_encoder"
    
    def __init__(
        self,
        input_channels=3,
        input_resolution=128,
        kernel_size=5,
        hidden_size=192,
        out_channels=192,
        image_norm_mode="zero_one",
        downsampling_ratio=2,
        num_hidden_layers=3,
        blocks_per_layer=2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_channels = input_channels
        self.input_resolution = input_resolution
        self.kernel_size = kernel_size
        self.hidden_size = hidden_size
        self.out_channels = out_channels
        self.image_norm_mode = image_norm_mode
        self.downsampling_ratio = downsampling_ratio
        self.num_hidden_layers = num_hidden_layers
        self.blocks_per_layer = blocks_per_layer

@dataclass
class CNNEncoderOutput(ModelOutput):
    features: Optional[Any] = None

class CNNEncoder(PreTrainedModel):
    config_class = CNNEncoderConfig
    
    def __init__(self, config: CNNEncoderConfig):
        super().__init__(config)
        
        # Initial downsampling convolution
        self.conv1 = nn.Conv2d(config.input_channels, config.hidden_size, 
                              kernel_size=config.downsampling_ratio, 
                              stride=config.downsampling_ratio, 
                              padding=0)
        self.norm1 = nn.GroupNorm(num_groups=1, num_channels=config.hidden_size)
        self.gelu = nn.GELU()
        
        # Create dynamic number of layers
        self.layers = nn.ModuleList()
        # in_channels = config.hidden_size
        
        for i in range(config.num_hidden_layers):
            self.layers.append(
                self._make_layer(config.hidden_size, config.hidden_size, config.blocks_per_layer, config.kernel_size)
            )
        
        # Final convolution to match output channels
        self.final_norm = nn.GroupNorm(num_groups=1, num_channels=config.hidden_size)
        self.final_conv = nn.Conv2d(
            config.hidden_size, 
            config.out_channels,
            kernel_size=1,
            stride=1
        )
        
    def _make_layer(self, in_channels, out_channels, num_blocks, kernel_size):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, kernel_size))
        for _ in range(num_blocks - 1):
            layers.append(ResidualBlock(out_channels, out_channels, kernel_size))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Initial downsampling
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.gelu(x)
        
        # Dynamic number of layers
        for layer in self.layers:
            x = layer(x)
        
        # Final projection
        x = self.final_norm(x)
        features = self.final_conv(x)
        return CNNEncoderOutput(features=features)

# for testing
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # dtype = torch.float16

    input_channels = 3
    input_resolution = 128
    input_tensor = torch.randn(2, input_channels, input_resolution, input_resolution, device=device)
    vit_encoder = VitEncoder(
        VitEncoderConfig(
            hidden_size=192,
            num_attention_heads=6,
            num_hidden_layers=6,
            image_size=input_resolution,
            patch_size=8,
            num_channels=input_channels
        )
    ).to(device)
    vit_encoder_output = vit_encoder(input_tensor)

    cnn_encoder = CNNEncoder(
        CNNEncoderConfig(
            hidden_size=192,
            out_channels=192,
            input_resolution=input_resolution,
            num_hidden_layers=3,
            blocks_per_layer=2,
            input_channels=input_channels,
            downsampling_ratio=8
        )
    ).to(device)
    cnn_encoder_output = cnn_encoder(input_tensor)

    print(vit_encoder_output.last_hidden_state.shape)
    print(cnn_encoder_output.features.shape)