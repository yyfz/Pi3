"""
Conv head is from MoGe (https://github.com/microsoft/moge)
"""

from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalized_view_plane_uv(width: int, height: int, aspect_ratio: float = None, dtype: torch.dtype = None, device: torch.device = None) -> torch.Tensor:
    "UV with left-top corner as (-width / diagonal, -height / diagonal) and right-bottom corner as (width / diagonal, height / diagonal)"
    if aspect_ratio is None:
        aspect_ratio = width / height
    
    span_x = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5
    span_y = 1 / (1 + aspect_ratio ** 2) ** 0.5

    u = torch.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, width, dtype=dtype, device=device)
    v = torch.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height, height, dtype=dtype, device=device)
    u, v = torch.meshgrid(u, v, indexing='xy')
    uv = torch.stack([u, v], dim=-1)
    return uv

class ResidualConvBlock(nn.Module):  
    def __init__(self, in_channels: int, out_channels: int = None, hidden_channels: int = None, padding_mode: str = 'replicate', activation: Literal['relu', 'leaky_relu', 'silu', 'elu'] = 'relu', norm: Literal['group_norm', 'layer_norm'] = 'group_norm'):  
        super(ResidualConvBlock, self).__init__()  
        if out_channels is None:  
            out_channels = in_channels
        if hidden_channels is None:
            hidden_channels = in_channels

        if activation =='relu':
            activation_cls = lambda: nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            activation_cls = lambda: nn.LeakyReLU(negative_slope=0.2, inplace=True)
        elif activation =='silu':
            activation_cls = lambda: nn.SiLU(inplace=True)
        elif activation == 'elu':
            activation_cls = lambda: nn.ELU(inplace=True)
        else:
            raise ValueError(f'Unsupported activation function: {activation}')

        self.layers = nn.Sequential(
            nn.GroupNorm(1, in_channels),
            activation_cls(),
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, padding_mode=padding_mode),
            nn.GroupNorm(hidden_channels // 32 if norm == 'group_norm' else 1, hidden_channels),
            activation_cls(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1, padding_mode=padding_mode)
        )
        
        self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0) if in_channels != out_channels else nn.Identity()  
  
    def forward(self, x):  
        skip = self.skip_connection(x)  
        x = self.layers(x)
        x = x + skip
        return x  


class ConvHead(nn.Module):
    def __init__(
        self, 
        num_features: int,
        dim_in: int, 
        dim_out: List[int], 
        dim_proj: int = 512,
        dim_upsample: List[int] = [256, 128, 128],
        dim_times_res_block_hidden: int = 1,
        num_res_blocks: int = 1,
        res_block_norm: Literal['group_norm', 'layer_norm'] = 'group_norm',
        last_res_blocks: int = 0,
        last_conv_channels: int = 32,
        last_conv_size: int = 1,
        projects: nn.Module = None,
        using_uv: bool = True
    ):
        super().__init__()
        
        self.using_uv = using_uv
        if projects is not None:
            self.projects = projects

        self.upsample_blocks = nn.ModuleList([
            nn.Sequential(
                self._make_upsampler(in_ch + 2 if using_uv else in_ch, out_ch),
                *(ResidualConvBlock(out_ch, out_ch, dim_times_res_block_hidden * out_ch, activation="relu", norm=res_block_norm) for _ in range(num_res_blocks))
            ) for in_ch, out_ch in zip([dim_proj] + dim_upsample[:-1], dim_upsample)
        ])

        self.output_block = nn.ModuleList([
            self._make_output_block(
                dim_upsample[-1] + 2 if using_uv else dim_upsample[-1], dim_out_, dim_times_res_block_hidden, last_res_blocks, last_conv_channels, last_conv_size, res_block_norm,
            ) for dim_out_ in dim_out
        ])
    
    def _make_upsampler(self, in_channels: int, out_channels: int):
        upsampler = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        )
        upsampler[0].weight.data[:] = upsampler[0].weight.data[:, :, :1, :1]
        return upsampler

    def _make_output_block(self, dim_in: int, dim_out: int, dim_times_res_block_hidden: int, last_res_blocks: int, last_conv_channels: int, last_conv_size: int, res_block_norm: Literal['group_norm', 'layer_norm']):
        return nn.Sequential(
            nn.Conv2d(dim_in, last_conv_channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            *(ResidualConvBlock(last_conv_channels, last_conv_channels, dim_times_res_block_hidden * last_conv_channels, activation='relu', norm=res_block_norm) for _ in range(last_res_blocks)),
            nn.ReLU(inplace=True),
            nn.Conv2d(last_conv_channels, dim_out, kernel_size=last_conv_size, stride=1, padding=last_conv_size // 2, padding_mode='replicate'),
        )
            
    def forward(self, hidden_states: torch.Tensor, image: torch.Tensor = None, patch_h: int = None, patch_w: int = None):
        if image is not None:
            img_h, img_w = image.shape[-2:]
            patch_h, patch_w = img_h // 14, img_w // 14
        else:
            assert patch_h is not None and patch_w is not None
            img_h = patch_h * 14
            img_w = patch_w * 14

        # Process the hidden states
        if self.projects is not None:
            x = self.projects(hidden_states).permute(0, 2, 1).unflatten(2, (patch_h, patch_w)).contiguous()
        else:
            x = hidden_states
        
        # Upsample stage
        # (patch_h, patch_w) -> (patch_h * 2, patch_w * 2) -> (patch_h * 4, patch_w * 4) -> (patch_h * 8, patch_w * 8)
        for i, block in enumerate(self.upsample_blocks):
            # UV coordinates is for awareness of image aspect ratio
            if self.using_uv:
                uv = normalized_view_plane_uv(width=x.shape[-1], height=x.shape[-2], aspect_ratio=img_w / img_h, dtype=x.dtype, device=x.device)
                uv = uv.permute(2, 0, 1).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
                x = torch.cat([x, uv], dim=1)
            for layer in block:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
        
        # (patch_h * 8, patch_w * 8) -> (img_h, img_w)
        x = F.interpolate(x, (img_h, img_w), mode="bilinear", align_corners=False)
        if self.using_uv:
            uv = normalized_view_plane_uv(width=x.shape[-1], height=x.shape[-2], aspect_ratio=img_w / img_h, dtype=x.dtype, device=x.device)
            uv = uv.permute(2, 0, 1).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
            x = torch.cat([x, uv], dim=1)

        if isinstance(self.output_block, nn.ModuleList):
            output = [torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False) for block in self.output_block]
        else:
            output = torch.utils.checkpoint.checkpoint(self.output_block, x, use_reentrant=False)
        
        return output