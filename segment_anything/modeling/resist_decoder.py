# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F


from .common import LayerNorm2d


from typing import Any, Optional, Tuple, Type, List
import numpy as np




class ResistDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        # transformer: nn.Module,
        activation: Type[nn.Module] = nn.GELU,

        image_embedding_size: Tuple[int, int] = (64,64),
        embed_dim: int = 256,
        
        num_heads: int = 8,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.BatchNorm2d,
        act_layer: Type[nn.Module] = nn.GELU,

        # decrease_scale: int = 2,
        
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.source_pe = nn.Parameter(
                torch.zeros(1, transformer_dim, int(image_embedding_size[0]), int( image_embedding_size[1]))
            )
        self.image_pe = nn.Parameter(
                torch.zeros(1, transformer_dim, int(image_embedding_size[0]), int( image_embedding_size[1]))
            )
        # self.transformer = transformer

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.cross_attention = CrossAttention(
            dim = self.transformer_dim,
            num_heads = self.num_heads
        )




        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )


        self.image_embedding_size = image_embedding_size


        self.block1 =  UpBlock(
                dim= embed_dim //8  ,
                num_heads=4,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                window_size= 16,
                input_size=(image_embedding_size[0] * 4, image_embedding_size[1] * 4),
                decrease_scale= 4,
            )
        
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(embed_dim // 32, embed_dim // 64, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 64),
            activation(),
            nn.Conv2d(embed_dim // 64, embed_dim // 256, kernel_size=3, stride=1, padding=1),
            activation()
        )

  
    def forward(
        self,
        image_embeddings: torch.Tensor,
        # image_pe: torch.Tensor,
        source_embeddings: torch.Tensor,
        # dense_prompt_embeddings: torch.Tensor,
        # multimask_output: bool,
    ) -> Tuple[torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        # image_embeddings = image_embeddings + self.image_pe
        image_embeddings = image_embeddings
        source_embeddings = source_embeddings 

        image_embeddings = image_embeddings.permute(0, 2, 3, 1) # change to (B, H, W, C)
        source_embeddings = source_embeddings.permute(0, 2, 3, 1)

        

        # cross attention
        resist = self.cross_attention(image = image_embeddings, source = source_embeddings)
        # resist = image_embeddings

        resist = resist.permute(0, 3, 1, 2) # change back to (B, C, H, W)
        resist = self.output_upscaling(resist)
        resist = self.block1(resist)

        resist = self.upsample(resist)


        resist = torch.sigmoid(100 * (resist))
        return resist


class UpBlock(nn.Module):
    """a block for source encoder"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.BatchNorm2d,
        act_layer: Type[nn.Module] = nn.GELU,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
        decrease_scale: int = 2,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
            increase_scale (int): Scale factor for increasing the number of channels.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)


        self.norm4 = norm_layer(int(dim / decrease_scale))

        self.act = act_layer()
        self.input_size = input_size

        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            # input_size=input_size ,
        )

        self.conv1 = nn.Conv2d(
            dim,
            dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim,
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            dim,
            dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.upsample = nn.ConvTranspose2d(dim, dim // decrease_scale, kernel_size=2, stride=2)

        

        self.pos_embed = nn.Parameter(
                torch.zeros(1, dim, int(input_size[0]), int( input_size[1]))
            )
        
        self.window_size = window_size
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:


        # residual block 
        shortcut = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x) 
        x = self.norm2(x) 
        x = x + shortcut
        x = self.act(x)

        # multiattention
        x = x + self.pos_embed

        x = x.permute(0, 2, 3, 1) # change to (B, H, W, C)
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        
        x = x.permute(0, 3, 1, 2) # change back 
        x = self.norm3(x) 

        x = self.upsample(x)
        x = self.norm4(x)
        x = self.act(x)

        return x
    
class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x

class CrossAttention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q =  nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)


    def forward(self, image: torch.Tensor, source:torch.Tensor) -> torch.Tensor:
        B, H, W, _ = image.shape
        # qkv with shape (3, B, nHead, H * W, C)
        kv = self.kv(image).reshape(B, H * W, 2, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q = self.q(source).reshape(B, H * W, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        k, v = kv.reshape(2, B * self.num_heads, H * W, -1).unbind(0)
        q = q.reshape(B * self.num_heads, H * W, -1)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        # if self.use_rel_pos:
        #     attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x
    



def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x