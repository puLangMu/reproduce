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




class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        # num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        # iou_head_depth: int = 3,
        # iou_head_hidden_dim: int = 256,
        image_embedding_size: Tuple[int, int] = (64, 64),
        embed_dim: int = 256,
        
        num_heads: int = 12,
        # mlp_ratio: float = 4.0,
        # out_chans: int = 256,  # changed from 2 to 256
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.BatchNorm2d,
        act_layer: Type[nn.Module] = nn.GELU,
        # use_abs_pos: bool = False,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        # window_size: int = 0,
        # global_attn_indexes: Tuple[int, ...] = (),
        decrease_scale: int = 2,
        
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
        self.embed_dim = embed_dim
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.pe_layer = PositionEmbeddingRandom(self.embed_dim // 2)

        # self.num_multimask_outputs = num_multimask_outputs

        # self.iou_token = nn.Embedding(1, transformer_dim)
        # self.num_mask_tokens = num_multimask_outputs + 1
        # self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )

        # self.output_hypernetworks_mlps = nn.ModuleList(
        #     [
        #         MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
        #         for i in range(self.num_mask_tokens)
        #     ]
        # )

        self.image_embedding_size = image_embedding_size


        self.block1 =  UpBlock(
                dim= embed_dim //8  ,
                num_heads=4,
                # mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
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

        # self.iou_prediction_head = MLP(
        #     transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        # )

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
        masks = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=self.get_dense_pe(),
            sparse_prompt_embeddings=source_embeddings,
            # dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select the correct mask or masks for output
        # if multimask_output:
        #     mask_slice = slice(1, None)
        # else:
        #     mask_slice = slice(0, 1)
        # masks = masks[:, mask_slice, :, :]
        # iou_pred = iou_pred[:, mask_slice]

        masks = self.block1(masks)
        masks = self.upsample(masks)

       
        masks = torch.sigmoid(40 * (masks - 0.5))
        masks = torch.where(masks > 0.5, torch.tensor(1.0, device=masks.device), torch.tensor(0.0, device=masks.device))
        # Prepare output
        return masks

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        # dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        # output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        # output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = sparse_prompt_embeddings

        # Expand per-image data in batch direction to be per-mask
        src = image_embeddings
        # src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, image_embeddings.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        # iou_token_out = hs[:, 0, :]
        # mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        # hyper_in_list: List[torch.Tensor] = []
        # for i in range(self.num_mask_tokens):
        #     hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        # hyper_in = torch.stack(hyper_in_list, dim=1)
        # b, c, h, w = upscaled_embedding.shape
        # masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # # Generate mask quality predictions
        # iou_pred = self.iou_prediction_head(iou_token_out)

        return upscaled_embedding
    
    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)


# # Lightly adapted from
# # https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
# class MLP(nn.Module):
#     def __init__(
#         self,
#         input_dim: int,
#         hidden_dim: int,
#         output_dim: int,
#         num_layers: int,
#         sigmoid_output: bool = False,
#     ) -> None:
#         super().__init__()
#         self.num_layers = num_layers
#         h = [hidden_dim] * (num_layers - 1)
#         self.layers = nn.ModuleList(
#             nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
#         )
#         self.sigmoid_output = sigmoid_output

#     def forward(self, x):
#         for i, layer in enumerate(self.layers):
#             x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
#         if self.sigmoid_output:
#             x = F.sigmoid(x)
#         return x

class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C
    

class UpBlock(nn.Module):
    """a block for source encoder"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        # mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.BatchNorm2d,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
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
        self.norm2 = norm_layer(int(dim / decrease_scale))

        self.act = act_layer()
        self.input_size = input_size

        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size ,
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
        # shortcut = x
        # x = self.norm1(x)
        # # Window partition
        # if self.window_size > 0:
        #     H, W = x.shape[1], x.shape[2]
        #     x, pad_hw = window_partition(x, self.window_size)

        # x = self.attn(x)
        # # Reverse window partition
        # if self.window_size > 0:
        #     x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        # x = shortcut + x
        # x = x + self.mlp(self.norm2(x))

        # x shape is (B, C, H, W)

        # residual block 
        shortcut = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x) 
        x = self.norm1(x) 
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
        x = self.norm1(x) 

        x = self.upsample(x)
        x = self.norm2(x)
        x = self.act(x)

        return x
    
class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
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

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

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