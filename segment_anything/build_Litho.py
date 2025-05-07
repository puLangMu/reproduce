import torch

from functools import partial

from .modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer, SourceEncoder, CrossAttention
from .modeling.LMLitho import LMLitho  # 修改导入路径


import torch
from torch import Tensor, nn

import math

from typing import Optional, Tuple, Type




def build_litho():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformer_dim = 256

    image_encoder = ImageEncoderViT(
        img_size=1024,  # 输入图像大小
        patch_size=16,  # Patch 大小
        in_chans=1,     # 输入通道数
        embed_dim=768,  # 嵌入维度
        depth=12,       # 模型深度
        num_heads=12,   # 注意力头数
        mlp_ratio=4.0,  # MLP 隐藏层与嵌入维度的比例
        out_chans=256,  # 输出通道数
    ).to(device)

    source_encoder = SourceEncoder(
        img_size=256,  # 输入图像大小
        in_chans=1,    # 输入通道数
        embed_dim=64,  # 嵌入维度
        depth=4,       # 模型深度
        num_heads=8,   # 注意力头数
    ).to(device)

    transformer = TwoWayTransformer(
        depth=2,
        embedding_dim=transformer_dim,
        mlp_dim=2048,
        num_heads=8,
    ).to(device)

    mask_decoder = MaskDecoder(
        transformer_dim=transformer_dim,
        transformer=transformer,
        image_embedding_size=(64, 64),
        embed_dim=256,
    ).to(device)

    model = LMLitho(
        image_encoder = image_encoder,
        source_encoder = source_encoder,
        mask_decoder = mask_decoder,
    ).to(device)

    return model


def build_light_litho():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformer_dim = 256

    image_encoder = ImageEncoderViT(
        img_size=256,  # 输入图像大小
        patch_size=16,  # Patch 大小
        in_chans=1,     # 输入通道数
        embed_dim=192,  # 嵌入维度
        depth=3,       # 模型深度
        num_heads=3,   # 注意力头数
        mlp_ratio=4.0,  # MLP 隐藏层与嵌入维度的比例
        out_chans=256,  # 输出通道数
    ).to(device)

    source_encoder = SourceEncoder(
        img_size=64,  # 输入图像大小
        in_chans=1,    # 输入通道数
        embed_dim=64,  # 嵌入维度
        depth=4,       # 模型深度
        num_heads=4,   # 注意力头数
    ).to(device)

    transformer = TwoWayTransformer(
        depth=2,
        embedding_dim=transformer_dim,
        mlp_dim=1024,
        num_heads=4,
    ).to(device)

    mask_decoder = MaskDecoder(
        transformer_dim=transformer_dim,
        transformer=transformer,
        image_embedding_size=(16, 16),
        embed_dim=256,
    ).to(device)

    model = LMLitho(
        image_encoder = image_encoder,
        source_encoder = source_encoder,
        mask_decoder = mask_decoder,
    ).to(device)

    return model


def build_source_litho():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformer_dim = 256

    image_encoder = ImageEncoderViT(
        img_size=1024,  # 输入图像大小
        patch_size=16,  # Patch 大小
        in_chans=1,     # 输入通道数
        embed_dim=768,  # 嵌入维度
        depth=12,       # 模型深度
        num_heads=12,   # 注意力头数
        mlp_ratio=4.0,  # MLP 隐藏层与嵌入维度的比例
        out_chans=256,  # 输出通道数
    ).to(device)

    source_encoder = SourceEncoder(
        img_size=256,  # 输入图像大小
        in_chans=1,    # 输入通道数
        embed_dim=64,  # 嵌入维度
        depth=4,       # 模型深度
        num_heads=8,   # 注意力头数
    ).to(device)

    transformer = CrossAttention(
        dim = 256
    ).to(device)

    mask_decoder = MaskDecoder(
        transformer_dim=transformer_dim,
        transformer=transformer,
        image_embedding_size=(64, 64),
        embed_dim=256,
    ).to(device)

    model = LMLitho(
        image_encoder = image_encoder,
        source_encoder = source_encoder,
        mask_decoder = mask_decoder,
    ).to(device)

    return model

def build_litho_one():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformer_dim = 256

    image_encoder = ImageEncoderViT(
        img_size=1024,  # 输入图像大小
        patch_size=16,  # Patch 大小
        in_chans=1,     # 输入通道数
        embed_dim=768,  # 嵌入维度
        depth=12,       # 模型深度
        num_heads=12,   # 注意力头数
        mlp_ratio=4.0,  # MLP 隐藏层与嵌入维度的比例
        out_chans=256,  # 输出通道数
    ).to(device)

    source_encoder = SourceEncoder(
        img_size=256,  # 输入图像大小
        in_chans=1,    # 输入通道数
        embed_dim=64,  # 嵌入维度
        depth=4,       # 模型深度
        num_heads=8,   # 注意力头数
    ).to(device)

    transformer = TwoWayTransformer(
        depth=1,
        embedding_dim=transformer_dim,
        mlp_dim=2048,
        num_heads=8,
    ).to(device)

    mask_decoder = MaskDecoder(
        transformer_dim=transformer_dim,
        transformer=transformer,
        image_embedding_size=(64, 64),
        embed_dim=256,
    ).to(device)

    model = LMLitho(
        image_encoder = image_encoder,
        source_encoder = source_encoder,
        mask_decoder = mask_decoder,
    ).to(device)

    return model