import torch
from segment_anything.modeling.image_encoder import ImageEncoderViT
from segment_anything.modeling.source_encoder import SourceEncoder
from segment_anything.modeling.mask_decoder import MaskDecoder


from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple


class LMLitho(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "binary"  # here i change it to binary 

    def __init__(self,
        image_encoder: ImageEncoderViT,
        source_encoder: SourceEncoder,
        mask_decoder: MaskDecoder,
        # pixel_mean: List[float] = [123.675, 116.28, 103.53],
        # pixel_std: List[float] = [58.395, 57.12, 57.375],

        # pixel_mean: List[float] = [123],
        # pixel_std: List[float] = [58.395], # i do not estimate the data 
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.source_encoder = source_encoder
        self.mask_decoder = mask_decoder
        # self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        # self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    # @property
    # def device(self) -> Any:
    #     return self.pixel_mean.device

    def forward(
        self,
        image_input: torch.Tensor,
        source_input: torch.Tensor,
       
    ) -> torch.Tensor:
        """
        Forward pass of the model.

        Arguments:
          image_input (torch.Tensor): The input image tensor.
          source_input (torch.Tensor): The input source tensor.
          multimask_output (bool): Whether to output multiple masks.

        Returns:
          torch.Tensor: The predicted masks.
        """
        # 图像编码
        image_embeddings = self.image_encoder(image_input)
        # 源编码
        source_embeddings = self.source_encoder(source_input)
        
        # mask_decoder
        masks= self.mask_decoder(
            image_embeddings=image_embeddings,
            source_embeddings=source_embeddings,
        )
        
        return masks
       