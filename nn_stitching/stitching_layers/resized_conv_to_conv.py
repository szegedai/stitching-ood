from typing import Optional

import torch
from torch import nn

from nn_stitching.stitching_layers.stitching_layer import StitchingLayer
from nn_stitching.models.stitched_model import StitchedModel


class ResizedConvToConvStitchingLayer(StitchingLayer):
    """"""

    def __init__(self,
                 resize_type: Optional[str] = "upsample",
                 pre_resize: Optional[bool] = True,
                 use_bn: Optional[bool] = False,
                 act_fn: nn.Module = nn.Identity(),
                 upsample_mode: str = "bilinear",
                 *args, **kwargs
    ) -> None:
        """"""
        super().__init__(*args, **kwargs)
        self.resize_type = resize_type
        self.pre_resize = pre_resize
        self.use_bn = use_bn
        self.act_fn = act_fn
        self.norm = None
        self.transform = None
        self.resize = None
        self.upsample_mode = upsample_mode

    def setup_transformation_layer(self,
                                   front_model: StitchedModel,
                                   end_model: StitchedModel
    ) -> None:
        """"""
        source_shape = front_model.get_stitching_source_shape()
        target_shape = end_model.get_stitching_target_shape()
        source_channels = source_shape[0]
        target_channels = target_shape[0]

        source_h = source_shape[1]
        source_w = source_shape[2]

        target_h = target_shape[1]
        target_w = target_shape[2]

        if self.resize_type == "upsample":
            self.resize = nn.Upsample(size=(target_h, target_w),
                                      mode=self.upsample_mode)

        elif self.resize_type == "conv":
            scale_factor = int(target_h / source_h)

            if scale_factor == 1:
                self.resize = nn.Identity()
            else:
                self.resize = nn.ConvTranspose2d(in_channels=source_channels,
                                                 out_channels=source_channels,
                                                 kernel_size=scale_factor,
                                                 stride=scale_factor)


        self.transform = nn.Conv2d(in_channels=source_channels,
                                   out_channels=target_channels,
                                   kernel_size=1,
                                   stride=1)

        if self.use_bn:
            self.norm = nn.BatchNorm2d(num_features=source_channels)
        else:
            self.norm = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.pre_resize:
            x = self.resize(x)
            x = self.norm(x)
            x = self.transform(x)
            x = self.act_fn(x)
            return x
        else:
            x = self.norm(x)
            x = self.transform(x)
            x = self.act_fn(x)
            x = self.resize(x)
            return x
