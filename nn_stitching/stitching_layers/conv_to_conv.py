from typing import Optional

import torch
from torch import nn

from nn_stitching.stitching_layers.stitching_layer import StitchingLayer
from nn_stitching.models.stitched_model import StitchedModel


class ConvToConvStitchingLayer(StitchingLayer):
    """"""

    def __init__(self,
                 use_bn: Optional[bool] = False,
                 act_fn: nn.Module = nn.Identity(),
                 init: Optional[str] = "rand",
                 *args, **kwargs
    ) -> None:
        """"""
        super().__init__(*args, **kwargs)
        self.use_bn = use_bn
        self.act_fn = act_fn
        self.norm = None
        self.transform = None
        self.init = init

    def setup_transformation_layer(self,
                                   front_model: StitchedModel,
                                   end_model: StitchedModel
    ) -> None:
        """"""
        source_shape = front_model.get_stitching_source_shape()
        target_shape = end_model.get_stitching_target_shape()
        source_channels = source_shape[0]
        target_channels = target_shape[0]

        self.transform = nn.Conv2d(in_channels=source_channels,
                                   out_channels=target_channels,
                                   kernel_size=1,
                                   stride=1)

        # Initialization
        if self.init == "eye":
            self._init_identity()

        if self.use_bn:
            self.norm = nn.BatchNorm2d(num_features=source_channels)
        else:
            self.norm = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.transform(x)
        x = self.act_fn(x)
        return x

    def _init_identity(self) -> None:
        weight_shape = self.transform.weight.shape
        in_chans = self.transform.in_channels
        out_chans = self.transform.out_channels

        if in_chans != out_chans:
            raise ValueError("Identity initialization requires in_chans and "
                             f"out_chans to be the same. Instead, got {in_chans} "
                             f"and {out_chans} respectively.")

        init_weight = torch.eye(in_chans).reshape(weight_shape)
        self.transform.weight = nn.Parameter(init_weight)

        bias_shape = self.transform.bias.shape
        init_bias = torch.zeros(bias_shape)
        self.transform.bias = nn.Parameter(init_bias)
