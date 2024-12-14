from typing import Optional

import torch
from torch import nn

from nn_stitching.stitching_layers.stitching_layer import StitchingLayer
from nn_stitching.models.stitched_model import StitchedModel


class TransToTransStitchingLayer(StitchingLayer):
    """"""

    def __init__(self,
                 norm: Optional[str] = None,
                 act_fn: nn.Module = nn.Identity(),
                 init: Optional[str] = "rand",
                 *args, **kwargs
    ) -> None:
        """"""

        super().__init__(*args, **kwargs)

        assert norm in [None, "", "bn", "ln"], f"Norm {norm} not supported!"

        self.norm = norm
        self.act_fn = act_fn
        self.norm = norm
        self.transform = None
        self.init = init

    def setup_transformation_layer(self,
                                   front_model: StitchedModel,
                                   end_model: StitchedModel
    ) -> None:
        source_shape = front_model.get_stitching_source_shape()
        target_shape = end_model.get_stitching_target_shape()
        source_channels = source_shape[0]
        source_dim = source_shape[-1]
        target_dim = target_shape[-1]

        self.transform = nn.Linear(in_features=source_dim,
                                   out_features=target_dim)

        # Initialization
        if self.init == "eye":
            self._init_identity()

        if self.norm == "bn":
            self.norm = nn.BatchNorm2d(num_features=source_channels)
        elif self.norm == "ln":
            self.norm = nn.LayerNorm(source_channels)
        else:
            self.norm = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.transform(x)
        x = self.act_fn(x)
        return x

    def _init_identity(self) -> None:
        weight_shape = self.transform.weight.shape
        init_weight = torch.eye(*weight_shape)
        self.transform.weight = nn.Parameter(init_weight)

        bias_shape = self.transform.bias.shape
        init_bias = torch.zeros(bias_shape)
        self.transform.bias = nn.Parameter(init_bias)
