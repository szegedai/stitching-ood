from typing import Optional

import torch
from torch import nn

from nn_stitching.stitching_layers.stitching_layer import StitchingLayer
from nn_stitching.models.stitched_model import StitchedModel


class ConvToTransStitchingLayer(StitchingLayer):
    """"""

    def __init__(self,
                 norm: Optional[str] = None,
                 act_fn: nn.Module = nn.Identity(),
                 init: Optional[str] = "rand",
                 cls_token: Optional[str] = None,
                 *args, **kwargs
    ) -> None:
        """"""

        super().__init__(*args, **kwargs)

        assert norm in [None, "", "bn", "ln"], f"Norm {norm} not supported!"
        assert cls_token in [None, "pool", "learn"], f"CLS token handling {cls_token} not supported!"

        self.norm = norm
        self.act_fn = act_fn
        self.norm = norm
        self.transform = None
        self.init = init
        self.cls_token_handling = cls_token
        self.cls_token = None

    def setup_transformation_layer(self,
                                   front_model: StitchedModel,
                                   end_model: StitchedModel
    ) -> None:
        source_shape = front_model.get_stitching_source_shape() # C x H x W
        target_shape = end_model.get_stitching_target_shape() # N x D
        source_channels = source_shape[0]
        target_embed_dim = target_shape[1]

        self.transform = nn.Conv2d(in_channels=source_channels,
                                   out_channels=target_embed_dim,
                                   kernel_size=1,
                                   stride=1)

        # # Initialization
        # if self.init == "eye":
        #     self._init_identity()

        if self.norm == "bn":
            self.norm = nn.BatchNorm2d(num_features=source_channels)
        elif self.norm == "ln":
            self.norm = nn.LayerNorm(source_channels)
        else:
            self.norm = nn.Identity()

        if self.cls_token_handling == "learn":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, target_embed_dim))
            nn.init.normal_(self.cls_token, std=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)

        x = self.transform(x)
        x = self.act_fn(x)

        B, C, _, _ = x.shape
        x = x.reshape(B, C, -1).permute(0, 2, 1)

        if self.cls_token_handling == "learn":
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        elif self.cls_token_handling == "pool":
            prepend_token = x.mean(dim=1)
            prepend_token = prepend_token.reshape(prepend_token.shape[0], 1, prepend_token.shape[1])
            x = torch.cat((prepend_token.expand(x.shape[0], -1, -1), x), dim=1)

        return x
