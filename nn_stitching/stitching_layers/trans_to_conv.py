import math
from typing import Optional

import torch
from torch import nn

from nn_stitching.stitching_layers.stitching_layer import StitchingLayer
from nn_stitching.models.stitched_model import StitchedModel


class TransToConvStitchingLayer(StitchingLayer):
    """"""

    def __init__(self,
                 norm: Optional[str] = None,
                 act_fn: nn.Module = nn.Identity(),
                 init: Optional[str] = "rand",
                 cls_token: Optional[str] = None,
                 upsample_mode: str = "bilinear",
                 *args, **kwargs
    ) -> None:
        """"""

        super().__init__(*args, **kwargs)

        assert norm in [None, "", "bn", "ln"], f"Norm {norm} not supported!"
        assert cls_token in [None, "drop", "add", "only"], f"CLS token handling {cls_token} not supported!"

        self.norm = norm
        self.act_fn = act_fn
        self.norm = norm
        self.transform = None
        self.init = init
        self.cls_token_handling = cls_token if cls_token else "drop"
        self.front_cls_token = False
        self.resize = None
        self.upsample_mode = upsample_mode

        if self.cls_token_handling == "add":
            # self.cls_token_weight = torch.autograd.Variable(torch.rand(1), requires_grad=True) # Variable([0])
            self.cls_token_weight = nn.Parameter(torch.Tensor([.1]))

    def setup_transformation_layer(self,
                                   front_model: StitchedModel,
                                   end_model: StitchedModel
    ) -> None:
        source_shape = front_model.get_stitching_source_shape() # N x D
        target_shape = end_model.get_stitching_target_shape() # C x H x W

        n_tokens = source_shape[0]

        # Only works with square spatial tokenization
        self.front_cls_token = int(math.sqrt(n_tokens)) != math.sqrt(n_tokens)

        source_channels = source_shape[1]
        target_channels = target_shape[0]
        target_h = target_shape[1]
        target_w = target_shape[2]

        self.resize = nn.Upsample(size=(target_h, target_w),
                                  mode=self.upsample_mode)

        self.transform = nn.Conv2d(in_channels=source_channels,
                                   out_channels=target_channels,
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

    def forward(self,
                x: torch.Tensor,
                x2: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.norm(x)

        if self.front_cls_token:
            if self.cls_token_handling == "drop":
                x = x[:, 1:, :]
            elif self.cls_token_handling == "add":
                # cls_tokens = x[:, 0, :]
                cls_tokens = torch.unsqueeze(x[:, 0, :], dim=1)
                cls_tokens = cls_tokens * self.cls_token_weight.expand_as(cls_tokens)
                x = x[:, 1:, :]
                x = torch.add(x, cls_tokens)
            elif self.cls_token_handling == "only":
                n_tokens = x.shape[1] - 1
                cls_tokens = torch.unsqueeze(x[:, 0, :], dim=1)
                x = cls_tokens.repeat(1, n_tokens, 1)

        B, N, _ = x.shape
        x = x.reshape(B, int(math.sqrt(N)), int(math.sqrt(N)), -1) \
             .permute(0, 3, 1, 2)

        x = self.resize(x)
        x = self.transform(x)
        x = self.act_fn(x)

        return x
