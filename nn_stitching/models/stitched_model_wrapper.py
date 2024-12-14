from typing import Tuple, Optional
from copy import deepcopy

import torch
from torch import nn

from nn_stitching.models.stitched_model import StitchedModel


class StitchedModelWrapper(StitchedModel):
    r"""Wrapper for custom models to be used in stitching."""

    def __init__(self,
                 model: nn.Module,
                 input_shape: Tuple[int, ...],
                 stitch_from: Optional[str] = None,
                 stitch_to: Optional[str] = None,
                 *args, **kwargs
    ) -> None:
        """"""
        super().__init__(input_shape, stitch_from, stitch_to, *args, **kwargs)
        self.model = deepcopy(model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _get_layer(self, layer_name: str) -> nn.Module:
        """"""
        for name, layer in self.model.named_modules():
            if name == layer_name:
                return layer

        raise Exception(f"Layer {layer_name} not in model!")

    def get_model(self) -> nn.Module:
        return self.model
