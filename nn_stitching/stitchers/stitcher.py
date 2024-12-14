from typing import Optional

import torch
from torch import nn

from nn_stitching.models.stitched_model import StitchedModel
from nn_stitching.stitching_layers.stitching_layer import StitchingLayer
from nn_stitching.utils.low_rank import low_rank_approx


class Stitcher(nn.Module):
    r"""Base neural network stitcher."""

    def __init__(self,
                 front_model: StitchedModel,
                 end_model: StitchedModel,
                 stitching_layer: StitchingLayer,
                 modify_only_stitching_layer: Optional[bool] = False,
                 front_rank: Optional[int] = None,
                 *args, **kwargs
    ) -> None:
        """"""

        super().__init__(*args, **kwargs)
        self.front_model = front_model
        self.end_model = end_model
        self.stitching_layer = stitching_layer
        self.activation = None
        self.modify_only_stitching_layer = modify_only_stitching_layer
        self.front_model_activation = None
        self.end_model_activation = None
        self.front_rank = front_rank
        self._setup_models()
        self._setup_connection()

    def forward(self, x: torch.Tensor, return_activations: bool = False) -> torch.Tensor:
        self.front_model(x)
        # self.activation = self.stitching_layer(self.activation)
        if return_activations:
            return self.end_model(x), {"front": self.front_model_activation,
                                       "end": self.end_model_activation,
                                       "stitched": self.activation}
        else:
            return self.end_model(x)

    def _setup_models(self) -> None:
        r"""Freeze front and end models and setup stitching transformation"""

        self.front_model.freeze()
        self.end_model.freeze()
        self.stitching_layer.setup_transformation_layer(self.front_model, self.end_model)

    def _setup_connection(self) -> None:
        r"""Register stitching connection between the models."""

        def _save_output_hook(module, input, output):
            self.activation = output
            self.front_model_activation = output

        def _override_output_hook(module, input, output):
            # output = self.activation
            # print("override called")
            if self.front_rank:
                self.activation = low_rank_approx(self.activation, self.front_rank)

            self.activation = self.stitching_layer(self.activation)
            self.end_model_activation = output
            # print("transform complete")
            return self.activation

        self.front_model.stitch_from_layer.register_forward_hook(_save_output_hook)
        self.end_model.stitch_to_layer.register_forward_hook(_override_output_hook)

    def eval(self) -> "Stitcher":
        if self.modify_only_stitching_layer:
            self.stitching_layer.eval()
            return self
        else:
            return super().eval()

    def train(self, mode: bool = True) -> "Stitcher":
        if self.modify_only_stitching_layer:
            self.stitching_layer.train(mode)
            return self
        else:
            return super().train(mode)
