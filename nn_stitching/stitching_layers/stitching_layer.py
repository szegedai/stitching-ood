from abc import ABC, abstractmethod

import torch
from torch import nn

from nn_stitching.models.stitched_model import StitchedModel


class StitchingLayer(nn.Module, ABC):
    r"""Base class for all stitching layers."""

    def __init__(self, *args, **kwargs) -> None:
        """"""
        super().__init__(*args, **kwargs)

    @abstractmethod
    def setup_transformation_layer(self,
                                   front_model: StitchedModel,
                                   end_model: StitchedModel
    ) -> None:
        raise NotImplementedError("Method setup_transformation_layer not implemented for type StitchingLayer")

    def set_transform_match(self, x1, x2) -> None:
        from nn_stitching.init.least_squares import ps_inv
        pinv = ps_inv(x1, x2)
        weight_shape = self.transform.weight.shape
        bias_shape = self.transform.bias.shape
        weight = nn.Parameter(pinv["w"].to(x1.device).reshape(weight_shape))
        bias = nn.Parameter(pinv["b"].to(x1.device).reshape(bias_shape))

        self.transform.weight = weight
        self.transform.bias = bias
