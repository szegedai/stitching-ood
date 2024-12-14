from abc import ABC
from typing import Tuple, Optional, Union

import torch
from torch import nn


class StitchedModel(nn.Module, ABC):
    r"""Base class for all models used in stitching.

    Args:
        input_shape (tuple): Shape of the model's input without batch size.
        stitch_from (string, optional): Name of the layer that provides its
            output as the input to the stitching layer. Leave unspecified for
            "end models".
        stitch_to (string, optional): Name of the layer which has its output
            overridden by the stitching layer's output. Leave unspecified for
            "front models".
    """

    def __init__(self,
                 input_shape: Tuple[int, ...],
                 stitch_from: Optional[str] = None,
                 stitch_to: Optional[str] = None,
                 *args, **kwargs
    ) -> None:
        """"""

        super().__init__(*args, **kwargs)
        self.input_shape = input_shape
        self.stitch_from = stitch_from
        self.stitch_to = stitch_to
        self.stitch_from_layer = None
        self.stitch_to_layer = None

    def get_stitching_target_shape(self) -> Tuple[int, ...]:
        r"""Get the desired size of the stitching layer's output without batch size."""

        self._setup_stitch_layers()

        output_shape = None

        def output_shape_hook(module, input, output):
            """Returns the output tensor's shape *without* batch size."""

            nonlocal output_shape
            if isinstance(output, tuple):
                output_shape = output[0].shape[1:]
            else:
                output_shape = output.shape[1:]

        if self.stitch_to is not None:
            shape_hook = self.stitch_to_layer.register_forward_hook(output_shape_hook)
            fake_input = torch.rand(size=(2, *self.input_shape))
            self.forward(fake_input)
            shape_hook.remove()

        return output_shape

    def get_stitching_source_shape(self) -> Tuple[int, ...]:
        r"""Get the size of the stitching layer's input without batch size."""

        self._setup_stitch_layers()

        output_shape = None

        def output_shape_hook(module, input, output):
            """Returns the output tensor's shape *without* batch size."""

            nonlocal output_shape
            if isinstance(output, tuple):
                output_shape = output[0].shape[1:]
            else:
                output_shape = output.shape[1:]

        if self.stitch_from is not None:
            shape_hook = self.stitch_from_layer.register_forward_hook(output_shape_hook)
            fake_input = torch.rand(size=(2, *self.input_shape))
            self.forward(fake_input)
            shape_hook.remove()

        return output_shape

    def freeze(self) -> None:
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def _get_layer(self, layer_name: str) -> nn.Module:
        for name, layer in self.named_modules():
            if name == layer_name:
                return layer

        raise Exception(f"Layer {layer_name} not in model!")

    def _setup_stitch_layers(self) -> None:
        if self.stitch_from_layer is None and self.stitch_from is not None:
            self.stitch_from_layer = self._get_layer(self.stitch_from)
        if self.stitch_to_layer is None and self.stitch_to is not None:
            self.stitch_to_layer = self._get_layer(self.stitch_to)

    def get_model(self) -> nn.Module:
        return self

    def model_equals(self, obj: nn.Module) -> bool:
        """"""

        if isinstance(obj, StitchedModel):
            return obj.get_model() == self.get_model() # works for wrapper too

        return obj == self.get_model()
