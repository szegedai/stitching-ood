from typing import Type, Union, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from general_utils.models.resnet.resnet import Add
from general_utils.models.resnet.preact_resnet import CIFAR10_MEAN, CIFAR10_STD, Swish


class _Block(nn.Module):
    """WideResNet Block."""

    def __init__(self,
                 in_planes,
                 out_planes,
                 stride,
                 activation_fn: Type[nn.Module] = nn.ReLU):
        super().__init__()
        self.batchnorm_0 = nn.BatchNorm2d(in_planes)
        self.relu_0 = activation_fn()
        # We manually pad to obtain the same effect as `SAME` (necessary when
        # `stride` is different than 1).
        self.conv_0 = nn.Conv2d(in_planes,
                                out_planes,
                                kernel_size=3,
                                stride=stride,
                                padding=0,
                                bias=False)
        self.batchnorm_1 = nn.BatchNorm2d(out_planes)
        self.relu_1 = activation_fn()
        self.conv_1 = nn.Conv2d(out_planes,
                                out_planes,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False)
        self.has_shortcut = in_planes != out_planes
        if self.has_shortcut:
            self.shortcut = nn.Conv2d(in_planes,
                                      out_planes,
                                      kernel_size=1,
                                      stride=stride,
                                      padding=0,
                                      bias=False)
        else:
            self.shortcut = None
        self._stride = stride
        self.add = Add()

    def forward(self, x):
        if self.has_shortcut:
            x = self.relu_0(self.batchnorm_0(x))
        else:
            out = self.relu_0(self.batchnorm_0(x))
        v = x if self.has_shortcut else out
        if self._stride == 1:
            v = F.pad(v, (1, 1, 1, 1))
        elif self._stride == 2:
            v = F.pad(v, (0, 1, 0, 1))
        else:
            raise ValueError('Unsupported `stride`.')
        out = self.conv_0(v)
        out = self.relu_1(self.batchnorm_1(out))
        out = self.conv_1(out)
        # out = torch.add(self.shortcut(x) if self.has_shortcut else x, out)
        out = self.add(self.shortcut(x) if self.has_shortcut else x, out)
        return out


class _BlockGroup(nn.Module):
    """WideResNet block group."""

    def __init__(self,
                 num_blocks,
                 in_planes,
                 out_planes,
                 stride,
                 activation_fn: Type[nn.Module] = nn.ReLU):
        super().__init__()
        block = []
        for i in range(num_blocks):
            block.append(
                _Block(i == 0 and in_planes or out_planes,
                       out_planes,
                       i == 0 and stride or 1,
                       activation_fn=activation_fn))
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class WideResNet(nn.Module):
    """WideResNet."""

    def __init__(self,
                 n_classes: int = 10,
                 n_channels: int = 3,
                 depth: int = 28,
                 width: int = 10,
                 activation_fn: Type[nn.Module] = Swish,
                 mean: Union[Tuple[float, ...], float] = CIFAR10_MEAN,
                 std: Union[Tuple[float, ...], float] = CIFAR10_STD,
                 padding: int = 0,
                 **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.depth = depth
        self.width = width

        # persistent=False to not put these tensors in the module's state_dict and not try to
        # load it from the checkpoint
        self.register_buffer('mean', torch.tensor(mean).view(n_channels, 1, 1),
                             persistent=False)
        self.register_buffer('std', torch.tensor(std).view(n_channels, 1, 1),
                             persistent=False)
        self.padding = padding
        num_channels = [16, 16 * width, 32 * width, 64 * width]
        assert (depth - 4) % 6 == 0
        num_blocks = (depth - 4) // 6
        self.init_conv = nn.Conv2d(n_channels,
                                   num_channels[0],
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   bias=False)
        self.layer = nn.Sequential(
            _BlockGroup(num_blocks,
                        num_channels[0],
                        num_channels[1],
                        1,
                        activation_fn=activation_fn),
            _BlockGroup(num_blocks,
                        num_channels[1],
                        num_channels[2],
                        2,
                        activation_fn=activation_fn),
            _BlockGroup(num_blocks,
                        num_channels[2],
                        num_channels[3],
                        2,
                        activation_fn=activation_fn))
        self.batchnorm = nn.BatchNorm2d(num_channels[3])
        self.relu = activation_fn()
        self.logits = nn.Linear(num_channels[3], n_classes)
        self.num_channels = num_channels[3]

    def forward(self, x):
        if self.padding > 0:
            x = F.pad(x, (self.padding,) * 4)
        out = (x - self.mean) / self.std
        out = self.init_conv(out)
        out = self.layer(out)
        out = self.relu(self.batchnorm(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.num_channels)
        return self.logits(out)


def wide_resnet_28_10(n_classes, n_channels, **kwargs) -> WideResNet:
    return WideResNet(n_classes, n_channels, 28, 10, **kwargs)


def wide_resnet_34_10(n_classes, n_channels, **kwargs) -> WideResNet:
    return WideResNet(n_classes, n_channels, 34, 10, **kwargs)
