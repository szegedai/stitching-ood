from typing import Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from general_utils.models.resnet.resnet import Add


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


class _Swish(torch.autograd.Function):
    """Custom implementation of swish."""

    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish(nn.Module):
    """Module using custom implementation."""

    def forward(self, input_tensor):
        return _Swish.apply(input_tensor)


class _PreActBlock(nn.Module):
    """Pre-activation ResNet Block."""

    def __init__(self, in_planes, out_planes, stride, activation_fn=nn.ReLU):
        super().__init__()
        self._stride = stride
        self.batchnorm_0 = nn.BatchNorm2d(in_planes)
        self.relu_0 = activation_fn()
        # We manually pad to obtain the same effect as `SAME` (necessary when
        # `stride` is different than 1).
        self.conv_2d_1 = nn.Conv2d(in_planes, out_planes, kernel_size=3,
                                   stride=stride, padding=0, bias=False)
        self.batchnorm_1 = nn.BatchNorm2d(out_planes)
        self.relu_1 = activation_fn()
        self.conv_2d_2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                                   padding=1, bias=False)
        self.has_shortcut = stride != 1 or in_planes != out_planes
        if self.has_shortcut:
            self.shortcut = nn.Conv2d(in_planes, out_planes, kernel_size=3,
                                      stride=stride, padding=0, bias=False)
        self.add = Add()

    def _pad(self, x):
        if self._stride == 1:
            x = F.pad(x, (1, 1, 1, 1))
        elif self._stride == 2:
            x = F.pad(x, (0, 1, 0, 1))
        else:
            raise ValueError('Unsupported `stride`.')
        return x

    def forward(self, x):
        out = self.relu_0(self.batchnorm_0(x))
        shortcut = self.shortcut(self._pad(x)) if self.has_shortcut else x
        out = self.conv_2d_1(self._pad(out))
        out = self.conv_2d_2(self.relu_1(self.batchnorm_1(out)))
        out = self.add(out, shortcut)
        return out


class PreActResNet(nn.Module):
    """Pre-activation ResNet."""

    def __init__(self,
                 n_classes: int = 10,
                 n_channels: int = 3,
                 depth: int = 18,
                 activation_fn: nn.Module = Swish,
                 mean: Union[Tuple[float, ...], float] = CIFAR10_MEAN,
                 std: Union[Tuple[float, ...], float] = CIFAR10_STD,
                 padding: int = 0,
                 **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.depth = depth

        # persistent=False to not put these tensors in the module's state_dict and not try to
        # load it from the checkpoint
        self.register_buffer('mean', torch.tensor(mean).view(n_channels, 1, 1),
                             persistent=False)
        self.register_buffer('std', torch.tensor(std).view(n_channels, 1, 1),
                             persistent=False)
        self.mean_cuda = None
        self.std_cuda = None
        self.padding = padding
        self.conv_2d = nn.Conv2d(n_channels, 64, kernel_size=3, stride=1,
                                 padding=1, bias=False)
        if depth == 18:
            num_blocks = (2, 2, 2, 2)
        elif depth == 34:
            num_blocks = (3, 4, 6, 3)
        else:
            raise ValueError('Unsupported `depth`.')
        self.layer_0 = self._make_layer(64, 64, num_blocks[0], 1, activation_fn)
        self.layer_1 = self._make_layer(64, 128, num_blocks[1], 2, activation_fn)
        self.layer_2 = self._make_layer(128, 256, num_blocks[2], 2, activation_fn)
        self.layer_3 = self._make_layer(256, 512, num_blocks[3], 2, activation_fn)
        self.batchnorm = nn.BatchNorm2d(512)
        self.relu = activation_fn()
        self.logits = nn.Linear(512, n_classes)

    def _make_layer(self, in_planes, out_planes, num_blocks, stride,
                    activation_fn):
        layers = []
        for i, stride in enumerate([stride] + [1] * (num_blocks - 1)):
            layers.append(
                _PreActBlock(i == 0 and in_planes or out_planes,
                             out_planes,
                             stride,
                             activation_fn))
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.padding > 0:
            x = F.pad(x, (self.padding,) * 4)
        out = (x - self.mean) / self.std
        out = self.conv_2d(out)
        out = self.layer_0(out)
        out = self.layer_1(out)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.relu(self.batchnorm(out))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return self.logits(out)


def preact_resnet18(n_classes: int, n_channels: int, **kwargs) -> PreActResNet:
    return PreActResNet(n_classes, n_channels, 18, **kwargs)


def preact_resnet34(n_classes: int, n_channels: int, **kwargs) -> PreActResNet:
    return PreActResNet(n_classes, n_channels, 34, **kwargs)
