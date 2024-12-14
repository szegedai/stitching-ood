from typing import Optional

import torch
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST


MNIST_TRAIN_TRANSFORM = transforms.Compose([
    transforms.Pad(4, padding_mode='edge'),
    transforms.RandomAffine(5, scale=(0.9, 1.1), shear=5, fill=0),
    transforms.CenterCrop(28),
    transforms.ToTensor(),
])

FASHION_TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(28,4),
    transforms.ToTensor(),
])


def mnist(root: str,
          train: bool,
          transform: Optional[torch.nn.Module] = None,
          target_transform: Optional[torch.nn.Module] = None,
          download: Optional[bool] = True,
          use_default_transform: Optional[bool] = True
) -> MNIST:
    """"""

    if transform is None and use_default_transform:
        transform = MNIST_TRAIN_TRANSFORM if train else transforms.ToTensor()

    return MNIST(root, train, transform, target_transform, download)


def fashion_mnist(root: str,
                  train: bool,
                  transform: Optional[torch.nn.Module] = None,
                  target_transform: Optional[torch.nn.Module] = None,
                  download: Optional[bool] = True,
                  use_default_transform: Optional[bool] = True
) -> FashionMNIST:
    """"""

    if transform is None and use_default_transform:
        transform = FASHION_TRAIN_TRANSFORM if train else transforms.ToTensor()

    return FashionMNIST(root, train, transform, target_transform, download)
