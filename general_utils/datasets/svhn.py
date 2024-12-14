from typing import Optional

import torch
from torchvision import transforms
from torchvision.datasets import SVHN

SVHN_TRAIN_TRANSFORM = transforms.Compose([
    transforms.Pad(4, padding_mode='edge'),
    transforms.RandomAffine(5, scale=(0.9, 1.1), shear=5, fill=0),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
])


def svhn(root: str,
         train: bool,
         transform: Optional[torch.nn.Module] = None,
         target_transform: Optional[torch.nn.Module] = None,
         download: Optional[bool] = True,
         use_default_transform: Optional[bool] = True
) -> SVHN:
    """"""

    if transform is None and use_default_transform:
        transform = SVHN_TRAIN_TRANSFORM if train else transforms.ToTensor()

    split = "train" if train else "test"

    return SVHN(root, split, transform, target_transform, download)
