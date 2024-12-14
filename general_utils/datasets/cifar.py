from os import path
from typing import Callable, Optional, List, Any, Tuple

import torch
import numpy as np
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, VisionDataset


CIFAR_TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, 4),
    transforms.ToTensor(),
])


LARGE_CIFAR_TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomRotation(15,),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])


CIFAR5M_TRAIN_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    # TODO: add more augmentations?
    transforms.ToTensor(),
])


def cifar10(root: str,
            train: bool,
            transform: Optional[torch.nn.Module] = None,
            target_transform: Optional[torch.nn.Module] = None,
            download: Optional[bool] = True,
            use_default_transform: Optional[bool] = True
) -> CIFAR10:
    """"""

    if transform is None and use_default_transform:
        transform = CIFAR_TRAIN_TRANSFORM if train else transforms.ToTensor()

    return CIFAR10(root, train, transform, target_transform, download)


def cifar10_large(root: str,
                  train: bool,
                  transform: Optional[torch.nn.Module] = None,
                  target_transform: Optional[torch.nn.Module] = None,
                  download: Optional[bool] = True,
                  use_default_transform: Optional[bool] = True
) -> CIFAR10:
    """"""

    if transform is None and use_default_transform:
        if train:
            transform = LARGE_CIFAR_TRAIN_TRANSFORM
        else:
            transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor()
            ])

    return CIFAR10(root, train, transform, target_transform, download)


def cifar100(root: str,
             train: bool,
             transform: Optional[torch.nn.Module] = None,
             target_transform: Optional[torch.nn.Module] = None,
             download: Optional[bool] = True,
             use_default_transform: Optional[bool] = True
) -> CIFAR100:
    """"""

    if transform is None and use_default_transform:
        transform = CIFAR_TRAIN_TRANSFORM if train else transforms.ToTensor()

    return CIFAR100(root, train, transform, target_transform, download)


def cifar100_large(root: str,
                   train: bool,
                   transform: Optional[torch.nn.Module] = None,
                   target_transform: Optional[torch.nn.Module] = None,
                   download: Optional[bool] = True,
                   use_default_transform: Optional[bool] = True
) -> CIFAR10:
    """"""

    if transform is None and use_default_transform:
        if train:
            transform = LARGE_CIFAR_TRAIN_TRANSFORM
        else:
            transform = transforms.Compose([
                transforms.Resize((244,244)),
                transforms.ToTensor()
            ])

    return CIFAR100(root, train, transform, target_transform, download)


class CIFAR5M(VisionDataset):
    """"""

    def __init__(self,
                 images: np.array,
                 labels: np.array,
                 transform: Callable[..., Any] = None,
                 target_transform: Callable[..., Any] = None
    ) -> None:
        """"""

        super().__init__("", None, transform, target_transform)

        self.images = images
        self.labels = labels


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = self.images[index]
        label = self.labels[index]

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self) -> int:
        return len(self.images)


def cifar5m(root: str,
            train: bool,
            transform: Optional[torch.nn.Module] = None,
            target_transform: Optional[torch.nn.Module] = None,
            download: Optional[bool] = True,
            use_default_transform: Optional[bool] = True
) -> CIFAR5M:
    """"""

    # Use CIFAR10 validation set if train=False
    if not train:
        return cifar10(root, False, transform, target_transform, download, use_default_transform)

    if transform is None and use_default_transform:
        transform = CIFAR5M_TRAIN_TRANSFORM if train else transforms.ToTensor()

    root_dir = path.join(root, "cifar5m")
    files = [path.join(root_dir, f"cifar5m_part{i}.npz") for i in range(6)]

    # Check if root directory exists
    if not path.isdir(root_dir) and download:
        # TODO: Download option for CIFAR-5M
        print("CIFAR-5M currently does not support download=True.")
        print(f"Please download CIFAR-5M from https://github.com/preetum/cifar5m to {root_dir}")
        raise Exception("CIFAR-5M currently does not support download=True.")

    # Check if all files exist before loading them
    for cifar5m_file in files:
        if not path.isfile(cifar5m_file):
            print(f"{cifar5m_file} not found!")
            print(f"Please download CIFAR-5M from https://github.com/preetum/cifar5m to {root_dir}")
            raise Exception(f"{cifar5m_file} not found!")

    # Load data
    images = []
    labels = []

    for cifar5m_file in files:
        cifar5m_part_data = np.load(cifar5m_file)
        images.extend(cifar5m_part_data["X"])
        labels.extend(cifar5m_part_data["Y"])

    images = np.array(images)
    labels = np.array(labels)

    # Convert to C x H x W format
    # images = np.transpose(images, (0, 3, 1, 2))

    return CIFAR5M(images, labels, transform, target_transform)
