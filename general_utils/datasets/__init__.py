from .cifar import cifar10, cifar100, cifar5m, cifar10_large, cifar100_large
from .imagenet import imagenet, imagenet_mem
from .mnist import mnist, fashion_mnist
from .svhn import svhn
from .oe import outlier_exposure

DATASET_META = {
    "mnist": {
        "n_classes": 10,
        "n_channels_in": 1,
        "dataset_func": mnist,
        "shape": (1, 28, 28)
    },
    "fashion": {
        "n_classes": 10,
        "n_channels_in": 1,
        "dataset_func": fashion_mnist,
        "shape": (1, 28, 28)
    },
    "cifar10": {
        "n_classes": 10,
        "n_channels_in": 3,
        "dataset_func": cifar10,
        "shape": (3, 32, 32)
    },
    "cifar100": {
        "n_classes": 100,
        "n_channels_in": 3,
        "dataset_func": cifar100,
        "shape": (3, 32, 32)
    },
    "cifar10_large": {
        "n_classes": 10,
        "n_channels_in": 3,
        "dataset_func": cifar10_large,
        "shape": (3, 224, 224)
    },
    "cifar100_large": {
        "n_classes": 100,
        "n_channels_in": 3,
        "dataset_func": cifar100_large,
        "shape": (3, 224, 224)
    },
    "cifar5m": {
        "n_classes": 10,
        "n_channels_in": 3,
        "dataset_func": cifar5m,
        "shape": (3, 32, 32)
    },
    "svhn": {
        "n_classes": 10,
        "n_channels_in": 3,
        "dataset_func": svhn,
        "shape": (3, 32, 32)
    },
    "imagenet": {
        "n_classes": 1000,
        "n_channels_in": 3,
        "dataset_func": imagenet,
        "shape": (3, 224, 224)
    },
    "imagenet_mem": {
        "n_classes": 1000,
        "n_channels_in": 3,
        "dataset_func": imagenet_mem,
        "shape": (3, 224, 224)
    },
    "outlier_exposure": {
        "n_classes": 1, # unsupervised
        "n_channels_in": 3,
        "dataset_func": outlier_exposure,
        "shape": (3, 32, 32)
    }
}
