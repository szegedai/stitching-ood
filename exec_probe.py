import argparse
import sys
from typing import Optional, Union

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from general_utils.general import set_random_seed
from general_utils.models import get_model
from general_utils.datasets import DATASET_META
from general_utils.eval import accuracy_loader

from nn_stitching.utils import get_internal_activation
from nn_stitching.utils.low_rank import low_rank_approx


def freeze(model: nn.Module) -> nn.Module:
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model


class ConvProber(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 layer_name: str,
                 n_classes: int,
                 hidden_size: int,
                 rank: Optional[int] = None,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = freeze(model)
        self.layer_name = layer_name
        self.probe = nn.Linear(hidden_size, n_classes)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.rank = rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activation = get_internal_activation(x, None, self.model, self.layer_name)
        if self.rank:
            activation = low_rank_approx(activation, self.rank)
        return self.probe(torch.flatten(self.pool(activation), 1))

    def train(self, mode: bool = True):
        self.probe.train(mode)
        return self

    def eval(self):
        self.probe.eval()
        return self


class TransformerProber(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 layer_name: str,
                 n_classes: int,
                 hidden_size: int,
                 cls_token: bool = True,
                 rank: Optional[int] = None,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = freeze(model)
        self.layer_name = layer_name
        self.probe = nn.Linear(hidden_size, n_classes)
        self.cls_token = cls_token
        self.rank = rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activation = get_internal_activation(x, None, self.model, self.layer_name)
        if self.rank:
            activation = low_rank_approx(activation, self.rank)

        if self.cls_token:
            activation = activation[:, 0]
        else:
            activation = activation.mean(dim=1)

        return self.probe(activation)


    def train(self, mode: bool = True):
        self.probe.train(mode)
        return self

    def eval(self):
        self.probe.eval()
        return self


def train_prober(model_path: str,
                 model_type: str,
                 layer_name: str,
                 dataset: str,
                 dataset_root: str,
                 batch_size: int,
                 n_epochs: int,
                 learning_rate: float,
                 weight_decay: float,
                 seed: int,
                 n_workers: int,
                 prefetch_factor: int,
                 device: Union[str, torch.device],
                 rank: Optional[int] = None,
                 verbose: bool = False
) -> float:
    set_random_seed(seed)

    # Dataset and data loader setup
    n_classes = DATASET_META[dataset]["n_classes"]
    n_channels = DATASET_META[dataset]["n_channels_in"]
    input_shape = DATASET_META[dataset]["shape"]
    dataset_func = DATASET_META[dataset]["dataset_func"]
    train_dataset = dataset_func(root=dataset_root, train=True, download=True, use_default_transform=True)
    val_dataset = dataset_func(root=dataset_root, train=False, download=True, use_default_transform=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=n_workers, prefetch_factor=prefetch_factor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=n_workers, prefetch_factor=prefetch_factor)

    # Load source model
    model = get_model(model_type, n_classes, n_channels)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.to(device)
    model.eval()

    # Setup prober
    fake_input = torch.rand((2, *input_shape)).to(device)
    activation = get_internal_activation(fake_input, None, model, layer_name)

    if len(activation.shape) == 4:
        hidden_size = activation.shape[1] # BxCxHxW -> C
        prober = ConvProber(model, layer_name, n_classes, hidden_size, rank)
    elif len(activation.shape) == 3:
        hidden_size = activation.shape[2] # BxNxD -> D
        prober = TransformerProber(model, layer_name, n_classes, hidden_size, rank=rank)
    elif len(activation.shape) == 2:
        hidden_size = activation.shape[1] # BxD -> D
        # not used in the article, not implemented here

    prober.to(device)

    # Training loop
    optimizer = torch.optim.Adam(prober.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [n_epochs - 5], 0.1)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(1, n_epochs + 1):

        # Train loop
        prober.train()
        sum_acc = 0
        n_samples = 0

        for x, y in train_loader:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            out = prober(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

            sum_acc += torch.argmax(out, 1).eq(y).sum().item()
            n_samples += len(y)

        scheduler.step()

        if verbose:
            print(f"Epoch #{epoch} train acc: {sum_acc / n_samples}")

            prober.eval()
            val_acc = accuracy_loader(prober, val_loader)
            print(f"Epoch #{epoch} val acc: {val_acc}")

    if verbose:
        return val_acc
    else:
        return accuracy_loader(prober, val_loader)


def _parse_args(args):
    parser = argparse.ArgumentParser(description="Representation probing")
    parser.add_argument("model_path", type=str)
    parser.add_argument("model_type", type=str)
    parser.add_argument("-l", "--layers", type=str, nargs="+", dest="layer_names")

    parser.add_argument("-d", "--dataset", type=str, default="cifar10", dest="dataset")
    parser.add_argument("-data", "--dataset-root", type=str, default="./data/pytorch", dest="dataset_root")
    parser.add_argument("-b", "--batch-size", type=int, default=64, dest="batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=10, dest="n_epochs")
    parser.add_argument("-lr", "--learning-rate", type=float, default=1e-3, dest="learning_rate")
    parser.add_argument("-wd", "--weight_decay", type=float, default=0, dest="weight_decay")
    parser.add_argument("-s", "--seed", type=int, default=0, dest="seed")
    parser.add_argument("-workers", "--loader-workers", type=int, default=1, dest="n_workers")
    parser.add_argument("-prefetch", "--prefetch-factor", type=int, default=1, dest="prefetch_factor")
    parser.add_argument("-gpu", "--gpu", type=int, default=0, dest="gpu")
    parser.add_argument("-ranks", "--ranks", type=int, default=[None], dest="ranks", nargs="*")
    parser.add_argument("--verbose", action="store_true", default=False, dest="verbose")

    return parser.parse_args(args)


def run(conf):
    set_random_seed(conf.seed)

    # General setup
    device = f"cuda:{conf.gpu}" if conf.gpu >= 0 else "cpu"

    for layer in conf.layer_names:
        probing_accuracies = {}

        for rank in conf.ranks:
            acc = train_prober(conf.model_path,
                            conf.model_type,
                            layer,
                            conf.dataset,
                            conf.dataset_root,
                            conf.batch_size,
                            conf.n_epochs,
                            conf.learning_rate,
                            conf.weight_decay,
                            conf.seed,
                            conf.n_workers,
                            conf.prefetch_factor,
                            device,
                            rank,
                            verbose=conf.verbose)
            probing_accuracies[rank] = acc

        probing_accuracies = dict(sorted(probing_accuracies.items(), reverse=True))

        print(f"{conf.model_path} -- {layer}")
        for k, v in probing_accuracies.items():
            print(f"Rank {k}: {v}")

        print()

        for v in probing_accuracies.values():
            print(v)



if __name__ == "__main__":
    print(sys.argv)
    args = sys.argv[1:]
    conf = _parse_args(args)
    run(conf)
    print("\n\n")
