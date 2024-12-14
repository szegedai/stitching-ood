import math
from typing import List, Union, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR

from general_utils.models import get_model
from general_utils.datasets import DATASET_META
from general_utils.train import ClassifierTrainer
from general_utils.train.samplers import MultiEpochDisjointSubsetSampler


def train(model: Union[nn.Module, str],
          dataset_name: str,
          # Training hyperparams
          n_epochs: int,
          batch_size: int,
          optimizer: Union[str, torch.optim.Optimizer],
          lr: float,
          wd: float,
          loss_fn: torch.nn.modules.loss._Loss,
          schedulers: List[Union[torch.optim.lr_scheduler.LRScheduler, str]] = [],
          reduce_lr_epochs: Optional[List[int]] = [], # For MultiStepLR only
          gradient_clip_threshold: Optional[float] = None,
          device: Optional[Union[str, torch.device]] = "cpu",
          # Dataset parameters
          dataset_root: Optional[str] = "./data/pytorch",
          download: Optional[bool] = True,
          train_transform: Optional[nn.Module] = None,
          train_target_transform: Optional[nn.Module] = None,
          use_default_train_transform: Optional[bool] = True,
          val_transform: Optional[nn.Module] = None,
          val_target_transform: Optional[nn.Module] = None,
          use_default_val_transform: Optional[bool] = True,
          # Data loader parameters
          n_workers: Optional[int] = 0,
          prefetch_factor: Optional[int] = None,
          # Model saving parameters
          verbose: Optional[bool] = True,
          save_frequency: Optional[int] = 1,
          save_folder: Optional[str] = "models",
          save_model: Optional[bool] = True,
          save_logs: Optional[bool] = True,
) -> nn.Module:

    # Resolve model if given by type
    if isinstance(model, str):
        n_classes = DATASET_META[dataset_name]["n_classes"]
        n_channels = DATASET_META[dataset_name]["n_channels_in"]
        model = get_model(model, n_classes, n_channels)

    # Put model to device
    model.to(device)


    # Get train and val dataset
    dataset_func = DATASET_META[dataset_name]["dataset_func"]
    train_dataset = dataset_func(root=dataset_root,
                                 train=True,
                                 transform=train_transform,
                                 target_transform=train_target_transform,
                                 download=download,
                                 use_default_transform=use_default_train_transform)
    val_dataset = dataset_func(root=dataset_root,
                               train=False,
                               transform=val_transform,
                               target_transform=val_target_transform,
                               download=download,
                               use_default_transform=use_default_val_transform)

    # Train and val loader setup
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=dataset_name != "cifar5m",
                              num_workers=n_workers,
                              prefetch_factor=prefetch_factor if n_workers > 0 else None,
                              pin_memory=True,
                              pin_memory_device=device,
                              sampler=MultiEpochDisjointSubsetSampler(train_dataset, 50000) if dataset_name=="cifar5m" else None)

    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=n_workers,
                            prefetch_factor=prefetch_factor if n_workers > 0 else None,
                            pin_memory=True,
                            pin_memory_device=device)

    # Optimizer setup
    if type(optimizer) == str:
        assert optimizer in ["sgd", "adam"], f"Optimizer {optimizer} is not supported"

        if optimizer == "adam":
            optimizer = Adam(model.parameters(), lr=lr, weight_decay=wd)
        elif optimizer == "sgd":
            optimizer = SGD(model.parameters(), lr=lr, weight_decay=wd)

    # LR scheduler setup: if reduce_lr_epochs is specified, add MultiStepLR,
    # otherwise convert every other LR scheduler if passed as string
    train_schedulers = []

    if not any(type(scheduler) == MultiStepLR for scheduler in schedulers) and len(reduce_lr_epochs) > 0:
        train_schedulers.append(MultiStepLR(optimizer=optimizer,
                                            milestones=reduce_lr_epochs,
                                            gamma=0.1))

    # Add other schedulers here
    # We only use MultiStepLR for now
    for scheduler in schedulers:
        if scheduler == "" or scheduler is None:
            continue
        elif type(scheduler) == str:
            raise TypeError(f"Scheduler type {scheduler} not supported!")
        else:
            train_schedulers.append(scheduler)

    # Trainer setup
    trainer = ClassifierTrainer(model=model,
                                optimizer=optimizer,
                                loss_fn=loss_fn,
                                n_epochs=n_epochs,
                                schedulers=train_schedulers,
                                save_frequency=save_frequency,
                                save_folder=save_folder,
                                gradient_clip_threshold=gradient_clip_threshold)

    # Train
    trained_model = trainer.fit_loader(train_loader=train_loader,
                                       val_loader=val_loader,
                                       verbose=verbose,
                                       save_logs=save_logs,
                                       save_model=save_model)

    return trained_model
