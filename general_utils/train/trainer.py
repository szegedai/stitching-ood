from abc import ABC, abstractmethod
import os
from typing import List, Optional

import torch
import pandas as pd


class Trainer(ABC):
    """"""

    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 loss_fn: torch.nn.modules.loss._Loss,
                 n_epochs: int,
                 schedulers: List[torch.optim.lr_scheduler._LRScheduler],
                 save_frequency: int,
                 save_folder: str,
                 gradient_clip_threshold: float,
    ) -> None:
        """"""

        self._model = model
        self._optimizer = optimizer
        self._loss_fn = loss_fn
        self._n_epochs = n_epochs
        self._schedulers = schedulers
        self._save_folder = save_folder
        self._save_frequency = save_frequency
        self._metrics = {}
        self.gradient_clip_threshold = gradient_clip_threshold

    def fit(self,
            x_train: torch.Tensor,
            y_train: torch.Tensor,
            x_val: torch.Tensor,
            y_val: torch.Tensor,
            batch_size: int,
            verbose: bool = True,
            save_logs: bool = False,
            save_model: bool = False
    ) -> torch.nn.Module:
        """"""

        if save_model or save_logs:
            _create_folder(self._save_folder)

        for epoch in range(1, self._n_epochs + 1):
            self._train_step(x_train, y_train, batch_size, epoch, verbose)
            self._val_step(x_val, y_val, batch_size, epoch, verbose)

            for scheduler in self._schedulers:
                scheduler.step()

            if save_logs:
                self._save_logs()

            if save_model and epoch % self._save_frequency == 0:
                self._save_model(epoch)

        return self._model

    def fit_loader(self,
                   train_loader: torch.utils.data.DataLoader,
                   val_loader: torch.utils.data.DataLoader,
                   verbose: bool = True,
                   save_logs: bool = True,
                   save_model: bool = False
    ) -> torch.nn.Module:
        """"""

        if save_model or save_logs:
            _create_folder(self._save_folder)

        for epoch in range(1, self._n_epochs + 1):
            self._train_step_loader(train_loader, epoch, verbose)
            self._val_step_loader(val_loader, epoch, verbose)

            for scheduler in self._schedulers:
                scheduler.step()

            if save_logs:
                self._save_logs()

            if save_model and epoch % self._save_frequency == 0:
                self._save_model(epoch)

        return self._model

    @abstractmethod
    def _train_step(self,
                    x_train: torch.Tensor,
                    y_train: torch.Tensor,
                    batch_size: int,
                    epoch: int,
                    verbose: bool
    ) -> None:
        """"""

        raise NotImplementedError("Abstract class Trainer foes not implement "
                                  "_val_step")

    @abstractmethod
    def _val_step(self,
                  x_val: torch.Tensor,
                  y_val: torch.Tensor,
                  batch_size: int,
                  epoch: int,
                  verbose: bool
    ) -> None:
        """"""

        raise NotImplementedError("Abstract class Trainer foes not implement "
                                  "_val_step")

    @abstractmethod
    def _train_step_loader(self,
                           train_loader: torch.utils.data.DataLoader,
                           epoch: int,
                           verbose: bool
    ) -> None:
        """"""

        raise NotImplementedError("Abstract class Trainer does not implement "
                                  "_train_step_loader")

    @abstractmethod
    def _val_step_loader(self,
                         val_loader: torch.utils.data.DataLoader,
                         epoch: int,
                         verbose: bool
    ) -> None:
        """"""

        raise NotImplementedError("Abstract class Trainer does not implement "
                                  "_val_step_loader")

    def _save_logs(self) -> None:
        """"""

        log_df = pd.DataFrame(self._metrics)
        log_file = os.path.join(self._save_folder, "training_logs.csv")
        log_df.to_csv(log_file, index=False)

    def _save_model(self, epoch: int) -> None:
        """"""

        model_name = f"model_{epoch}.pt"
        model_file = os.path.join(self._save_folder, model_name)
        torch.save(self._model.state_dict(),
                   model_file,
                   _use_new_zipfile_serialization=False)


def _create_folder(folder: str) -> None:
    """"""

    if not os.path.exists(folder):
        os.makedirs(folder)
