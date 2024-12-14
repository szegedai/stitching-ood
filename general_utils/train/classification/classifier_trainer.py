import math
from typing import Optional

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from general_utils.eval import eval_combined_loader, eval_combined
from general_utils.train import Trainer


class ClassifierTrainer(Trainer):
    """"""

    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 loss_fn: ... = torch.nn.CrossEntropyLoss(),
                 n_epochs: int = 30,
                 schedulers: ... = [],
                 save_frequency: int = 1,
                 save_folder: str = "models",
                 gradient_clip_threshold: float = None,
    ) -> None:
        """"""

        super().__init__(model,
                         optimizer,
                         loss_fn,
                         n_epochs,
                         schedulers,
                         save_frequency,
                         save_folder,
                         gradient_clip_threshold,)

        self._metrics = {
            "train_acc": [],
            "train_loss": [],
            "val_acc": [],
            "val_loss": []
        }

    def _train_step(self,
                    x_train: torch.Tensor,
                    y_train: torch.Tensor,
                    batch_size: int,
                    epoch: int,
                    verbose: bool
    ) -> None:
        """"""

        device = next(self._model.parameters()).device

        n_samples = len(y_train)
        n_batches = math.ceil(n_samples / batch_size)
        x_idx = np.arange(n_samples)
        np.random.shuffle(x_idx)

        sum_acc = 0
        sum_loss = 0

        for batch_idx in tqdm(range(n_batches)):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, n_samples)

            x_batch = x_train[x_idx[batch_start:batch_end]].detach().clone()
            y_batch = y_train[x_idx[batch_start:batch_end]]

            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            self._optimizer.zero_grad()
            output = self._model(x_batch)
            loss = self._loss_fn(output, y_batch)
            loss.backward()

            if self.gradient_clip_threshold is not None:
                nn.utils.clip_grad_norm_(self._model.parameters(), self.gradient_clip_threshold)

            self._optimizer.step()

            sum_loss += loss.item() * len(y_batch)
            sum_acc += torch.argmax(output, 1).eq(y_batch).sum().item()

        acc = sum_acc / n_samples
        loss = sum_loss / n_samples

        self._metrics["train_acc"].append(acc)
        self._metrics["train_loss"].append(loss)

        # Print progress in verbose mode
        if verbose:
            print(f"Epoch #{epoch} train loss: {loss}, accuracy: {acc}")

    def _val_step(self,
                  x_val: torch.Tensor,
                  y_val: torch.Tensor,
                  batch_size: int,
                  epoch: int,
                  verbose: bool
    ) -> None:
        """"""

        self._model.eval()

        # acc = accuracy(self._model, x_val, y_val, batch_size)
        # loss = mean_loss(self._model, x_val, y_val, batch_size, self._loss_fn)

        metrics = eval_combined(self._model, x_val, y_val, batch_size, ["acc", "loss"], self._loss_fn)

        self._metrics["val_acc"].append(metrics["acc"])
        self._metrics["val_loss"].append(metrics["loss"])

        # Print progress in verbose mode
        if verbose:
            print(f"Epoch #{epoch} val loss: {metrics['loss']}, accuracy: {metrics['acc']}")

    def _train_step_loader(self,
                           train_loader: torch.utils.data.DataLoader,
                           epoch: int,
                           verbose: bool
    ) -> None:
        """"""

        device = next(self._model.parameters()).device

        self._model.train()

        sum_acc = 0
        sum_loss = 0
        n_samples = 0

        for input, target in tqdm(train_loader):
            input, target = input.to(device), target.to(device)

            self._optimizer.zero_grad()
            output = self._model(input)
            loss = self._loss_fn(output, target)
            loss.backward()

            if self.gradient_clip_threshold is not None:
                nn.utils.clip_grad_norm_(self._model.parameters(), self.gradient_clip_threshold)

            self._optimizer.step()

            sum_loss += loss.item() * len(target)
            # sum_acc += torch.argmax(output, 1).eq(target).sum().item()
            sum_acc += torch.argmax(output, 1).eq(target).sum().item()
            n_samples += len(target)

        acc = sum_acc / n_samples
        loss = sum_loss / n_samples

        self._metrics["train_acc"].append(acc)
        self._metrics["train_loss"].append(loss)

        # Print progress in verbose mode
        if verbose:
            print(f"Epoch #{epoch} train loss: {loss}, accuracy: {acc}")

    def _val_step_loader(self,
                         val_loader: torch.utils.data.DataLoader,
                         epoch: int,
                         verbose: bool
    ) -> None:
        """"""

        self._model.eval()

        # acc = accuracy_loader(self._model, val_loader)
        # loss = mean_loss_loader(self._model, val_loader, self._loss_fn)

        metrics = eval_combined_loader(self._model, val_loader, ["acc", "loss"], self._loss_fn)

        self._metrics["val_acc"].append(metrics["acc"])
        self._metrics["val_loss"].append(metrics["loss"])

        # Print progress in verbose mode
        if verbose:
            print(f"Epoch #{epoch} val loss: {metrics['loss']}, accuracy: {metrics['acc']}")
