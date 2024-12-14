import math
from typing import List, Optional, Dict

import torch


def eval_combined(model: torch.nn.Module,
                  x: torch.Tensor,
                  y: torch.Tensor,
                  batch_size: int,
                  metrics: List[str],
                  loss_fn: Optional[torch.nn.modules.loss._Loss] = None,
) -> Dict[str, float]:

    sum_metrics = {}
    for metric in metrics:
        sum_metrics[metric] = 0

    n_samples = len(y)
    n_batches = math.ceil(n_samples / batch_size)

    model.eval()
    device = next(model.parameters()).device

    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, n_samples)

        x_batch = x[batch_start:batch_end].detach().clone()
        y_batch = y[batch_start:batch_end]

        x_batch, y_batch = x_batch.to(device), y_batch.to(device)


        with torch.no_grad():
            if "acc" in metrics or "loss" in metrics:
                output = model(x_batch)


            for metric in metrics:
                if metric == "acc":
                    sum_metrics["acc"] += torch.argmax(output, 1).eq(y_batch).sum().item()
                elif metric == "loss" and loss_fn is not None:
                    sum_metrics["loss"] += loss_fn(output, y_batch).item() * len(y_batch)

    for metric in metrics:
        sum_metrics[metric] /= n_samples

    return sum_metrics


def eval_combined_loader(model: torch.nn.Module,
                         data_loader: torch.utils.data.DataLoader,
                         metrics: List[str],
                         loss_fn: Optional[torch.nn.modules.loss._Loss] = None,
) -> Dict[str, float]:

    n_samples = 0
    sum_metrics = {}
    for metric in metrics:
        sum_metrics[metric] = 0

    model.eval()
    device = next(model.parameters()).device

    for x_batch, y_batch in data_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)


        with torch.no_grad():
            if "acc" in metrics or "loss" in metrics:
                output = model(x_batch)


            for metric in metrics:
                if metric == "acc":
                    sum_metrics["acc"] += torch.argmax(output, 1).eq(y_batch).sum().item()
                elif metric == "loss" and loss_fn is not None:
                    sum_metrics["loss"] += loss_fn(output, y_batch).item() * len(y_batch)

            n_samples += len(y_batch)


    for metric in metrics:
        sum_metrics[metric] /= n_samples

    return sum_metrics
