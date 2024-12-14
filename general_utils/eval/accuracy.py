import math

import torch


def accuracy(model: torch.nn.Module,
             x: torch.Tensor, # All data
             y: torch.Tensor, # All data
             batch_size: int
) -> float:
    """"""

    corr = 0
    all = 0
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
            output = model(x_batch)
            corr += torch.argmax(output, 1).eq(y_batch).sum().item()
            all += len(y_batch)

    return corr / all


def accuracy_loader(model: torch.nn.Module,
                    data_loader: torch.utils.data.DataLoader
) -> float:
    """"""

    corr = 0
    all = 0

    model.eval()
    device = next(model.parameters()).device

    for input, target in data_loader:
        input, target = input.to(device), target.to(device)

        with torch.no_grad():
            output = model(input)
            corr += torch.argmax(output, 1).eq(target).sum().item()
            all += len(target)

    return corr / all