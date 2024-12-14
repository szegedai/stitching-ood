import math
import torch


def mean_loss(model: torch.nn.Module,
              x: torch.Tensor,
              y: torch.Tensor,
              batch_size: int,
              loss_fn: torch.nn.modules.loss._Loss
) -> float:
    """"""

    sum_loss = 0
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
            sum_loss += loss_fn(output, y_batch).item() * len(y_batch)

    return sum_loss / n_samples


def mean_loss_loader(model: torch.nn.Module,
                     data_loader: torch.utils.data.DataLoader,
                     loss_fn: torch.nn.modules.loss._Loss
) -> float:
    """"""

    n_samples = 0
    sum_loss = 0

    model.eval()
    device = next(model.parameters()).device

    for input, target in data_loader:
        input, target = input.to(device), target.to(device)

        with torch.no_grad():
            output = model(input)
            sum_loss += loss_fn(output, target).item() * len(target)
            n_samples += len(target)

    return sum_loss / n_samples
