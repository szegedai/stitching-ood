import torch


def low_rank_approx(x: torch.Tensor, n_dims: int) -> torch.Tensor:

    orig_shape = x.shape
    is_conv = len(orig_shape) == 4

    # Reshape
    if is_conv:
        x = x.permute(0, 2, 3, 1)
        orig_shape = x.shape

    n_features = x.shape[-1]
    x = x.reshape(-1, n_features)

    # Low Rank Approximation
    offset = x.mean(0)
    x = x - offset

    u, s, vt = torch.linalg.svd(x, full_matrices=False)
    s[n_dims:] = 0
    x_lr = u @ torch.diag_embed(s) @ vt
    x_lr = x_lr + offset

    # Convert back to original shape
    x = x_lr.reshape(*orig_shape)

    if is_conv:
        x = x.permute(0, 3, 1, 2)

    return x
