from functools import partial
from typing import Optional, List, Tuple, Union, Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from general_utils.datasets import DATASET_META

from nn_stitching.utils import get_internal_activation
from nn_stitching.utils.ood import get_ood_scores, get_ood_detection_results


def eval_ood_detection(model: nn.Module,
                       ood_dataset: Optional[str] = None,
                       ood_dataset_root: Optional[str] = "./data/pytorch",
                       ood_loader: Optional[DataLoader] = None,
                       id_scores: Optional[List[float]] = None,
                       id_dataset: Optional[str] = None,
                       id_dataset_root: Optional[str] = "./data/pytorch",
                       id_loader: Optional[DataLoader] = None,
                       batch_size: Optional[int] = 64,
                       num_to_avg: Optional[int] = 10,
                       device: Optional[Union[str, torch.device]] = "cuda:0"
) -> Tuple[float, float, float]:
    r"""Evaluates OOD detection capabilities of a model.

    This method uses energy-based OOD detection as described by Liu et al. in
    https://arxiv.org/abs/2010.03759

    Passing either dataset name or data loader is enough for OOD dataset,
    both are not required. Similarly for the ID dataset, passing either the ID
    energy scores, dataset name or data loader is enough, more than one is not
    required.

    Example call:
    eval_ood_detection(model, id_dataset="cifar10", ood_dataset="svhn")

    Returns OOD detection AUROC, AUPR and FPR95
    """

    assert ood_dataset is not None or ood_loader is not None, \
    "For OOD evaluation please provide either OOD dataset name or OOD data loader"

    assert id_scores is not None or id_dataset is not None or id_loader is not None, \
    "For OOD evaluation please provide either ID scores or ID dataset name or ID data loader."

    if id_scores is None:

        if id_loader is None:
            id_dataset_func = DATASET_META[id_dataset]["dataset_func"]
            id_dataset = id_dataset_func(root=id_dataset_root, train=False)
            id_loader = DataLoader(id_dataset, batch_size=batch_size, shuffle=False)

        id_scores, _, _ = get_ood_scores(model,
                                         id_loader,
                                         batch_size,
                                         ood_num_examples=len(id_dataset),
                                         temperature=1.0,
                                         use_xent=False,
                                         score="energy",
                                         in_dist=True,
                                         device=device)

    if ood_loader is None:
        ood_dataset_func = DATASET_META[ood_dataset]["dataset_func"]
        ood_dataset = ood_dataset_func(root=ood_dataset_root, train=False)
        ood_loader = DataLoader(ood_dataset, batch_size=batch_size, shuffle=False)

    auroc, aupr, fpr = get_ood_detection_results(model,
                                                 ood_loader,
                                                 batch_size,
                                                 ood_num_examples=len(ood_loader) * batch_size,
                                                 in_score=id_scores,
                                                 temperature=1.0,
                                                 use_xent=False,
                                                 score="energy",
                                                 num_to_avg=num_to_avg,
                                                 in_dist=False,
                                                 device=device)

    return auroc, aupr, fpr


def eval_representation_ood_detection(model: nn.Module,
                                      layer_name: str,
                                      detector: nn.Module,
                                      ood_dataset: Optional[str] = None,
                                      ood_dataset_root: Optional[str] = "./data/pytorch",
                                      ood_loader: Optional[DataLoader] = None,
                                      id_scores: Optional[List[float]] = None,
                                      id_dataset: Optional[str] = None,
                                      id_dataset_root: Optional[str] = "./data/pytorch",
                                      id_loader: Optional[DataLoader] = None,
                                      batch_size: Optional[int] = 64,
                                      num_to_avg: Optional[int] = 10,
                                      device: Optional[Union[str, torch.device]] = "cuda:0",
                                      generator: Optional[Any] = None,
                                      save_to = None
) -> Tuple[float, float, float]:
    r"""Performs OOD detection on a model's hidden representations.

    This method uses energy-based OOD detection as described by Liu et al. in
    https://arxiv.org/abs/2010.03759

    Passing either dataset name or data loader is enough for OOD dataset,
    both are not required. Similarly for the ID dataset, passing either the ID
    energy scores, dataset name or data loader is enough, more than one is not
    required.

    Example call:
    eval_representation_ood_detection(model, "block1.layer.0", detector,
                                      id_dataset="cifar10", ood_dataset="svhn")

    Returns OOD detection AUROC, AUPR and FPR95
    """

    assert ood_dataset is not None or ood_loader is not None, \
    "For OOD evaluation please provide either OOD dataset name or OOD data loader"

    assert id_scores is not None or id_dataset is not None or id_loader is not None, \
    "For OOD evaluation please provide either ID scores or ID dataset name or ID data loader."

    if generator is None:
        generator = partial(get_internal_activation, model=model, layer_name=layer_name)

    if id_scores is None:

        if id_loader is None:
            id_dataset_func = DATASET_META[id_dataset]["dataset_func"]
            id_dataset = id_dataset_func(root=id_dataset_root, train=False)
            id_loader = DataLoader(id_dataset, batch_size=batch_size, shuffle=False)

        id_scores, _, _ = get_ood_scores(detector,
                                         id_loader,
                                         batch_size,
                                         ood_num_examples=len(id_dataset),
                                         generator=generator,
                                         temperature=1.0,
                                         use_xent=False,
                                         score="energy",
                                         in_dist=True,
                                         device=device)

    if ood_loader is None:
        ood_dataset_func = DATASET_META[ood_dataset]["dataset_func"]
        ood_dataset = ood_dataset_func(root=ood_dataset_root, train=False)
        ood_loader = DataLoader(ood_dataset, batch_size=batch_size, shuffle=False)

    auroc, aupr, fpr = get_ood_detection_results(detector,
                                                 ood_loader,
                                                 batch_size,
                                                 ood_num_examples=len(ood_loader) * batch_size,
                                                 in_score=id_scores,
                                                 generator=generator,
                                                 temperature=1.0,
                                                 use_xent=False,
                                                 score="energy",
                                                 num_to_avg=num_to_avg,
                                                 in_dist=False,
                                                 device=device,
                                                 save_to=save_to)

    return auroc, aupr, fpr
