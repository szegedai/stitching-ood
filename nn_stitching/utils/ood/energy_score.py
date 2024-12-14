"""
Implementation taken from Liu et al. (2020): https://arxiv.org/abs/2010.03759
"""

from typing import Optional, Callable, Union

import torch
import numpy as np
import sklearn.metrics as sk
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


RECALL_LEVEL = 0.95


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level=RECALL_LEVEL, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                 np.array_equal(classes, [-1, 1]) or
                 np.array_equal(classes, [0]) or
                 np.array_equal(classes, [-1]) or
                 np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])


def get_measures(_pos, _neg, recall_level=.95):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr


concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()


def get_ood_scores(model: nn.Module,
                   loader: DataLoader,
                   test_bs: int,
                   ood_num_examples: int,
                   generator: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                   temperature: Optional[float] = 1.0,
                   use_xent: Optional[bool] = False,
                   score: Optional[str] = "energy",
                   in_dist: Optional[bool] = False,
                   device: Optional[Union[str, torch.device]] = "cpu"
):
    _score = []
    _right_score = []
    _wrong_score = []

    # with torch.no_grad():
    for batch_idx, (data, target) in enumerate(loader):
        if batch_idx >= ood_num_examples // test_bs and in_dist is False:
            break

        data, target = data.to(device), target.to(device)

        if generator: data = generator(data, target)

        output = model(data)
        smax = to_np(F.softmax(output, dim=1))

        if use_xent:
            _score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1))))
        else:
            if score == "energy":
                _score.append(-to_np((temperature*torch.logsumexp(output / temperature, dim=1))))
            else: # original energy and Mahalanobis (but Mahalanobis won't need this returned)
                _score.append(-np.max(smax, axis=1))

        if in_dist:
            preds = np.argmax(smax, axis=1)
            targets = target.cpu().numpy().squeeze()
            right_indices = preds == targets
            wrong_indices = np.invert(right_indices)

            if use_xent:
                _right_score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[right_indices])
                _wrong_score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[wrong_indices])
            else:
                _right_score.append(-np.max(smax[right_indices], axis=1))
                _wrong_score.append(-np.max(smax[wrong_indices], axis=1))

    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(_score)[:ood_num_examples].copy()


def get_ood_detection_results(model: nn.Module,
                              loader: DataLoader,
                              test_bs: int,
                              ood_num_examples: int,
                              in_score: float, # From previous in-dist measurements
                              generator: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                              temperature: Optional[float] = 1.0,
                              num_to_avg: Optional[int] = 10,
                              use_xent: Optional[bool] = False,
                              score: Optional[str] = "energy", # an option, "MSP" can be used
                              in_dist: Optional[bool] = False,
                              out_as_positive: Optional[bool] = False,
                              device: Optional[Union[str, torch.device]] = None,
                              save_to: Optional[str] = None
):

    aurocs, auprs, fprs = [], [], []

    for _ in range(num_to_avg):
        out_score = get_ood_scores(model, loader, test_bs, ood_num_examples, generator, temperature, use_xent, score, in_dist, device)

        if in_dist:
            out_score = out_score[0]

        if save_to: np.save(save_to, out_score)

        if out_as_positive : # OE's defines out samples as positive
            measures = get_measures(out_score, in_score)
        else:
            measures = get_measures(-in_score, -out_score)

        aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])

    auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)
    # auroc_list.append(auroc); aupr_list.append(aupr); fpr_list.append(fpr)

    return auroc, aupr, fpr


def get_id_metrics(pos, neg, recall_level=RECALL_LEVEL):
    '''
    :param pos: 1's class, class to detect, outliers, or wrongly predicted
    example scores
    :param neg: 0's class scores
    '''

    auroc, aupr, fpr = get_measures(pos[:], neg[:], recall_level)
    return auroc, aupr, fpr
