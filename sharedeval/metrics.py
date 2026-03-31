import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    jaccard_score,
    accuracy_score,
    average_precision_score,
)


def precision(y_true, y_pred, average=None, **kwargs):
    return precision_score(y_true, y_pred, average=average, zero_division=0, **kwargs)


def recall(y_true, y_pred, average=None, **kwargs):
    return recall_score(y_true, y_pred, average=average, zero_division=0, **kwargs)


def f1(y_true, y_pred, average=None, **kwargs):
    return f1_score(y_true, y_pred, average=average, zero_division=0, **kwargs)


def jaccard(y_true, y_pred, average=None, **kwargs):
    return jaccard_score(y_true, y_pred, average=average, zero_division=0, **kwargs)


def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def map_for_classification(y_pred, y_true):
    """Mean Average Precision for multi-label classification."""
    try:
        return average_precision_score(y_true, y_pred, average='macro')
    except ValueError:
        return 0.0


def f1max(y_true, y_pred):
    """Find the maximum F1 score across thresholds."""
    thresholds = np.linspace(0, 1, 101)
    best_f1 = 0.0
    for t in thresholds:
        preds_bin = (y_pred > t).astype(int)
        score = f1_score(y_true, preds_bin, average='macro', zero_division=0)
        if score > best_f1:
            best_f1 = score
    return best_f1


def f1max_thres(y_true, y_pred):
    """Find the threshold that maximizes the F1 score."""
    thresholds = np.linspace(0, 1, 101)
    best_f1 = 0.0
    best_thres = 0.5
    for t in thresholds:
        preds_bin = (y_pred > t).astype(int)
        score = f1_score(y_true, preds_bin, average='macro', zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thres = t
    return best_thres


def mloc_iou(y_true, y_pred):
    """Mean Localization IoU (Intersection over Union)."""
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    intersection = np.sum(np.minimum(y_true, y_pred), axis=0)
    union = np.sum(np.maximum(y_true, y_pred), axis=0)
    iou = np.where(union > 0, intersection / union, 0.0)
    return np.mean(iou)
