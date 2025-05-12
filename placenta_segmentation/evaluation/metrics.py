"""
Evaluation metrics: Dice coefficient, Hausdorff distance, volume difference.
"""
import numpy as np
from scipy.spatial.distance import directed_hausdorff

def dice_score(y_true: np.ndarray, y_pred: np.ndarray, class_id: int):
    t = (y_true == class_id).astype(np.uint8)
    p = (y_pred == class_id).astype(np.uint8)
    inter = np.sum(t * p)
    if inter == 0 and np.sum(t) == 0:
        return 1.0
    return 2*inter / (np.sum(t) + np.sum(p))

def hausdorff_distance(y_true: np.ndarray, y_pred: np.ndarray, class_id: int):
    t = (y_true == class_id).astype(np.uint8)
    p = (y_pred == class_id).astype(np.uint8)
    hd = 0.0
    for i in range(t.shape[2]):
        hd = max(hd, directed_hausdorff(t[:,:,i], p[:,:,i])[0])
    return hd

def volume_difference(y_true: np.ndarray, y_pred: np.ndarray, class_id: int):
    t = (y_true == class_id).astype(np.uint8)
    p = (y_pred == class_id).astype(np.uint8)
    diff = np.sum(p) - np.sum(t)
    perc = diff / (np.sum(t) + 1e-9)
    return diff, perc
