"""
Load a trained model and run inference on new volumes.
"""
import os
import numpy as np
from tensorflow.keras.models import load_model
from placenta_segmentation.models.networks import UNet3Plus
from placenta_segmentation.config import INPUT_SHAPE

def predict_volume(model_weights: str, img_npy: str, out_npy: str):
    """
    Predict segmentation for one volume:
      - load .npy image volume (D×H×W)
      - break into blocks of depth INPUT_SHAPE[2]
      - run model.predict and aggregate results
      - save full-volume label map as .npy
    """
    model = UNet3Plus(weights=model_weights)
    vol = np.load(img_npy)
    D, H, W = vol.shape
    blk = INPUT_SHAPE[2]
    seg_vol = np.zeros((D, H, W), dtype=np.uint8)
    for i in range(D - blk + 1):
        patch = vol[i:i+blk]
        patch_in = patch[np.newaxis, ..., np.newaxis]  # 1×H×W×blk×1
        pred = model.predict(patch_in)[0]  # H×W×blk×3
        labels = np.argmax(pred, axis=-1)
        seg_vol[i:i+blk] = np.maximum(seg_vol[i:i+blk], labels)
    np.save(out_npy, seg_vol)
