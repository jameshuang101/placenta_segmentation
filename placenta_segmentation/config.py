"""
Global configuration for paths and hyperparameters.
"""
import os

# Paths (users should override or set via environment variables)
ROOT = os.getenv('PSG_ROOT', '/path/to/placenta_project')
IMG_MAT_DIR = os.path.join(ROOT, 'data/mat/images')
LBL_MAT_DIR = os.path.join(ROOT, 'data/mat/labels')
IMG_NPY_DIR = os.path.join(ROOT, 'data/npy/images')
LBL_NPY_DIR = os.path.join(ROOT, 'data/npy/labels')

# Training hyperparameters
BATCH_SIZE = int(os.getenv('PSG_BATCH_SIZE', 8))
EPOCHS = int(os.getenv('PSG_EPOCHS', 200))
LEARNING_RATE = float(os.getenv('PSG_LR', 1e-4))

# Model parameters
INPUT_SHAPE = (256, 256, 5, 1)
NUM_CLASSES = 3

# Device config for TensorFlow
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass
