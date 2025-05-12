"""
Trainer module: handles data generators, model compilation, and training loop.
"""
import os
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from placenta_segmentation.config import BATCH_SIZE, EPOCHS, LEARNING_RATE, INPUT_SHAPE, NUM_CLASSES
from placenta_segmentation.models.networks import UNet3Plus
from placenta_segmentation.models.losses import combined_loss, dice_coef
from placenta_segmentation.data.preprocessing import block_volumes
from placenta_segmentation.data.utils import list_files

def data_generator(img_dir, lbl_dir):
    """
    Yields batches of (images, labels) for training.
    """
    files = list_files(img_dir, ext='.npy')
    n = len(files)
    idx = 0
    while True:
        x_batch = np.zeros((BATCH_SIZE, *INPUT_SHAPE), dtype=np.float32)
        y_batch = np.zeros((BATCH_SIZE, *INPUT_SHAPE[:3], NUM_CLASSES), dtype=np.uint8)
        for i in range(BATCH_SIZE):
            fname = files[idx]
            img = np.load(os.path.join(img_dir, fname))
            lbl = np.load(os.path.join(lbl_dir, fname))
            # simple augmentation: horizontal flip
            if np.random.rand() < 0.5:
                img = np.flip(img, axis=1)
                lbl = np.flip(lbl, axis=1)
            x_batch[i] = img[..., np.newaxis]
            y_batch[i] = lbl
            idx = (idx + 1) % n
        yield x_batch, y_batch

def train_model(train_img, train_lbl, val_img, val_lbl, weights_path):
    """
    Compile and train the UNet3Plus model.
    """
    model = UNet3Plus()
    model.compile(optimizer=Adam(LEARNING_RATE),
                  loss=combined_loss,
                  metrics=[dice_coef, 'accuracy'])
    ckpt = ModelCheckpoint(weights_path, save_best_only=True, monitor='val_loss', mode='min')
    es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)
    model.fit(
        data_generator(train_img, train_lbl),
        validation_data=data_generator(val_img, val_lbl),
        steps_per_epoch=100, validation_steps=20,
        epochs=EPOCHS, callbacks=[ckpt, es]
    )
