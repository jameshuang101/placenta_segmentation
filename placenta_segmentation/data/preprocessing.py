"""
Data loading and preprocessing: converts .mat to normalized .npy volumes,
crops empty slices, and blocks volumes for training.
"""
import os
import numpy as np
import scipy.io
from tqdm import tqdm
from .utils import list_files

def mat_to_npy(mat_dir: str, npy_dir: str, var_name: str):
    """
    Convert all .mat files in mat_dir to .npy arrays saved in npy_dir.
    Args:
        mat_dir: directory containing .mat files
        npy_dir: target directory for .npy outputs
        var_name: variable name inside .mat ('mrImage','plLabel','utLabel')
    """
    os.makedirs(npy_dir, exist_ok=True)
    for fname in list_files(mat_dir, ext='.mat'):
        data = scipy.io.loadmat(os.path.join(mat_dir, fname))[var_name]
        np.save(os.path.join(npy_dir, fname.replace('.mat', '.npy')), data.astype(np.float32))

def normalize_and_crop(img_npy: str, lbl_npy: str, out_img: str, out_lbl: str):
    """
    Normalize intensities to [0,1] using 5th–99.9th percentiles,
    find the first/last slice containing any label, crop and save.
    """
    image = np.load(img_npy)
    label = np.load(lbl_npy)
    p5, p999 = np.percentile(image, [5, 99.9])
    image = np.clip((image - p5) / (p999 - p5), 0, 1).astype(np.float32)

    # find non-empty region along axis=2
    mask = (label.sum(axis=(0,1)) > 0)
    start = np.argmax(mask)
    end = mask.shape[0] - np.argmax(mask[::-1]) 
    image_c = image[:, :, start:end]
    label_c = label[:, :, start:end]

    np.save(out_img, image_c)
    np.save(out_lbl, label_c)

def block_volumes(img_dir: str, lbl_dir: str, out_img_dir: str, out_lbl_dir: str, block_size: int =5):
    """
    Break each volume into sliding blocks of block_size along axis=2.
    """
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)
    for fname in list_files(img_dir, ext='.npy'):
        vol = np.load(os.path.join(img_dir, fname))
        lbl = np.load(os.path.join(lbl_dir, fname))
        for i in tqdm(range(vol.shape[2] - block_size + 1), desc=f'Blocking {fname}'):
            img_blk = vol[:, :, i:i+block_size]
            lbl_blk = lbl[:, :, i:i+block_size]
            np.save(os.path.join(out_img_dir, fname.replace('.npy', f'_blk{i:02d}.npy')), img_blk)
            np.save(os.path.join(out_lbl_dir, fname.replace('.npy', f'_blk{i:02d}.npy')), lbl_blk)
