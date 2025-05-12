#!/usr/bin/env python
"""
CLI: train the segmentation model.
"""
import argparse
from placenta_segmentation.training.trainer import train_model
from placenta_segmentation.config import IMG_NPY_DIR, LBL_NPY_DIR

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_img', default=IMG_NPY_DIR+'/train/images')
    parser.add_argument('--train_lbl', default=LBL_NPY_DIR+'/train/labels')
    parser.add_argument('--val_img',   default=IMG_NPY_DIR+'/val/images')
    parser.add_argument('--val_lbl',   default=LBL_NPY_DIR+'/val/labels')
    parser.add_argument('--out',       required=True, help='Path to save model weights')
    args = parser.parse_args()
    train_model(args.train_img, args.train_lbl, args.val_img, args.val_lbl, args.out)

if __name__ == '__main__':
    main()
