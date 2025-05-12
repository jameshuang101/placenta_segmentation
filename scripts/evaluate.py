#!/usr/bin/env python
"""
CLI: compute metrics on a test set.
"""
import argparse
import os
import numpy as np
from placenta_segmentation.evaluation.metrics import dice_score, hausdorff_distance, volume_difference

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', required=True, help='Directory of predicted .npy')
    parser.add_argument('--gt_dir',   required=True, help='Directory of ground-truth .npy')
    parser.add_argument('--class_id', type=int, default=1, help='Class to evaluate (1=placenta,2=uterus)')
    args = parser.parse_args()

    files = [f for f in os.listdir(args.gt_dir) if f.endswith('.npy')]
    for fname in files:
        gt = np.load(os.path.join(args.gt_dir, fname))
        pred = np.load(os.path.join(args.pred_dir, fname))
        dsc = dice_score(gt, pred, args.class_id)
        hd  = hausdorff_distance(gt, pred, args.class_id)
        vd, vp = volume_difference(gt, pred, args.class_id)
        print(f"{fname}: DSC={dsc:.3f}, HD={hd:.3f}, ΔVol={vd} vox ({vp*100:.1f}%)")

if __name__ == '__main__':
    main()
