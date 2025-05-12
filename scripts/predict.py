#!/usr/bin/env python
"""
CLI: run inference on new volumes.
"""
import argparse
from placenta_segmentation.inference.predictor import predict_volume

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Trained weights (.h5)')
    parser.add_argument('--input', required=True, help='Input .npy volume')
    parser.add_argument('--output', required=True, help='Output segmentation .npy')
    args = parser.parse_args()
    predict_volume(args.model, args.input, args.output)

if __name__ == '__main__':
    main()
