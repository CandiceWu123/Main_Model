#!/bin/bash
CUDA_VISIBLE_DEVICES=7 python test.py --test_dataset Leaf_Disease --nshot 1 --vote --load './logs/train/fold_0_1115_120325/best_model.pt' --num_queries 15 --dec_layer 3 --backbone 'resnet101' --feature_extractor_path '../resnet101.pth'  --model_name resnet101_test   --test_datapath '../../data/Dataset'


