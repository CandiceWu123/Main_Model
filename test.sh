#!/bin/bash
CUDA_VISIBLE_DEVICES=7 python test.py --test_dataset Artificial_Luna_Landscape --nshot 1 --average --load './logs/train/fold_0_1111_093807/17best_model.pt' --num_queries 50 --dec_layer 1 --backbone 'resnet50' --feature_extractor_path '../resnet50.pth'  --model_name resnet50_test   --test_datapath '../../data/Dataset'

