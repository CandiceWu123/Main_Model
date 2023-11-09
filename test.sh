#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python test.py --test_dataset ClinicDB --nshot 20 --load './best_model/resnet50/best_model.pt' --bsz 1 --test_epoch 2 --test_num 1000 --backbone 'resnet50' --feature_extractor_path '../resnet50.pth'  --model_name swin_test --num_queries 15 --dec_layer 1 --vote --test_datapath '../../Dataset'

