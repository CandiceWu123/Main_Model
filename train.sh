#!/bin/bash
torchrun --nnodes=1 --nproc_per_node=6 --master_port=22055 train.py --bsz 12 --nepoch 250 --feature_extractor_path '../resnet101.pth' --backbone 'resnet101' --resume '' --lr 1e-4 --benchmark 'fss' --datapath '../../data/FSS-1000' --num_queries 50  --dec_layers 1 --model_name resnet101 --fold 0 --test_num 1000 --test_datapath '../../data/Dataset' 

