#!/bin/bash
torchrun --nnodes=1 --nproc_per_node=6 --master_port=22046 train.py --bsz 20 --nepoch 250 --feature_extractor_path '../resnet50.pth' --backbone 'resnet50' --resume '' --lr 1e-4 --benchmark 'fss' --datapath '../../data/FSS-1000' --num_queries 100  --dec_layers 1 --model_name MOD-RES50 --fold 0 --test_num 1000 --test_datapath '../../data/Dataset' --neptune

