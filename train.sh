#!/bin/bash
python -m torch.distributed.run --nnodes=1 --nproc_per_node=5 --master_port=22038 train.py --bsz 8 --nepoch 250 --feature_extractor_path '../resnet101.pth' --backbone 'resnet101' --resume '' --lr 1e-4 --benchmark 'fss' --datapath '../../FSS-1000' --num_queries 15  --dec_layers 1 --model_name MOD-RES50 --fold 0 --test_num 1000 --test_datapath '../../Dataset'

