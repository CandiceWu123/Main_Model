#!/bin/bash
NUMS=(20)
DATASETS=("Animal" "Aerial" "Artificial_Luna_Landscape" "Crack_Detection" "Magnetic_tile_surface" "Eyeballs" "Leaf_Disease" "ClinicDB")
for DATASET in "${DATASETS[@]}"
do
  for NUM in "${NUMS[@]}"
  do
    tmux kill-session -t "$DATASET$NUM"
    # tmux new-session -d -s "$DATASET$NUM"
    # tmux send-keys -t "$DATASET$NUM" "conda activate pytorch" C-m
    # tmux send-keys -t "$DATASET$NUM" "python test.py --test_dataset $DATASET --load './logs/train/fold_0_1101_182144/80best_model.pt' --bsz 1 --test_epoch 2 --test_num 1000 --backbone 'swin-l' --feature_extractor_path '../swin-l.pth' --nshot $NUM --model_name swin_test --num_queries 15 --dec_layer 3 --average" C-m
  done
done
