#!/bin/bash

# input arguments
DATA="${1-Hetero}"
device=${1-1}
num_trials=${1-1}
print_every=${1-1}

# general settings
hidden_final=128 
dropout=0 
cross_weight=1.0
fuse_weight=0.9
weight_decay=1e-3

# dataset-specific settings
case ${DATA} in
IFTTT)
  hidden_gxn=196
  graph_type="homogeneous"
  num_epochs=100 # 1000 or larger for contrastive learning 
  learning_rate=0.0001
  sortpooling_k=31
  batch_size=64
  k1=0.8
  k2=0.5
  split_ratio=0.8
  mode="classification"
  ;;
SMT)
  hidden_gxn=196
  graph_type="homogeneous"
  num_epochs=500
  learning_rate=0.0001
  sortpooling_k=31
  batch_size=64
  k1=0.8
  k2=0.5
  split_ratio=0.8
  mode="classification"
  ;;
Hetero)
  hidden_gxn=128
  graph_type="heterogeneous"
  num_epochs=30
  learning_rate=0.0001
  sortpooling_k=32
  batch_size=8
  k1=0.7
  k2=0.5
  split_ratio=0.8
  mode="classification"
  ;;
esac


python main.py \
      --dataset $DATA \
      --lr $learning_rate \
      --epochs $num_epochs \
      --hidden_dim $hidden_gxn \
      --final_dense_hidden_dim $hidden_final \
      --readout_nodes $sortpooling_k \
      --pool_ratios $k1 $k2 \
      --batch_size $batch_size \
      --device $device \
      --dropout $dropout \
      --cross_weight $cross_weight\
      --fuse_weight $fuse_weight\
      --weight_decay $weight_decay\
      --num_trials $num_trials\
      --print_every $print_every\
      --graph_type $graph_type\
      --split_ratio $split_ratio\
      --mode $mode\
