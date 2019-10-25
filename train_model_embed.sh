#! /bin/bash


STEPS=500
LR=0.001
BATCH_SIZE=128
DATA_DIR="/home/kiran/radio_modulation/pytorch/embedding"


#Euclidean optimizer
python3 run_net.py --train --data_dir $DATA_DIR --steps $STEPS --model_path 'model_E3.pt' --batch_size $BATCH_SIZE --learning_rate $LR --classes 3 --optim 0

#Hyperbolic 
python3 run_net.py --train --data_dir $DATA_DIR --steps $STEPS --model_path 'model_H3.pt' --batch_size $BATCH_SIZE --learning_rate $LR --classes 3 --optim 1
