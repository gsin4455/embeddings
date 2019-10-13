#! /bin/bash


STEPS=500
LR=0.002
BATCH_SIZE=128
DATA_DIR="/home/kiran/radio_modulation/pytorch/embedding"

#Euclidean optimizer
python3 run_net.py --train --data_dir $DATA_DIR --steps $STEPS --model_path model_E.pt --batch_size $BATCH_SIZE --learning_rate $LR --classes 5 --optim 0


#Hyperbolic 
python3 run_net.py --train --data_dir $DATA_DIR --steps $STEPS --model_path 'model_H.pt' --batch_size $BATCH_SIZE --learning_rate $LR --classes 5 --optim 1
