#!/bin/bash

path="synthetic-coevolution/src/"
cd $path

mkdir model_weights
mkdir results
mkdir runs
mkdir log_filess

python main.py --dataset=ed_3 --basemodel=rbd_plm --rbd_plm_backbone=esm_8m --combined_attn_layers=1 --inter_attn_layers=1 
python main.py --dataset=ed_10 --basemodel=rbd_plm --rbd_plm_backbone=esm_8m --combined_attn_layers=1 --inter_attn_layers=3 
python main.py --dataset=main --basemodel=rbd_plm --rbd_plm_backbone=esm_8m --combined_attn_layers=1 --inter_attn_layers=3

# plm only
python main.py --dataset=ed_3 --basemodel=rbd_plm_lr --rbd_plm_backbone=esm_8m
python main.py --dataset=ed_10 --basemodel=rbd_plm_lr --rbd_plm_backbone=esm_8m
python main.py --dataset=main --basemodel=rbd_plm_lr --rbd_plm_backbone=esm_8m

# transformer 
python main.py --dataset=ed_3 --basemodel=transformer 
python main.py --dataset=ed_10 --basemodel=transformer
python main.py --dataset=main --basemodel=transformer

# logistic regression
python main.py --dataset=ed_3 --basemodel=logistic_regression 
python main.py --dataset=ed_10 --basemodel=logistic_regression
python main.py --dataset=main --basemodel=logistic_regression

# cnn
python main.py --dataset=ed_3 --basemodel=cnn 
python main.py --dataset=ed_10 --basemodel=cnn
python main.py --dataset=main --basemodel=cnn