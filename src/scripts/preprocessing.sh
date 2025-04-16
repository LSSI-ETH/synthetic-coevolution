#!/bin/bash

# path to preprocessing scripts
preprocessing_path="synthetic-coevolution/src/preprocessing/"

cd $preprocessing_path

python train_test_split.py
python train_test_split_edit_distance.py
python train_test_split_holdout_positions.py