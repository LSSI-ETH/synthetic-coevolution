#!/bin/bash

# path to preprocessing scripts
preprocessing_path="synthetic-coevolution/src/preprocessing/"

cd $preprocessing_path

python data_fetch.py
python data_preprocess.py
python combine_datasets.py
python add_full_rbd_seq_and_labels.py
python train_test_split.py
python train_test_split_edit_distance.py