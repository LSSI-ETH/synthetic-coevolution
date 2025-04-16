#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from data_helpers.general_data_utils import *
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import logging


#===========================   Get Dataset              ================================ 

def add_label_vector(data_frame, args, df_id = 'train'):
    
    label_cols = data_frame.columns[7:-1]
    label_cols = list(label_cols)
    label_df= pd.DataFrame(data = data_frame, 
                           columns = [label_cols[i] for i in range(len(label_cols))])
   
    label_df['label_vector'] = label_df.values.tolist()
    data_frame['label_vector'] = label_df['label_vector'].copy()
    return data_frame


def get_xs_and_ys_holdout_positions(args):
    
    """
    Loads dataset from file and returns x's and y's to be encoded
    
    Parameters
    ----------
    argparse args

    Returns
    -------
    x: pandas.core.series.Seriesa
        RBD sequences (strings) 
    y: pandas.core.series.Series
        Series containing label vectors, each row with n labels: [y1, y2, ..., yn]  
    """    
    
    print(f'Loading {args.dataset} dataset', flush = True)
    
    train = pd.read_csv(f'{args.train_path}/{args.dataset}_train.gz')
    val = pd.read_csv(f'{args.train_path}/{args.dataset}_val.gz')
    test = pd.read_csv(f'{args.train_path}/{args.dataset}_test.gz')
    
    print(f'Dataset Loaded: {args.dataset}', flush = True)

    print(f'Adding Label Vector', flush = True)
    train = add_label_vector(data_frame = train, args = args)
    val = add_label_vector(data_frame = val, args = args)
    test = add_label_vector(data_frame = test, args = args)
    
    logging.info(f'Train, Val, Test: {len(train)}, {len(val)}, {len(test)}')

    feature_col = 'aa_seq'
    
    x_train = train[feature_col]
    x_val = val[feature_col]
    x_test = test[feature_col]
    
    y_train = train['label_vector']
    y_val = val['label_vector']
    y_test = test['label_vector']

    #----------------------------------------------------------------------------
    #processing for DB Loss label balanced sampling
    
    print('Processing for DB Loss label balanced sampling', flush = True)
    freq = train.copy()
    label_cols = freq.columns[7:-2]
    label_cols = list(label_cols)
    
    
    # account for mabs not present in training set
    train_class_freq = {}
    for col in label_cols:
        try:
            train_class_freq[col] = freq[col].value_counts()[1]
        except:
            train_class_freq[col] = 1
    
        #train_class_freq = {col: freq[col].value_counts()[1] for col in label_cols}
    train_class_freq = list(train_class_freq.values())

    print('Train Class Completed', flush = True)
    #----------------------------------------------------------------------------
    
    logging.info(f'len(x_train) = {len(x_train)}')
    logging.info(f'len(x_val) = {len(x_val)}')
    logging.info(f'len(x_test) = {len(x_test)}')

    #cleanup for large datasets
    del freq, train, val, test 

    return x_train, x_val, x_test, y_train, y_val, y_test, train_class_freq






#============== Batch Datasets ===================
def batch_datasets_holdout_positions(args):
    
    x_train, x_val, x_test, y_train, y_val, y_test, class_freq = get_xs_and_ys_holdout_positions(args)
    
    if 'esm' not in args.rbd_plm_backbone:
        
        x_train = encode_ngrams(x_train, args)
        x_val = encode_ngrams(x_val, args)
        x_test = encode_ngrams(x_test, args)
        
    elif 'esm' in args.rbd_plm_backbone:
        
        x_train = x_train.to_list()
        x_val = x_val.to_list()
        x_test = x_test.to_list()
   
    train_dataset = x_y_to_dataset(x_train, y_train, is_test = False, args = args)
    val_dataset = x_y_to_dataset(x_val, y_val, is_test = False, args = args)
    test_dataset = x_y_to_dataset(x_test, y_test, is_test = True, args = args)

    return train_dataset, val_dataset, test_dataset, class_freq
    