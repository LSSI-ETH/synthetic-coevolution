#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from data_helpers.general_data_utils import *
import logging

#===========================   Get Dataset              ================================ 

def add_label_vector(data_frame, args, df_id = 'train'):
    
    label_cols = data_frame.columns[7:]
    label_cols = list(label_cols)
    label_df= pd.DataFrame(data = data_frame, 
                           columns = [label_cols[i] for i in range(len(label_cols))])
   
    label_df['label_vector'] = label_df.values.tolist()
    data_frame['label_vector'] = label_df['label_vector'].copy()
    return data_frame


def get_xs_and_ys_main_data(args):
    
    """
    Loads dataset from file and returns x's and y's to be encoded
    
    Parameters
    ----------
    argparse args

    Returns
    -------
    x: pandas.core.series.Series
        RBD sequences (strings) 
    y: pandas.core.series.Series
        Series containing label vectors, each row with n labels: [y1, y2, ..., yn]  
        label values for each y1, ..., yn are either 1, 0, or -1
        where -1 corresponds to unkown
    """    
    
    train = pd.read_csv(f'{args.train_path}/train.gz')
    val = pd.read_csv(f'{args.train_path}/val.gz')
    test = pd.read_csv(f'{args.train_path}/test.gz')
    
    train = add_label_vector(data_frame = train, args = args)
    val = add_label_vector(data_frame = val, args = args)
    test = add_label_vector(data_frame = test, args = args)
    
    y_val = val['label_vector']
    y_test = test['label_vector']
    
    logging.info(f'Train, Val, Test: {len(train)}, {len(val)}, {len(test)}')

    if 'esm' not in args.dataset:
        feature_col = 'aa_seq'
        
    elif 'esm' in args.dataset:
        feature_col = 'embedding'
        
    x_train = train[feature_col]
    x_val = val[feature_col]
    x_test = test[feature_col]
    
    y_train = train['label_vector']
    y_test = test['label_vector']

    #----------------------------------------------------------------------------
    # processing for DB Loss label balanced sampling
    freq = train.copy()
    label_cols = freq.columns[7:-1]
    label_cols = list(label_cols)
    
    train_class_freq = {col: freq[col].value_counts()[1] for col in label_cols}
    train_class_freq = list(train_class_freq.values())
    
    logging.info(f'len(x_train) = {len(x_train)}')
    logging.info(f'len(x_val) = {len(x_val)}')
    logging.info(f'len(x_test) = {len(x_test)}')
    
    # cleanup for large datasets
    del freq, train, val, test 

    return x_train, x_val, x_test, y_train, y_val, y_test, train_class_freq


#============== Batch Datasets ===================
def batch_datasets_main(args):
    
    x_train, x_val, x_test, y_train, y_val, y_test, class_freq = get_xs_and_ys_main_data(args)
    
    if 'esm' not in args.dataset and 'esm' not in args.rbd_plm_backbone:
        
        x_train = encode_ngrams(x_train, args)
        x_val = encode_ngrams(x_val, args)
        x_test = encode_ngrams(x_test, args)
        
    elif 'esm' in args.dataset or 'esm' in args.rbd_plm_backbone:
        
        x_train = x_train.to_list()
        x_val = x_val.to_list()
        x_test = x_test.to_list()
    
    train_loader, train_sampler = data_to_loader(x = x_train, y=y_train, 
                                  is_test = False, args = args)
    
    val_loader, _ = data_to_loader(x = x_val, y=y_val, 
                                is_test = False, args = args)
    
    test_loader, _ = data_to_loader(x = x_test, y=y_test, 
                                 is_test = True, args = args)
    
    return train_loader, val_loader, test_loader, class_freq, train_sampler 