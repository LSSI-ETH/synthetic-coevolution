#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import data_fetch as df
import pandas as pd

def remove_duplicates(data):
    '''
    Function to remove duplicates and duplicates with arbitrary labels i.e. for the same (aa_seq,mab) pair 'binding' labels are different

    Parameters :

        data : pandas.Dataframe containing viriants and antibody binding information. Should include columns name ['aa_seg','mab','binding]

    Return :

        pandas.Dataframe : data without duplicate (sequences,antibody pairs) and (sequences,antibody pairs) that have arbitrary labels
    '''
    
    data['index'] = data.index
    mask = data.duplicated(subset=['aa_seq' , 'mab'],keep=False)
    doubles = data[mask].groupby(['aa_seq' , 'mab'],as_index = False).agg(unique_values = ("binding","unique"), index = ("index","unique"))
    indices = []
    count = 0
    ## Searching for the indexes of the (aa_seq, mab) pairs whose label is arbitrary 
    for d in doubles.values:   
        if d[2].shape[0] > 1:
            count += 1
            p = d[3].tolist()
            indices += p
    ## Removing elements with arbitrary labels
    data.drop(index=indices ,inplace = True)
    print('Removed sequences with arbitrary labels:',len(indices))
    ## Removing duplicate elements
    data.drop_duplicates(subset=['aa_seq','mab'],keep='first',inplace = True)
    return data

def remove_noisy_datapoints(data, threshold = 2):
    '''
    Function to remove entries with noisy seqeunces (include '*')

    Parameters :

        data : pandas.Dataframe containing the sequences to be cleaned
        threshold, int: read count threshold at which to remove sequences

    Return :

        pandas.Dataframe : data without sequences including '*'
    '''
    
    data.drop(index = data[data['aa_seq'].str.contains('\*')].index, inplace = True)

    return data


if __name__=='__main__':

    print('data_preprocess.py')
    data_raw_df = pd.read_csv(df.PATH + 'rbd_data_raw.csv')
    print(data_raw_df.shape[0])
    data_ = remove_duplicates(data_raw_df)
    print(data_.shape[0])
    data_ = remove_noisy_datapoints(data_)
    print(data_.shape[0])
    data_.to_csv(df.PATH + 'rbd_data_preprocessed.csv', index=False)
    PATH = 'preprocessing/data_preprocessing/' 
    data_ = pd.read_csv(PATH + 'rbd_data_preprocessed.csv')

    
    
    