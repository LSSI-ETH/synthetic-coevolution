#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import glob
import warnings
from os import system, name
import os
import sys

warnings.filterwarnings("ignore")

PATH = 'preprocessing/data_preprocessing/' 
PATH_CSV = 'data_preprocessing/'

def get_bind(bind):
    if bind == 'B':
        return 1
    elif bind == 'E':
        return 0
    else:
        raise Exception('Uknown label found',bind)

def get_variant(mab):
    if 'wt' in mab:
        return 'wt'
    elif 'a' in mab:
        return 'a'
    elif 'b' in mab:
        return 'b'
    elif 'd' in mab:
        return 'd'
    elif 'O' in mab:
        return 'O'
    else:
        if 'ACE2' in mab:
            return 'ACE2'
        else:
            raise Exception('Uknown variant found',mab)
        


if __name__ == '__main__':

    print('data_fetch.py')

    filenames = glob.glob(PATH_CSV + '*.csv') 
    data = pd.DataFrame()
    filenames = sorted(filenames)
    data_raw_df = pd.DataFrame(columns = [ 'aa_seq', 'variant' , 'rbd_class' , 'library' , 'mab' , 'binding'  , 'consensus_count' ,'origin_file' ])
    datasets_metadata_df = pd.DataFrame(columns = [ 'file' , 'data_points'])
    i = 1
    
    count_threshold = 2
    
    for file in filenames:
        
        print(i ,' / 192 files processed', end="\r")
        
        file_df = pd.read_csv(file)
        file_name = file.split('/')[-1]
        file_ls = file_name.split('_')
        temp = pd.DataFrame(columns = data_raw_df.columns )
        temp['aa_seq'] = file_df['junction_aa']
        temp['rbd_class'] = file_df['v_call']
        temp['consensus_count'] = file_df['consensus_count']
        temp['library'] = file_ls[0]
        temp['mab'] = file_ls[1]
        temp['binding'] = get_bind(file_ls[2])
        temp['variant'] = get_variant(file_ls[1])
        temp['origin_file'] = file_name
        temp = temp[temp['consensus_count'] >= count_threshold]
        data_raw_df = data_raw_df.append(temp)
        meta_df = pd.DataFrame(columns = datasets_metadata_df.columns )
        meta_df.loc[len(meta_df.index)] = [file_name,temp.shape[0]]
        
        datasets_metadata_df = datasets_metadata_df.append(meta_df)
        
        i += 1
    
    print('\n')
    print(data_raw_df.shape)
    data_raw_df.to_csv(PATH + 'rbd_data_raw.csv', index=False)
    print(datasets_metadata_df.shape)
    datasets_metadata_df.to_csv(PATH + 'datasets_metadata.csv', index=False)
