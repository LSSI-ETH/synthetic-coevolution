#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Split preprocessed data into train, val, test sets
using package iterstrat for multi-label data set splitting

#https://github.com/trent-b/iterative-stratification
#https://pypi.org/project/iterative-stratification/

'''

import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from Levenshtein import distance as levenshtein_distance


#===========================   Consensus Sequence & ED ============================== 
def consensus_seq_and_LD_to_df(data_frame, seq_col_str = 'aa_seq', return_df = True):
    
    #============================================================================== 
    #======== Calculate residue-position frequencies & store in seq_count_df ======
    #============================================================================== 
    sequences = pd.Series(data_frame[seq_col_str]).to_list()
       
    max_len = max(map(len, sequences))
    seq_count_dict = defaultdict(lambda: [0]*max_len)  # d[char] = [pos0, pos12, ...]
    for seq in sequences:
        for i, char in enumerate(seq): 
            seq_count_dict[char][i] += 1
       
    seq_count_df = pd.DataFrame.from_dict(seq_count_dict)
    seq_count_df = seq_count_df.T
    seq_count_df.columns = [str(i) for i in range(len(sequences[0]))]
    consensus_seq = list(seq_count_df.idxmax(axis=0))
    consensus_seq = ''.join(consensus_seq)
    #============================================================================== 
    #============================================================================== 
    data_frame = add_LD_to_df(data_frame, seq_col_str = seq_col_str, consensus_sequence = consensus_seq)
    
    if return_df == True:    return data_frame, consensus_seq
    else: return consensus_seq


def add_LD_to_df(data_frame, seq_col_str, consensus_sequence):
    """
    calculates edit distance from consensus sequence for each amino acid sequence in dataframe. returns 
    dataframe with appended LD column

    Parameters
    ----------
    data_frame : pandas.Dataframe
        dataframe containing amino acid sequences
    seq_col_str : str
        string of column ID containing sequences in dataframe    
    consensus_sequence : str
        string containing consensus sequence of full data set.

    Returns
    -------
    data_frame_out : pandas.DataFrame
        data_frame with the column LD appended to it, containing the edit distance for each sequence from the consensus sequence.

    """

    wt_str = consensus_sequence
    
    LD_arr = []
    for i in range(len(data_frame)):
        LD_arr.append( levenshtein_distance(wt_str, 
                                            data_frame[seq_col_str].iloc[i]) )
    data_frame_out = data_frame.copy()
    data_frame_out['LD'] = LD_arr
    
    return data_frame_out






PATH = '../data' 


data_frame = pd.read_csv(PATH + '/full_data_pre_split.gz')

data_frame, wild_type = consensus_seq_and_LD_to_df(data_frame)


label_cols = data_frame.columns[7:-1]

#ensure ace2 escapes are represented
ace2_negs = data_frame[data_frame['ACE2'] == 0].copy()
ace2_negs = ace2_negs[ace2_negs['num_labels'] == 1]

#remove ace2 non-binders
data_frame = data_frame[data_frame['num_labels'] > 1]
data_frame = data_frame[data_frame['ACE2'] == 1]

df = pd.DataFrame()
for mab in ['LY1404', 'S309', '87wt', '555wt','33wt']:
    tmp_df = data_frame[data_frame[mab].isin([0,1])]
    df = pd.concat([df,tmp_df], ignore_index = True)
    
data_frame = df.copy()
data_frame = data_frame.drop_duplicates(subset = ['aa_seq']).sample(frac = 1, 
                                                                random_state = 42)

label_df= pd.DataFrame(data = data_frame, 
                       columns = [label_cols[i] for i in range(len(label_cols))])

#switch 1 to 0 and 0 to 1
m0 = label_df == 0
m1 = label_df == 1

label_df = label_df.where(~m0,1)
label_df = label_df.where(~m1,0)
data_frame[label_cols] = label_df

# flip neg ace2 label accordingly
ace2_negs['ACE2'] = 1

#drop sequence presnet in external test sets to prevent data leakage
excluded_testset1 = pd.read_csv('../data/rbd_test_seqs_voc.csv')
excluded_testset2 = pd.read_csv('../data/he_paper_voc.csv')
excluded_testset3 = pd.read_csv('../data/taft_weber_vocs.csv')
data_frame = data_frame[~data_frame.aa_seq.isin(excluded_testset1.aa_seq)]
data_frame = data_frame[~data_frame.aa_seq.isin(excluded_testset2.aa_seq)]
data_frame = data_frame[~data_frame.aa_seq.isin(excluded_testset3.aa_seq)]

data_frame = data_frame.drop_duplicates(subset = ['aa_seq'], keep = 'first')

holdout = [86, 108, 111, 112, 116, 118, 120, 123, 124, 129, 130, 144, 146, 154, 155, 160, 171, 175] #train: 84771, test: 264567

def train_test_split_by_subsets(df, wild_type, positions_to_avoid):
    sequence_list = df.aa_seq.to_list()

    seqs_for_test_set = []
    for i in range(len(sequence_list)):
        wt_mismatch_count = 0
        entry = sequence_list[i]
        
        for j in range(len(positions_to_avoid)):
            eval_idx = positions_to_avoid[j]
            if entry[eval_idx] !=  wild_type[eval_idx]:
                wt_mismatch_count += 1
        
        if wt_mismatch_count == 0:
            seqs_for_test_set.append(entry)
    
    train = data_frame[~data_frame['aa_seq'].isin(seqs_for_test_set)]
    test = data_frame[data_frame['aa_seq'].isin(seqs_for_test_set)]
    
    return train, test

train, test = train_test_split_by_subsets(data_frame, wild_type, holdout)

test.to_csv(f'{PATH}/holdout_test.gz', index = False, header = True)


def create_train_val_sets(train):
    #==================== Split into Train/Val/Test ================================
    x = train['aa_seq']
    y = train[label_cols]
    
    x = x.to_numpy()
    y = y.to_numpy()
    
    msss = MultilabelStratifiedShuffleSplit(n_splits=2, 
                                            test_size=0.3, random_state=42)
    
    for train_index, test_index in msss.split(x, y):
        x_train, x_val = x[train_index], x[test_index]
        y_train, y_val = y[train_index], y[test_index]
    
    # make some memory space
    del x
    del y
    
    val = train[train['aa_seq'].isin(x_val)]
    train = train[train['aa_seq'].isin(x_train)]
    return train, val

train, val = create_train_val_sets(train)

train.to_csv(f'{PATH}/holdout_train.gz', index = False, header = True)
val.to_csv(f'{PATH}/holdout_val.gz', index = False, header = True)
