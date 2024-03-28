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
from collections import defaultdict
from Levenshtein import distance as levenshtein_distance
from sklearn.model_selection import train_test_split

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

#ensure ace2 escapes are represented
label_cols = data_frame.columns[7:]
data_frame, _ = consensus_seq_and_LD_to_df(data_frame)

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


#==================== Split into Train/Val/Test ================================
x = data_frame['aa_seq']
y = data_frame[label_cols]

x = x.to_numpy()
y = y.to_numpy()

msss = MultilabelStratifiedShuffleSplit(n_splits=2, 
                                        test_size=0.3, random_state=42)

for train_index, test_index in msss.split(x, y):
    x_train, x_test_temp = x[train_index], x[test_index]
    y_train, y_test_temp = y[train_index], y[test_index]

# make some memory space
del x
del y

# split to test / validation
msss = MultilabelStratifiedShuffleSplit(n_splits=2, 
                                        test_size=0.5, random_state=42)

for test_index, val_index in msss.split(x_test_temp, y_test_temp):
    x_test, x_val = x_test_temp[test_index], x_test_temp[val_index]
    y_test, y_val = y_test_temp[test_index], y_test_temp[val_index]

train = data_frame[data_frame['aa_seq'].isin(x_train)]
val = data_frame[data_frame['aa_seq'].isin(x_val)]
test = data_frame[data_frame['aa_seq'].isin(x_test)]

#remove single value high edit distance sequences for ease of splitting
ace2_negs = ace2_negs[ace2_negs['LD'] < 22]
num_negs = 25000

#sample LD distribution to mirror that of full negative set
ace2_negs_val_test, ace2_negs_train, = train_test_split(ace2_negs, test_size = num_negs,
                                       random_state = 1, shuffle = True, stratify = ace2_negs['LD'])

    
ace2_negs_test, ace2_negs_val, = train_test_split(ace2_negs_val_test, test_size = int(0.2* num_negs),
                                       random_state = 1, shuffle = True, stratify = ace2_negs_val_test['LD'])

_, ace2_negs_test, = train_test_split(ace2_negs_test, test_size = num_negs,
                                       random_state = 1, shuffle = True, stratify = ace2_negs_test['LD'])


train = pd.concat([train,ace2_negs_train],ignore_index=True).sample(frac = 1, random_state = 42)
train = pd.concat([train,ace2_negs_val],ignore_index=True).sample(frac = 1, random_state = 42)
test = pd.concat([test,ace2_negs_test],ignore_index=True).sample(frac = 1, random_state = 42)

train = train.drop_duplicates(subset = ['aa_seq'], keep = False)
val = val.drop_duplicates(subset = ['aa_seq'], keep = False)
test = test.drop_duplicates(subset = ['aa_seq'], keep = False)

train = train.drop(columns = ['LD'])
val = val.drop(columns = ['LD'])
test = test.drop(columns = ['LD'])

train.to_csv(f'{PATH}/train_24.gz', index = False, header = True)
val.to_csv(f'{PATH}/val_24.gz', index = False, header = True)
test.to_csv(f'{PATH}/test_24.gz', index = False, header = True)
