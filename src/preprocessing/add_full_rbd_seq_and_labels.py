#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

'''
 Input: Extracted substring RBD mutant sequences containing only mutated variable regions
 Output: Full RBD sequences of length 201 amino acids
'''


def combine_rbd_classes_to_single_string(class1_df, class2_df, class3_df):
    
    data1 = class1_df.copy()
    data2 = class2_df.copy()
    data3 = class3_df.copy()
    
    class1_wt = 'LYRLFRKSNLKPFERDISTEIYQAGST'
    class2_wt = 'KNEGFNCYFPLQSYGFQPTNGVGY'
    class3_wt = 'NNLDSKVGGNYNYL'
    
    full_rbd_wt = 'NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKST'

    #class 1 processing
    data1['degen_region'] = data1['aa_seq'].copy()
    data1['aa_seq'] = full_rbd_wt[0:121] + data1['degen_region'] +  full_rbd_wt[148:]
    data1 = data1.drop(columns = ['degen_region'])
    
    #class 2 processing
    data2['417aa'] = data2['aa_seq'].str[0].copy()
    data2['439aa'] = data2['aa_seq'].str[1].copy()
    data2['484-505'] = data2['aa_seq'].str[2:].copy()
    data2['aa_seq'] = full_rbd_wt[0:86] + data2['417aa'] + full_rbd_wt[87:108] + data2['439aa'] + full_rbd_wt[109:153] + data2['484-505'] + full_rbd_wt[175:]
    data2 = data2.drop(columns = ['417aa', '439aa', '484-505'])
    
    #class 3 processing
    data3['degen_region'] = data3['aa_seq'].copy()
    data3['aa_seq'] = full_rbd_wt[0:108] + data3['degen_region'] +  full_rbd_wt[122:]
    data3 = data3.drop(columns = ['degen_region'])
    
    combined_data = pd.concat([data1,data2,data3], ignore_index = True)
    
    return combined_data   

def label_distr_per_class(data_frame):
    
    #create string containing mab names & labels for a given rbd sequence
    data_frame = data_frame.copy()
    data_frame['mab+bind'] = data_frame['mab'] + data_frame['binding'].astype(str)
    data = data_frame.groupby(['aa_seq', 'rbd_class'],as_index = False).agg(mabs = ('mab+bind','unique'))
    data['num_labels'] = data['mabs'].apply(lambda x: len(x)) #count number of labels per RBD seq
    data['label_set'] = data['mabs'].apply(lambda x: set(x)) #create new column of mab labels as set for faster processing
    data['mabs_joined'] = data['mabs'].apply(lambda x: '_'.join(x))
    categories = pd.Categorical(data['mabs_joined'])
    data['label_cat'] = categories.codes 
    return data

    
def get_label_vector(data_frame):
    
    pnp_mabs = ['2C08', '87', '555', '33', '50', '16', '64', '82']
    variant_list = ['wt','a','b','d','O']
    mab_list = []
    for i in range(len(pnp_mabs)):
        for j in range(len(variant_list)):
            mab_list.append(pnp_mabs[i] + variant_list[j])
    
    #remove mabs without data
    no_data_labels =  ['87O', '555b', '555O', '16b', '82d', '16O', '50O', '64O']
    mab_list = [x for x in mab_list if x not in no_data_labels]
    
    #add cell study mabs and ACE2 to list
    cell_mabs = ['A23581', 'GR32A4', 'LY1404','S2E12', 'S2H97', 'S309', 'ACE2']
    
    for mab in cell_mabs:
        mab_list.append(mab)
    
    mab_set_dict = {} #create dictionary of sets containing binding / nonbinding labels for each mab of interest
    for mab in mab_list:
        mab_set_dict[mab] = set([mab + '1', mab + '0'])
    
    df = data_frame.copy()
    label_df = pd.DataFrame()

    #for each RBD sequence in data_frame, identify binding label for mab of interest & append '-1' if no label exists
    for mab, mab_set in mab_set_dict.items():     
        
        tmp = df['label_set'].apply(lambda x: mab+'-1'  if not bool(x & mab_set) else x)
        tmp = tmp.apply(lambda x: ''.join(x)) #convert single element sets to stirngs
        label_df[mab] = tmp.str.extract(mab+'(-?[0,1])').astype(str)
        label_df[mab] = label_df[mab].apply(pd.to_numeric)
    
    return pd.concat([df,label_df], axis = 1)
    

if __name__ == '__main__':
    PATH = '../data/' 
    
    data_ = pd.read_csv(PATH + 'pnp_and_cell_data.gz')
    
    data1 = data_[data_['rbd_class'] == 'RBD_Class_1']
    data2 = data_[data_['rbd_class'] == 'RBD_Class_2']
    data3 = data_[data_['rbd_class'] == 'RBD_Class_3']    
    
    print('processing aa sequences to include full RBD..')
    #assemble full length rbd sequences from class-based substrings
    data = combine_rbd_classes_to_single_string(class1_df = data1, 
                                                class2_df = data2, 
                                                class3_df = data3
                                                )
    
    del data_
    del data1
    del data2
    del data3
    
    print('getting label set for mabs')
    #get label set for each mab
    data  = label_distr_per_class(data_frame = data)
    
    print('formatting labels to data frame')
    #parse label set to creat df with one mab per column containing labels
    data = get_label_vector(data_frame = data)
    
    print('correcting for ace2')
    #convert unk ace2 labels to binders
    #as all non-labeled squences were pre-screened for binding to ace2
    data['ACE2'] = np.where(data['ACE2'] == -1, 1, data['ACE2'])
    
    data = data.drop_duplicates(subset = ['aa_seq'])
    
    data.to_csv('../data/full_data_pre_split.gz', index=False, header = True, 
                compression ={'method': 'gzip', 'compresslevel': 1, 'mtime': 1})