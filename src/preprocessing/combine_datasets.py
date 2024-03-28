#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd


# load taft_weber_cell2022 data & combine with syn-coev data
# match column headers as well as mab terminology


cell_paper_data = pd.read_csv('data_preprocessing/Taft_Weber_Cell2022_FullData.csv')

preproc_path = 'data_preprocessing/'
pnp_data = pd.read_csv(preproc_path + 'rbd_data_preprocessed.csv')

#append an L to beginning of class 1 sequences to match PnP dataset
rbd1 = cell_paper_data[cell_paper_data['Library'] == 'RBM1']
rbd1['sequence_aa']  = 'L' + rbd1['sequence_aa']
cell_paper_data = cell_paper_data.drop(rbd1.index)
cell_paper_data = pd.concat([cell_paper_data, rbd1], ignore_index = True)

cell_paper_data['Library'] = cell_paper_data['Library'].replace({'RBM1': 'RBD_Class_1', 'RBM2': 'RBD_Class_2', 
                                       'RBM3': 'RBD_Class_3'})

cell_paper_data['Antibody'] = cell_paper_data['Antibody'].replace({'nCOV50': '50wt', 'REGN33': '33wt','LY555': '555wt',
                                          'LY555': '555wt', 'nCOV64': '64wt' , 'LY16': '16wt' , 'nCOV82': '82wt' ,
                                          'REGN87': '87wt' })

cell_paper_data = cell_paper_data.rename(columns = {'Library': 'rbd_class', 'Antibody': 'mab', 'sequence_aa': 'aa_seq',
                          'Label': 'binding'})

pnp_data = pnp_data.drop(columns = ['variant', 'consensus_count', 'origin_file', 'index', 'library'])
cell_paper_data = cell_paper_data.drop(columns = ['Unnamed: 0', 'Distance'])

cell_paper_data['binding'] = cell_paper_data['binding'].astype(int)

combined = pd.concat([pnp_data,cell_paper_data], ignore_index = True).drop_duplicates(subset = ['aa_seq', 'rbd_class', 'mab', 'binding'])


combined.to_csv('../data/pnp_and_cell_data.gz', header=True,index=False)
