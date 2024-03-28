#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns

# load results
path = '../results'
results = '/full_results.csv'
data = pd.read_csv(path + results)

# rename columns 
data['basemodel'] = data['basemodel'].replace({'rbd_plm': 'RBD-pLM', 
                                               'transformer': 'Transformer',
                                               'transformer2': 'Transformer2',
                                               'cnn': 'CNN',
                                               'logistic_regression': 'Logistic Regression',
                                               'mlp': 'MLP',
                                               'rbd_plm_lr': 'pLM Only'})

data = data.rename(columns={'mcc_test':'Full', 
                            'mcc_head_test':'Head', 
                            'mcc_mid_test': 'Mid', 
                            'mcc_tail_test': 'Tail',
                            'taft_mcc': 'Synthetic',
                            'he_mcc': 'Natural',
                            'dataset': 'Dataset',
                            'basemodel': 'Model',
                            })


data['Dataset'] = data['Dataset'].replace({'ed_10': 'ED 10',
                                           'ed_3': 'ED 3',
                                           'ed_10': 'ED 10', 
                                           'main': 'Full Data', 
                                           })

# melt dataframe

data =  pd.melt(data, 
                id_vars = ['Model','Dataset', 'seed',], 
                value_vars = ['Full', 'Head', 'Mid', 'Tail', 'Synthetic', 'Natural']
                )


data.variable = data.variable.astype('category')
data.variable = data.variable.astype(str)
sns.set_theme(rc={'figure.figsize':(10,7)}, style="whitegrid")
plt.figure()

col= 'Dataset'
data = data.fillna(0)


# plot
results = sns.catplot(data=data, kind = 'bar', x="variable", y="value",
                      palette='YlGnBu',
                      legend = 'full',  
                      hue = 'Model',# hue='rbd_plm_backbone', 
                      errorbar=('sd'),
                      hue_order = ['RBD-pLM', 'Transformer', 'pLM Only', 'CNN', 'Logistic Regression'],
                      col = col, col_order = ['ED 3','ED 10', 'Full Data'],)#col = 'loss_fn', 

(results.tight_layout(w_pad = 0)
 .set_xticklabels(data['variable'].unique())#,rotation=45)
 .set_axis_labels("Metric", "MCC")
 .set_titles("Task: {col_name}"))

# add numbers to barplot
# extract the matplotlib axes_subplot objects from the FacetGrid
for j in range(results.axes.shape[0]):
    for i in range(results.axes.shape[1]):
        ax = results.axes[j].flat[i]
        
        # iterate through the axes containers
        for c in ax.containers:
            labels = [f'{round(v.get_height(),2)}' for v in c]
            #ax.bar_label(c, labels=labels, label_type='center')
            ax.bar_label(c, labels = labels, 
                         padding=5, fmt='%.2f', label_type='edge', fontsize=9, 
                         rotation='vertical')

# save fig    
plt.subplots_adjust(wspace=0.1)
fig_name_str = f'performance_results'
plt.savefig(f'{fig_name_str}.png', dpi=300,  bbox_inches = "tight")



