#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load results
path = '../results'
results = '/full_results.csv'
data = pd.read_csv(path + results)
data = data[data['dataset'] == 'holdout']

# rename columns 
data['basemodel'] = data['basemodel'].replace({'rbd_plm': 'RBD-pLM', 
                                               'transformer': 'Transformer',
                                               'cnn': 'CNN',
                                               'logistic_regression': 'Logistic Regression',
                                               'rbd_plm_lr': 'pLM Only'})

data = data.rename(columns={'mcc_test':'Full', 
                            'mcc_head_test':'Head', 
                            'mcc_mid_test': 'Mid', 
                            'mcc_tail_test': 'Tail',
                            'taft_mcc': 'Synthetic',
                            'he_mcc': 'Natural',
                            'dataset': 'Dataset',
                            'basemodel': 'Model',
                            'holdout': 'Holdout',
                            })


data['Dataset'] = data['Dataset'].replace({'ed_10': 'ED 10',
                                           'ed_3': 'ED 3',
                                           'main': 'Full Data', 
                                           'holdout': 'Holdout',
                                           })

# melt dataframe
data =  pd.melt(data, 
                id_vars = ['Model','Dataset', 'seed','learn_rate', 'lr_scheduler'], 
                value_vars = ['Full', 'Head', 'Mid', 'Tail', ]
                )

data.variable = data.variable.astype('category')
data.variable = data.variable.astype(str)
sns.set_theme(rc={'figure.figsize':(10,7)}, style="whitegrid")
plt.figure()

col= 'Dataset'
data = data.fillna(0)

palette = sns.color_palette('colorblind')
palette[4] = (0.2, 0.2, 0.2)  # Dark gray for the last element


# plot
results = sns.catplot(data=data, kind = 'box', x="variable", y="value",
                      palette=palette,
                      legend = 'full',  
                      hue = 'Model',
                      errorbar=('sd'),
                      col = 'Dataset',
                      row = 'learn_rate',
                      hue_order = ['RBD-pLM', 'Transformer', 'pLM Only', 'CNN', 'Logistic Regression'],
                    )

(results.tight_layout(w_pad = 0)
 .set_xticklabels(data['variable'].unique())
 .set_axis_labels("Metric", "MCC")
 .set_titles("Task: Holdout Positions")
 .set(ylim=(0, None))
 )


# Extract the axes_subplot objects from the FacetGrid
for j in range(results.axes.shape[0]):
    for i in range(results.axes.shape[1]):
        ax = results.axes[j].flat[i]
        
        hue_order = ['RBD-pLM', 'Transformer', 'pLM Only', 'CNN', 'Logistic Regression']
        palette = sns.color_palette(palette, len(hue_order))
        
        # Add dodge adjustment based on the number of categories per group
        dodges = np.linspace(-0.3, 0.3, len(hue_order))
        
        # Filtering data for the current Dataset (col value)
        col_value = results.col_names[i]
        col_data = data[data['Dataset'] == col_value]
        
        # Iterate through the boxplot containers to annotate the mean
        for idx, model in enumerate(hue_order):
            model_data = col_data[col_data['Model'] == model]
            color = palette[idx]

            print(f'col_data: {col_data}')
            print(f'model_data: {model_data}')
            for variable in model_data['variable'].unique():
                variable_data = model_data[model_data['variable'] == variable]['value']
                
                # Calculate mean
                mean_val = variable_data.mean()
                
                # Find the x position for annotation
                x_pos = np.where(data['variable'].unique() == variable)[0][0] + dodges[idx]
                
                # Annotate the mean
                ax.annotate(f'{mean_val:.2f}', xy=(x_pos, mean_val), xytext=(x_pos, mean_val + 0.02), 
                            ha='center', va='bottom', fontsize=12, color=color, rotation=90)

# save fig    
plt.subplots_adjust(wspace=0.1)
fig_name_str = f'performance_results_holdout'
plt.savefig(f'{fig_name_str}.png', dpi=300,  bbox_inches = "tight")
plt.savefig(f'{fig_name_str}.svg',  bbox_inches = "tight")