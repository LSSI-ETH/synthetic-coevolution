#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import pandas as pd

# aggregate results files for plotting

path = '../results/'
extension = 'csv'
os.chdir(path)
df_list = glob.glob('*.{}'.format(extension))
print(df_list)

out_df = pd.DataFrame()
for df in df_list:
    tmp_df = pd.read_csv(df)
    out_df = pd.concat([out_df,tmp_df], ignore_index = True)
    
out_df.to_csv('full_results.csv', header = True, index = False)