# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kstest, ttest_ind
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
import os



def correlation_map(features,name):
    plt.figure(figsize=(20, 12))
    corr = features.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cut_off = 0.01  # only show cells with abs(correlation) at least this value
    extreme_1 = 0.75  # show with a star
    extreme_2 = 0.85  # show with a second star
    extreme_3 = 0.90  # show with a third star
    mask |= np.abs(corr) < cut_off
    corr = corr[~mask]  # fill in NaN in the non-desired cells

    remove_empty_rows_and_cols = True
    if remove_empty_rows_and_cols:
        wanted_cols = np.flatnonzero(np.count_nonzero(~mask, axis=1))
        wanted_rows = np.flatnonzero(np.count_nonzero(~mask, axis=0))
        corr = corr.iloc[wanted_cols, wanted_rows]

    annot = [[f"{val:.4f}"
              + ('' if abs(val) < extreme_1 else '\n★')  # add one star if abs(val) >= extreme_1
              + ('' if abs(val) < extreme_2 else '★')  # add an extra star if abs(val) >= extreme_2
              + ('' if abs(val) < extreme_3 else '★')  # add yet an extra star if abs(val) >= extreme_3
              for val in row] for row in corr.to_numpy()]
    # cmap="Blues"
    heatmap = sns.heatmap(corr, vmin=-1, vmax=1, annot=annot, fmt='', cmap='BrBG')
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=15, horizontalalignment='right')
    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=15, horizontalalignment='right')
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize': 20}, pad=16)
    #plt.show()
    plt.savefig(fr'images\{name}.png')

def check(x):
    if x == 'yes' :
        return 1
    elif x == 'no':
        return 0


   
    
# Specify the path to the CSV file
csv_file_path = r"/var/share/rs1/LCL_DATA/preparded_houshold_data/block9_MAC005517.csv"



# Get the name of the CSV file
file_name = os.path.basename(csv_file_path)

#read the csv file
df=pd.read_csv(csv_file_path)

print(df.columns)



# Feature that their correlation to be computed 
features = df[['energy(kWh/hh)', 'visibility','windBearing','temperature','dewPoint','pressure',
                 'apparentTemperature','windSpeed','humidity','holydays']]


#Compute and plot the correlations between the features
correlation_map(features,file_name)

print(df['energy(kWh/hh)'].corr(df['holydays']))