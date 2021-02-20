"""
Author: Alex GU (44287207)
Date: Tue Oct  6

Module to analyse data for multicollinearity.
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


SAVEIMG = False

# LaTeX rendering, pgf fonts False
# Sets matplotlib settings for LaTeX export
LATEX_RENDER = False

if LATEX_RENDER == True: plt.rcParams.update({
    'font.family': 'serif',
    'font.size'  : 16,
    "text.usetex": True,
    "pgf.rcfonts": False
    })



def correlation_heatmap(df):
    correlations = df.corr()

    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f',
                square=True, linewidths=.5, cbar_kws={"shrink": .70}) # annot=True
    plt.show()



if __name__ == '__main__':
    
    
    # Make sure to set what dataset is being used
    DATASET         = 'datasets_generated/testdb-norm.csv'
    CULLED_DATASET  = 'datasets_generated/testdb-norm-culled.csv'
    DATASET_ACTIVE = DATASET
    
    df = pd.read_csv(DATASET_ACTIVE)
    correlation_heatmap(df)