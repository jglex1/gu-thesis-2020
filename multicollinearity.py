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


# Make sure to set what dataset is being used
dataset = '../datasets_generated/db-sem2-norm-v2_8_27.csv'
culled_dataset = '../datasets_generated/db-sem2-culled-norm-v1_8_27.csv'
DATASET_ACTIVE = dataset



def correlation_heatmap(df):
    correlations = df.corr()

    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f',
                square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70})
    plt.show()


if __name__ == '__main__':
    
    df = pd.read_csv(DATASET_ACTIVE)
    correlation_heatmap(df)