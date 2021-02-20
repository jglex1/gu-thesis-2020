"""
Author: Alex GU (44287207)
Date: Mon Aug  3 2020

# --- Description --- #
Module to handle the Support Vector Machine analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve


# Save generated images?
SAVEIMG = False

# LaTeX rendering, pgf fonts False
# Sets matplotlib settings for LaTeX export
# WARNING: Requires MikTeX installed
LATEX_RENDER = False

if LATEX_RENDER == True: plt.rcParams.update({
    'font.family': 'serif',
    'font.size'  : 15, # 16 standard
    "text.usetex": True,
    "pgf.rcfonts": False
    })


# ------------------------------ FUNCTIONS ---------------------------------- #


def untransform_form_energy(x):
    
    maxx = 1.459
    minn = -3.844
    
    return x*(maxx - minn) + minn


def generate_learning_curve(X, Y):
    
    train_sizes = np.linspace(0.05,1,20)
    
    train_sizes, train_scores, test_scores = learning_curve(
        svm.SVR(kernel='rbf', C=16, gamma=0.22), X, Y, cv=5, shuffle=True,
        train_sizes = train_sizes, scoring='neg_mean_squared_error')
   
    # Calculate statistical properties
    train_scores_mean = -1*np.mean(train_scores, axis=1)
    train_scores_std = -1*np.std(train_scores, axis=1)
    test_scores_mean = -1*np.mean(test_scores, axis=1)
    test_scores_std = -1*np.std(test_scores, axis=1)
    
    # Plot learning curve
    fig, ax = plt.subplots(figsize=(9,7),dpi=120)
    ax.grid()
    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Test score (5-fold cross-validated)")
    ax.legend()
    ax.set_xlabel("Number of Training examples")
    ax.set_ylabel(r"MSE")
    plt.show()
    
    return train_scores_mean, train_scores_std, test_scores_mean, test_scores_std
    

# ------------------------------ Main ----------------------------------- #


if __name__ == '__main__':
    
    f = lambda x: untransform_form_energy(x)
    
    # Dataset dir
    DATASET         = 'datasets_generated/testdb-norm.csv'
    CULLED_DATASET  = 'datasets_generated/testdb-norm-culled.csv'
    
    # Data
    df = pd.read_csv(DATASET)
    #df = pd.read_csv(culled_dataset)
    X = df.drop(['Formation energy [eV/atom]','Band gap [eV]'],axis=1)
    Y = df['Formation energy [eV/atom]']
    
    # Split into train and test data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, shuffle=True)
    
    # Fit regressor
    regr = svm.SVR(kernel='rbf', C=16, gamma=0.22) # Numbers from tuning
    regr.fit(X_train, Y_train)
    
    # Make Prediction
    predicted = regr.predict(X_test)
    r2 = metrics.r2_score(Y_test, predicted)
    rmse = metrics.mean_squared_error(Y_test, predicted)
    mae = metrics.mean_absolute_error(Y_test, predicted)
    
    scores = cross_val_score(regr, X_test, Y_test, cv=5, scoring='r2')
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 1.96))
    
    # Plot results
    fig, ax = plt.subplots(figsize=(9,7),dpi=120)
    x_range = np.linspace(0,len(Y_test),len(Y_test))
    Y_test, predicted = zip(*sorted(zip(Y_test, predicted)))
    ax.plot(f(np.array(Y_test)), f(np.array(predicted)), 'o')
    lims = (-4,1)
    ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    #ax.set_xlim(lims)
    #ax.set_ylim(lims)
    ax.set_xlabel("Actual Formation Energy")
    ax.set_ylabel("Predicted Formation Energy")
    ax.annotate(r'$R^2$ = {:.3f}'.format(r2), xy=(0.1, 0.8), xycoords='axes fraction')
    ax.annotate('RMSE = {:.3f}'.format(rmse), xy=(0.1, 0.75), xycoords='axes fraction')
    ax.annotate('MAE = {:.3f}'.format(mae), xy=(0.1, 0.7), xycoords='axes fraction')
    fig.tight_layout()
    plt.show()
    
    #train_scores_mean, train_scores_std, test_scores_mean, test_scores_std = generate_learning_curve(X, Y)