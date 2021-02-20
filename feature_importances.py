"""
Author: Alex GU
Date: Mon Aug  3 2020

# --- Description --- #
Determines feature importance using RF regression.
"""

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from tqdm import tqdm

# Save generated images?
SAVEIMG = False

# LaTeX rendering, pgf fonts False
# Sets matplotlib settings for LaTeX export
# WARNING: Requires MikTeX installed
LATEX_RENDER = False

if LATEX_RENDER == True: plt.rcParams.update({
    'font.family': 'serif',
    'font.size'  : 10, # 16 standard
    "text.usetex": True,
    "pgf.rcfonts": False
    })


# ------------------------------ FUNCTIONS ---------------------------------- #


def impurity_feature_importance(rf, norm=True):

    results = get_imp_feature_importance_metrics(rf,adv=True)  
    names_sorted = results[4]
    if norm==True:
        imp = results[3]
    else:
        imp = results[2]

    # culled=(6,12) font 16
    # non-culled=(6,16) and font 14
    fig, ax = plt.subplots(figsize=(6,12)) 
    ax.barh(names_sorted, imp)
    ax.set_xlabel("Relative importance (normalised by max)")
    plt.tight_layout()
    
    if SAVEIMG == True:
        plt.savefig(f'{IMG_PATH_MOD}\impurity-standard.png', format='png')
        plt.savefig(f'{IMG_PATH_MOD}\impurity-standard.eps', format='eps')
    
    
def plot_feature_impurity(X, imp, title='', xlabel="Relative importance (normalised by max)"):
    """HELPER FUNCTION FOR: plot_average_impurity_feature_importance"""
    
    tree_importance_sorted_idx = np.argsort(imp)
    names_sorted = X.columns[tree_importance_sorted_idx]
    
    fig, ax = plt.subplots(figsize=(6,12), dpi=100)
    ax.barh(names_sorted, imp[tree_importance_sorted_idx])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    plt.tight_layout()
    if SAVEIMG == True:
        plt.savefig(f'{IMG_PATH_MOD}\impurity-iterations.png', format='png',dpi=1000)
        plt.savefig(f'{IMG_PATH_MOD}\impurity-iterations.eps', format='eps')
    plt.show()
    
    
def permutation_feature_importance_two(rf, X_test, Y_test, X_train, Y_train):
    
    # Test
    result_test = permutation_importance(rf, X_test, Y_test, n_repeats=20,n_jobs=2)
    sorted_mean_imp_test = result_test.importances_mean.argsort()
    
    # Train
    result_train = permutation_importance(rf, X_train, Y_train, n_repeats=20,n_jobs=2)
    sorted_mean_imp_train = result_train.importances_mean.argsort()
    
    fig, (ax1,ax2) = plt.subplots(2,1, sharex=True, figsize=(10,8), dpi=100)
    # Plot Test
    ax1.boxplot(result_test.importances[sorted_mean_imp_test].T, vert=False, labels=X_test.columns[sorted_mean_imp_test])
    ax1.set_title("Permutation Importances (Test set)")    
    # Plot Train
    ax2.boxplot(result_train.importances[sorted_mean_imp_train].T, vert=False, labels=X_train.columns[sorted_mean_imp_train])
    ax2.set_title("Permutation Importances (Training set)")
    ax2.set_xlabel("Mean decrease in accuracy")
    fig.tight_layout()
    
    
def permutation_feature_importance(rf, X_data, Y_data, set_name):
    
    result = permutation_importance(rf, X_data, Y_data, n_repeats=20,n_jobs=2)
    sorted_mean_imp = result.importances_mean.argsort()
    fig, ax = plt.subplots(figsize=(12,10), dpi=100)
    ax.boxplot(result.importances[sorted_mean_imp].T, vert=False, labels=X_test.columns[sorted_mean_imp])
    ax.set_title(f"Permutation Importances ({set_name} set)") 
    ax.set_xlabel("Mean decrease in accuracy")
    fig.tight_layout()

    
    
def get_imp_feature_importance_metrics(rf, adv=False):
    
    imp = rf.feature_importances_
    imp_norm = (imp - np.min(imp)) / (np.max(imp) - np.min(imp))
    
    if adv==True:
        tree_importance_sorted_idx = np.argsort(rf.feature_importances_)
        names_sorted = X.columns[tree_importance_sorted_idx]
        imp_sorted = imp[tree_importance_sorted_idx]
        imp_norm_sorted = imp_norm[tree_importance_sorted_idx]
        return imp, imp_norm, imp_sorted, imp_norm_sorted, names_sorted
    else:
        return imp, imp_norm


def kde_plot(selection, X, results_norm):
    """
    selection    : (str) Name of feature whose feature importance spectrum 
                    over the n iterations is to be plotted.
    X            : (np.array) Dataset of values.
    results_norm : (np.array) Second output from compute_n_iterations() 
    """
    
    #fig = plt.figure()
    name_idx = list(X.columns).index(selection)
    data = results_norm[:,name_idx]
    sns.kdeplot(data, bw=.2, label=selection)
    sns.rugplot(data)
    plt.xlabel("Importance")
    plt.title('Distribution of "{}" importance over {} iterations'.format(selection, len(results_norm)))
    plt.xlim(0, 1)    
    plt.show()


def compute_n_iterations(n, X, Y, test_size=0.3):
    """Computes n iterations of random forest model and records the results of
    the feature importance in a matrix.
    
    Parameters
    ----------
    n : int
        Number of iterations of random forest regressor to initalise.
    X, Y : np.array
        The X and Y dataset (i.e. features and labels) for training the random
        forest regressor model.
    test_size : float
        The percentage split value between the test and training set. The
        default is 0.3, i.e. 30% of the data is reserved for testing.

    Returns
    -------
    old_results : A matrix where the rows correspond to the feature importance
    values at each iteration.
    old_results_norm : Same as above except the values each iteration are normalised.
    """
    
    for i in tqdm(range(n)):
        
        # Split data
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
        
        # Fit RF regressor
        rfr = RandomForestRegressor()
        rfr.fit(X_train, Y_train)
        
        # Get feature importance results
        results = get_imp_feature_importance_metrics(rfr)[0]
        results_norm = get_imp_feature_importance_metrics(rfr)[1]
        if i != 0:
            old_results = np.vstack((old_results, results))
            old_results_norm = np.vstack((old_results_norm, results_norm))
        else:
            old_results = results    
            old_results_norm = results_norm
    
    return old_results, old_results_norm


def columnwise_avg(results):
    """Given a numpy array of size n x m, return a vector of size m with the
    average value for each column of the matrix.
    
    Returned vector can either be absolute or normalised, i.e.
        avg -> absolute mean
        avgs_norm -> normalised mean (normalised by avg)
    """
    
    avgs = []    
    for i in range(len(results[0,:])):
        avgs.append(np.mean(results[:,i]))
    
    avgs = np.array(avgs)
    avgs_norm = (avgs - np.min(avgs)) / (np.max(avgs) - np.min(avgs))

    return avgs, avgs_norm


def plot_average_impurity_feature_importance(n, X, Y):
    """
    Uses:
        + compute_n_iterations()
        + columnwise_avg()
    
    Parameters
    ----------
    n : int
        The number of iterations to complete.
    X, Y : np.array
        The X and Y dataset (i.e. features and labels) for training the random
        forest regressor model.

    Returns
    -------
    None. But will output a plot of the impurity based feature importance. 
    """
    
    # Gets a matrix of all feature importance data for n iterations
    results, results_norm = compute_n_iterations(n, X, Y)
    # Averages the results over n iterations
    _, avg_results_norm = columnwise_avg(results)
    # Plot the results
    plot_feature_impurity(X, avg_results_norm, title="Average feature importance over {} iterations".format(n))
    plot_box_plot(X, results_norm)

    return results, results_norm


def plot_box_plot(X, results_norm):
    
    tree_importance_sorted_idx = np.argsort(np.mean(results_norm,axis=0))
    names_sorted = X.columns[tree_importance_sorted_idx]
    fig, ax = plt.subplots(figsize=(6,12), dpi=100)
    ax.boxplot(results_norm[:,tree_importance_sorted_idx], vert=False, labels=names_sorted)
    ax.set_xlabel("Relative importance (normalised by max)")
    ax.set_title("Distribution of 100 iterations of importances")
    fig.tight_layout()
    if SAVEIMG == True:
        plt.savefig(f'{IMG_PATH_MOD}\impurity-iterations-box.png', format='png',dpi=1000)
        plt.savefig(f'{IMG_PATH_MOD}\impurity-iterations-box.eps', format='eps')
    plt.show()


if __name__ == '__main__':
        
    # Dataset dir
    DATASET         = 'datasets_generated/testdb-norm.csv'
    CULLED_DATASET  = 'datasets_generated/testdb-norm-culled.csv'
    
    # Dataset for analysis
    DATASET_ACTIVE = CULLED_DATASET
    RANDOM_NUMBER = False 
    
    # Image save file dir
    IMG_PATH_MOD = "images\\feature-imp\\"
    
    
    # ------------------------------ Main ----------------------------------- #
    
    
    df = pd.read_csv(DATASET_ACTIVE)

    # Seperate data and labels, handle incl. of random number
    X = df.drop(['Formation energy [eV/atom]','Band gap [eV]'],axis=1)
    Y = df['Formation energy [eV/atom]']
    if RANDOM_NUMBER == True:
        X['Random Number [0,1]'] = np.random.random_sample(X.shape[0])    

    # Split data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
    
    # Fit RF regressor
    rfr = RandomForestRegressor()
    rfr.fit(X_train, Y_train)
    predicted = rfr.predict(X_test)
    
    
    if 0:
        # Impurity based feature importance
        impurity_feature_importance(rfr)
        
        # Permutation feature importance
        permutation_feature_importance_two(rfr, X_test, Y_test, X_train, Y_train)
        permutation_feature_importance(rfr, X_test, Y_test, set_name='Test')
        permutation_feature_importance(rfr, X_train, Y_train, set_name='Train')
        
        # Plots the feature importances for n iterations
        results, results_norm = plot_average_impurity_feature_importance(100, X, Y)
        
        # kde_plot
        kde_plot("A en pauling", X, results_norm)