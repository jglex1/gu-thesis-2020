"""
Author: Alex GU (44287207)
Date: Thu Aug  5 2020

Hyperparameter optimisation.
"""

import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV


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


def plot3d(X,Y,Z,wire=False):
    """3D visualisation of supplied mesh grid data and elevation."""
    
    # Meshgrid and elevation data
    X, Y = np.meshgrid(X,Y)
    
    for i in range(len(Z)):
        for j in range(len(Z[0,:])):
            if Z[i,j] < 0.82:
                X[i,j] = np.nan
                Y[i,j] = np.nan
                Z[i,j] = np.nan
    
    # Plotting
    fig = plt.figure(dpi=160)
    ax = fig.gca(projection='3d')
    if wire == True:
        ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2)
    else:
        surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis,
                            linewidth=0.2, antialiased=False)
    ax.set_zlim3d(0.82,None)
    ax.set_xlabel("C value")
    ax.set_ylabel("gamma value")
    ax.set_zlabel(r"Average 5-fold cross-validated $R^2$ value")
    
    if wire == False: fig.colorbar(surf, shrink=0.4, aspect=5, pad=0.175)
    
    plt.show()
    
    return X, Y, Z


def get_results_matrix(regr, x_list, y_list):
    

    # 3D Plotting
    unit_length = len(y_list)
    z = regr.cv_results_['mean_test_score'][0:unit_length]
    for i in range(1,len(x_list)):
        z = np.vstack((z, regr.cv_results_['mean_test_score'][i*unit_length:(i+1)*unit_length]))
    z = z.T
    
    return z


# Plotting 2D
def plot2d_gamma(regr):
    """Only works if there are two parameters being varied."""
    
    a = regr.cv_results_['mean_test_score']
    b = regr.cv_results_['params']
    c = np.vstack((b, a)).T  
    
    sns.reset_orig()  # get default matplotlib styles back
    clrs = sns.color_palette('husl', n_colors=len(c_list))  # a list of RGB tuples
    
    fig, ax = plt.subplots()
    length = len(gamma_list)
    for num, C_value in enumerate(c_list):
        lines=ax.plot(gamma_list, c[:,1][length*(num):length*(num+1)], '.-', label='C={:.4f}'.format(C_value))
        lines[0].set_color(clrs[num])
        ax.legend(title='C values')
        ax.set_ylim([0.75,None])
        ax.set_xlabel("Gamma")
        ax.set_ylabel("Average 5-fold cross-validated R^2 value")
    plt.show()
    

def plot2d(regr):
    """Only works if there are two parameters being varied."""
    
    a = regr.cv_results_['mean_test_score']
    b = regr.cv_results_['params']
    c = np.vstack((b, a)).T  
    
    sns.reset_orig()  # get default matplotlib styles back
    clrs = sns.color_palette('husl', n_colors=len(c_list))  # a list of RGB tuples
    
    fig, ax = plt.subplots()
    length = len(c_list)
    for num, g_value in enumerate(gamma_list):
        lines=ax.plot(c_list, c[:,1][length*(num):length*(num+1)], '.-', label='γ={:.4f}'.format(g_value))
        lines[0].set_color(clrs[num])
        ax.legend(title='γ values')
        ax.set_xlabel("Regularisation parameter, C")
        ax.set_ylabel(r"Average 5-fold cross-validated $R^2$ value")
    fig.tight_layout()
    plt.show()
    

def search(param_grid, X_train, Y_train):
    # Initialise grid search SVM and train (5-fold cross validation)
    regr = svm.SVR()
    regr_gs = GridSearchCV(regr, param_grid)
    regr_gs.fit(X_train, Y_train)
    
    # Results table
    res = pd.DataFrame(regr_gs.cv_results_)
    res = res[['param_gamma','param_C','param_kernel','mean_test_score']]
    print(res)
    
    return regr_gs


if __name__ == "__main__":

    # Dataset dir
    DATASET         = 'datasets_generated/testdb-norm.csv'
    CULLED_DATASET  = 'datasets_generated/testdb-norm-culled.csv'

    # Data
    df = pd.read_csv(CULLED_DATASET)
    X = df.drop(['Formation energy [eV/atom]','Band gap [eV]'],axis=1)
    Y = df['Formation energy [eV/atom]']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    
    # Search space
    c_list = np.logspace(0,3,40)
    gamma_list = np.logspace(0,-3,60) # 70 
    #gamma_list = [0.22]
    param_grid_rbf=[{'C': c_list,
                'gamma': gamma_list,
                'kernel':['rbf']}]
    
    regr_gs = search(param_grid_rbf, X_train, Y_train)
    
    # 3D plotting
    z = get_results_matrix(regr_gs, c_list, gamma_list)
    X, Y, Z = plot3d(c_list, gamma_list, z)
    X, Y, Z = plot3d(c_list, gamma_list, z, wire=True)
    
    #plot2d(regr_gs)
    
    # Return best parameters
    print(regr_gs.best_params_)