## README
This repository contains the code relating to the honours thesis written by Alex Gu under supervision of Dr. Ruth Knibbe, titled *"Predicting Perovskite Formation Energy using Machine Learning in the development of Oxygen Electrode Materials for High Temperature Electrolysis in Solid Oxide Fuel Cells"*.

## Requirements

* It is recommended to use the Anaconda distribution of Python as it comes with several handy features including Spyder and is tailored for data science, you can download it here: https://www.anaconda.com/products/individual.
* It is recommended (and generally good programming practice) to develop the code for this project in a seperate environment, furthermore, a list of requirements, `requirements.txt` outlines the packages which I have installed in the working version of my code.
 
To create an environment using these requirements, in the Anaconda terminal (search for this after installed), type:

`conda create --name <env> --file requirements.txt`

**Note:**

* Ensure that `requirements.txt` is in the same directory as the Anaconda terminal working directory
* Replace `<env>` with the desired name of your environment, this can be of your choosing

**To launch the environment:** Every time you want to launch the environment,

* Open Anaconda terminal
* `conda activate <env>` to activate the environment
* `spyder` to launch Spyder (note: may need to `conda install spyder` if Spyder hasn't been installed in this environment)


## Overview

* `data_preprocessing.py` prepares the dataset for analysis, mainly handles the combination of the provided dataset and the mendeleev dataset (from the mendeleev Python package, note: this isn't supported in Python 3.8+). I recommend performing some data preprocessing as part of your pipeline to make it easier to see the effect of changing normalisation methods, etc. The way I currently have this implemented is not optimal.
* `feature_importances.py` calculates the feature importances using impurity-based importance and permutation-based importance and graphs the results. Note that there are ways of computing "importance".
* `model_svm.py` contains the code relating to the support vector machine regression.
* `model_svm_hyperparameter_opt.py` contains the code relating to tuning the hyperparameters of the support vector machine algorithm, note: there are other hyperparameters to tune aside from the ones included here.
* `model_nn.py` contains the code to implement a neural network in Pytorch.
* `model_nn_opt.py` contains the code which performs a grid search over the possible parameters of the neural network, e.g. learning rate, optimisation method, etc.
* `multicollinearity.py` contains some code to examine the correlations of the input features, note: there is considerably more "exploratory data analysis" which could be done other than a correlation heatmap.
* `pytorch_early_stopping.py` implements the early stopping method for the neural network. This code is written by "Bjarten" (link to original inside file) and used freely under the MIT License.


