"""
Author: Alex GU (44287207)
Date: Wed Oct 14 2020

Optimise Neural Network
"""


#
# Note this code could take several MINUTES to complete or even HOURS depending
# on the number of variables must be looped over. To make it run quicker, try
# reducing the number of parameters, e.g. activation functions, network architectures,
# learning rates, patience for early stopping, etc.
#


from model_nn import train_validate_test_split, training_loop, plot_predicted_vs_actual
import pandas as pd
from torch import nn, optim
from tqdm import tqdm


def permute_act_fcn_models():
    
    models = []
    activation_fcns = [nn.ReLU, nn.Tanh, nn.LeakyReLU]
    
    for fcn in activation_fcns:
        
        
        # [64-50-1]
        MODEL = nn.Sequential(
            nn.Linear(64,50),
            fcn(),
            nn.Linear(50,1),
            fcn()).double()
        
        
        '''
        # [64-50-50-50-1]
        MODEL = nn.Sequential(
            nn.Linear(64,50),
            fcn(),
            nn.Linear(50,50),
            fcn(),
            nn.Linear(50,50),
            fcn(),
            nn.Linear(50,1),
            fcn()).double()
        
        '''
        
        '''
        # [64-100-100-1]
        MODEL = nn.Sequential(
            nn.Linear(64,50),
            fcn(),
            nn.Linear(50,50),
            fcn(),
            nn.Linear(50,1),
            fcn()).double()
        '''
        
        # Append all models to list
        models.append(MODEL)
    
    return models


if __name__ == '__main__':
    
    # 1. Select dataset
    dataset_path = 'datasets_generated/testdb-norm.csv'
    df = pd.read_csv(dataset_path)
    df_end = df["Formation energy [eV/atom]"]
    df = df.drop(["Formation energy [eV/atom]", 'Band gap [eV]'], axis=1)
    df["Formation energy [eV/atom]"] = df_end
    
    # Split df into train-validation-test DataLoader objects
    train_set, val_set, test_set, atom_datasets = train_validate_test_split(df, [0.2, 0.2]) # 0.2 test, 0.2 val
    
    # Ranges to sweep over
    learning_rate_sweep = [0.001]
    model_sweep         = permute_act_fcn_models()
    loss_critera_sweep  = [nn.MSELoss()]
    optimiser_sweep     = [optim.SGD] # Could also include: optim.Adam
    scheduler_sweep     = [optim.lr_scheduler.ReduceLROnPlateau] # Also: None, 
    
    records = []
    possible_combinations = []
    
    # SWEEP
    for LEARNING_RATE in learning_rate_sweep:
        for MODEL in model_sweep:
            for LOSS_CRITERION in loss_critera_sweep:
                for optimiser_fcn in optimiser_sweep:
                    OPTIMISER = optimiser_fcn(MODEL.parameters(), lr=LEARNING_RATE)
                    for scheduler_fcn in scheduler_sweep:
                        if scheduler_fcn != None:
                            SCHEDULER = scheduler_fcn(OPTIMISER, 'min',
                                                      factor= 0.9, verbose=True)
                        else:
                            SCHEDULER = None
                            
                        possible_combinations.append([LEARNING_RATE, MODEL, LOSS_CRITERION,
                                       OPTIMISER, SCHEDULER])
    
    for params in tqdm(possible_combinations):
        
        LEARNING_RATE, MODEL, LOSS_CRITERION, OPTIMISER, SCHEDULER = params
        
        # Train and eval model
        losses = training_loop(end_epochs=1000,
                                NN_model = MODEL,
                                train_data = train_set,
                                validation_data = val_set,
                                loss_fcn = LOSS_CRITERION,
                                optimiser = OPTIMISER,
                                scheduler = SCHEDULER,
                                patience = 21,
                                min_epoch_pre_scheduler = 150,
                                verbose=False)
        
        test_r2, train_r2 = plot_predicted_vs_actual(MODEL,
                                                 atom_datasets)
        
        records.append([LEARNING_RATE, MODEL,
                       OPTIMISER, SCHEDULER, min(losses["train"]),
                       min(losses["validation"]), train_r2, test_r2])
        
        print("Done...")
    
    results = pd.DataFrame(records)
