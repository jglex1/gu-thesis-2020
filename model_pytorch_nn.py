"""
Author: Alex GU (44287207)
Date: Thu Oct 1

PyTorch implementation of Neural Network.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from pytorch_early_stopping import EarlyStopping

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split



class AtomDataset(Dataset):
    """A small class to manage the DataLoader"""
    
    def __init__(self, df, label_col_name='Formation energy [eV/atom]'):
        
        self.datalist = torch.from_numpy(df.drop(label_col_name,axis=1).values)
        self.labels = torch.from_numpy(df[label_col_name].values)
        
    def __getitem__(self, index):
        
        return self.datalist[index], self.labels[index]
    
    def __len__(self):
        
        return self.datalist.shape[0]
    
    
    
def train_validate_test_split(df, proportion):
    """Splits the dataset into three (3) iterable DataLoader objects: train,
    validation and test sets.
    
    Parameters
        df          : (pd.Dataframe) Pandas dataframe containing the data.
        proportions : (list) List containing the train-validate-test split
                      proportions of the form [test, validate].

    Returns
        (DataLoader) training_set, validation_set, test_set
    """
    
    # Check
    assert len(proportion) == 2, "Requires 2 proportion values"
    
    # Get splitting proportions
    test_prop = proportion[0]
    train_val_prop = proportion[1]/(1-test_prop)
    
    # Seperate dataset into train-validate-test
    trainval_df, test_df = train_test_split(df, test_size=test_prop, shuffle=True)
    train_df, validation_df = train_test_split(trainval_df, test_size=train_val_prop, shuffle=True)
    
    # Create iterable Tensor objects
    training_dataset = AtomDataset(train_df)
    validation_dataset = AtomDataset(validation_df)
    test_dataset = AtomDataset(test_df)
    
    # Create iterable DataLoader objects
    training_set = DataLoader(dataset = training_dataset, batch_size= 1, shuffle = True)
    validation_set = DataLoader(dataset = validation_dataset, batch_size= 1, shuffle = True)
    test_set = DataLoader(dataset = test_dataset, batch_size= 1, shuffle = True)

    return training_set, validation_set, test_set, [training_dataset, validation_dataset, test_dataset]



def training_loop(end_epochs, NN_model, train_data, validation_data, loss_fcn,
                  optimiser, scheduler, patience, min_epoch_pre_scheduler,
                  verbose=True):
    """Trains a Neural Network.
    
        end_epochs          - (int) max number of epochs to train
        NN_model            - Neural Network model to be trained
        train_data          - (DataLoader) Training set
        validation_data     - (DataLoader) Validation set
        loss_fcn            - Loss criterion
        optimiser           - Optimizer function
        scheduler           - Variable LR scheduler
        patience            - How long to wait after last time validation loss improved.
    
    """
    
    # To track losses
    losses = {
    "train": [],
    "validation": []
    }
    
    # Initialise EarlyStopping class
    early_stopping = EarlyStopping(patience=patience, verbose=verbose)

    for epoch in range(1, end_epochs+1):
        
        # TRAINING
        train_epoch_loss = 0.0
        NN_model.train()
        
        for train_x, train_y in train_data:
            NN_model.zero_grad()
            y_train_output = NN_model(train_x)
            train_loss = loss_fcn(y_train_output[0], train_y)
            train_loss.backward()
            optimiser.step()
            train_epoch_loss += train_loss.item()
        
        # VALIDATION  
        val_epoch_loss = 0.0
        NN_model.eval()
        
        with torch.no_grad():
            for val_x, val_y in validation_data:
                y_val_output = NN_model(val_x)
                val_loss = loss_fcn(y_val_output[0], val_y)
                val_epoch_loss += val_loss.item()
        
        # Append epoch loss stats
        avg_epoch_train_loss = train_epoch_loss/len(train_data)
        avg_epoch_val_loss = val_epoch_loss/len(validation_data)
        losses["train"].append(avg_epoch_train_loss)
        losses["validation"].append(avg_epoch_val_loss)
        
        # Print epoch losses to console
        if epoch <=10 or (epoch % 10) == 0:
            if verbose: print(f'Epoch {epoch:0>3d}: | Train Loss: {avg_epoch_train_loss:.5f} | Val Loss: {avg_epoch_val_loss:.5f}')
        
        # DYNAMIC LR
        if scheduler != None and epoch > min_epoch_pre_scheduler:
            scheduler.step(val_epoch_loss)
            
        # EARLY STOPPING
        early_stopping(avg_epoch_val_loss, NN_model)
        if early_stopping.early_stop:
            print("Early stopping")
            NN_model.load_state_dict(torch.load("checkpoint.pt")) # Recover best model
            break # Break from Epoch loop
   
    return losses


def untransform_form_energy(x):
    
    maxx = 1.459
    minn = -3.844
    
    return x*(maxx - minn) + minn
        
        
def plot_predicted_vs_actual(model, atom_datasets, NORM=True, plot_true=False):
    
    # Split into the different datasets
    training_dataset, validation_dataset, test_dataset = atom_datasets
    
    # X/Y
    X_test, X_train = test_dataset.datalist, training_dataset.datalist
    Y_test, Y_train = test_dataset.labels, training_dataset.labels
    
    with torch.no_grad():
        pred_test = model(X_test)
        pred_train = model(X_train)
    
    test_r2 = r2_score(Y_test, pred_test)
    print("Test: ",test_r2)
    train_r2 = r2_score(Y_train, pred_train)
    print("Train: ", train_r2)
    
    f = lambda x: x
    if NORM == False:
        f = lambda x: untransform_form_energy(x)
    
    if plot_true:
        fig, axs = plt.subplots(1, 2, figsize=(8,8),dpi=100)
        axs[0].plot(f(Y_test), f(pred_test), '.')
        lims = (-4,1)
        axs[0].plot(lims, lims, 'k--', alpha=0.75, zorder=0)
        axs[0].set_aspect('equal')
        axs[0].set_xlabel("Actual Formation Energy (Normalised)")
        axs[0].set_ylabel("Predicted Formation Energy (Normalised)")
        axs[0].annotate(r'$R^2$ = {:.3f}'.format(test_r2), xy=(0.1, 0.8), xycoords='axes fraction')
        # Right plot
        axs[1].plot(f(Y_train), f(pred_train), '.', label='Train')
        axs[1].plot(lims, lims, 'k--', alpha=0.75, zorder=0)
        axs[1].set_aspect('equal')
        axs[1].annotate(r'$R^2$ = {:.3f}'.format(train_r2), xy=(0.1, 0.8), xycoords='axes fraction')
        axs[1].set_xlabel("Actual Formation Energy (Normalised)")
        plt.tight_layout()
        plt.show()

    return test_r2, train_r2    


def visualise_loss(train_loss_list, val_loss_list):
    
    fig, ax = plt.subplots()
    ax.plot(range(1,len(train_loss_list)+1), train_loss_list, label="Training Loss")
    ax.plot(range(1,len(val_loss_list)+1), val_loss_list, label="Validation Loss")
    
    # Plot early stopping point
    minposs = val_loss_list.index(min(val_loss_list))+1 
    ax.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')
    
    # Labels
    ax.set_xlabel("Epochs")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    
    
    # 1. Select dataset
    dataset_path = "../datasets_generated/db-sem2-norm-v2_8_27-modified.csv"
    df = pd.read_csv(dataset_path)
    
    # 2. Split df into train-validation-test DataLoader objects
    train_set, val_set, test_set, atom_datasets = train_validate_test_split(df, [0.2, 0.2]) # 0.2 test, 0.2 val
    
    # 3. Select model and define everything
    LEARNING_RATE = 0.1
    MAX_EPOCHS = 1000
    
    MODEL = nn.Sequential(
        nn.Linear(64,50),
        nn.ReLU(),
        nn.Linear(50,50),
        nn.ReLU(),
        nn.Linear(50,1),
        nn.ReLU()).double()
    
    LOSS_CRITERION = nn.MSELoss()
    OPTIMISER = optim.SGD(MODEL.parameters(), lr=LEARNING_RATE)
    SCHEDULER = optim.lr_scheduler.ReduceLROnPlateau(OPTIMISER, 'min', factor= 0.9, verbose=True)
    
    # 4. Train 
    losses = training_loop(end_epochs=1000,
                          NN_model = MODEL,
                          train_data = train_set,
                          validation_data = val_set,
                          loss_fcn = LOSS_CRITERION,
                          optimiser = OPTIMISER,
                          scheduler = SCHEDULER,
                          patience = 31, # early stopping counter (21)
                          min_epoch_pre_scheduler = 150)
    
    # 5. Plot
    test_r2, train_r2 = plot_predicted_vs_actual(MODEL, atom_datasets)
    #plot_predicted_vs_actual(MODEL, atom_datasets, NORM=False, plot_true=True)
    
    # 6. Visualise loss
    visualise_loss(losses["train"], losses["validation"])