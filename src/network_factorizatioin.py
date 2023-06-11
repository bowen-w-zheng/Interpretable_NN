"""
A neural network implementation for factorizing response matrix
"""

import numpy as np
import math
import os
import sys
import time
from torchvision import datasets, transforms
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

class NMFFC(nn.Module):
    def __init__(self, input_dim, output_dim, n_factors):
        super(NMFFC, self).__init__()

        # The weight matrix serves as W in the factorization
        self.fc = nn.Linear(input_dim, n_factors, bias=False)

        # This layer serves as H in the factorization
        self.decoder = nn.Linear(n_factors, output_dim, bias=False)

        # Enforce non-negativity with ReLU
        self.relu = nn.ReLU()

    def forward(self, x):
        encoded = self.fc(x)
        decoded = self.decoder(encoded)
        return decoded
    

def get_residual_ratio(X_true, X_pred):
    return np.square(X_true - X_pred).sum() / np.square(X_true - np.mean(X_true)).sum()

# function for setting up the model 
def setup_model(n_samples, n_response_patterns, n_factors, model_choice):
    # device 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using device: ", device)

    # model
    if model_choice == "NMFFC":
        model = NMFFC(n_samples, n_response_patterns, n_factors)
    
    return model.to(device)


# function for training the model
def train_NMFFC(model, X, epochs, max_lr = 0.1, batch_size = 10, l1_factor = 0.01):
    """
    params:
        model: the model to be trained
        X: the data matrix, should be organized as p by n, where p is the number of output dimensions, n is the number of samples
    """
    # One-hot encode samples
    n_samples = X.shape[1]
    I_one_hot = np.eye(n_samples)
    # Convert data to tensors
    I_tensor = torch.from_numpy(I_one_hot).float()
    X_tensor = torch.from_numpy(X.T).float()  # Transposed to match the samples
    # Create dataset and data loader
    dataset = TensorDataset(I_tensor, X_tensor)
    dataloader = DataLoader(dataset, batch_size = batch_size,  shuffle=True)  # Adjust batch size as needed

    # Apply NN-based NMF
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    # set a cycle scheduler to adjust the learning rate
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=len(dataloader), epochs=epochs)
    # Define the loss function

    # encourage sparsity with the loss function 

    loss_fn = nn.MSELoss()

    # Prepare to store loss for plotting
    loss_values = []

    # Training loop
    for epoch in range(epochs):
        for _, (I_batch, X_batch) in enumerate(dataloader):
            # Forward pass: Compute predicted y by passing x to the model
            X_pred = model(I_batch)

            # Compute loss
            mse_loss = loss_fn(X_pred, X_batch)
            l1_regularization = torch.tensor(0.).to(X_batch.device)
            for param in model.parameters():
                l1_regularization += torch.norm(param, 1)
            loss = mse_loss + l1_factor * l1_regularization

            # Zero gradients, perform a backward pass, and update the weights
            optimizer.zero_grad()
            loss.backward()

            # add non-negativity constraint
            for param in model.parameters():
                param.data = torch.clamp(param.data, min=0)

            optimizer.step()
        scheduler.step()
        loss_values.append(loss.item())

    # Compute residual
    X_pred = model(X_tensor).detach().numpy()
    print(f'mean subtrated residual is: {get_residual_ratio(X, X_pred.T)}')
    # Plot loss
    plt.plot(loss_values)
    plt.title('Loss values over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
