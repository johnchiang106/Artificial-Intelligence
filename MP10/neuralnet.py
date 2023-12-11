# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by James Soole for the Fall 2023 semester

"""
This is the main entry point for MP10. You should only modify code within this file.
The unrevised staff files will be used for all other files and classes when code is run, 
so be careful to not modify anything else.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import get_dataset_from_arrays
from torch.utils.data import DataLoader


class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size, reg_strength=0.01):
        """
        Initializes the layers of your neural network.

        @param lrate: learning rate for the model
        @param loss_fn: A loss function defined as follows:
            @param yhat - an (N,out_size) Tensor
            @param y - an (N,) Tensor
            @return l(x,y) an () Tensor that is the mean loss
        @param in_size: input dimension
        @param out_size: output dimension

        For Part 1 the network should have the following architecture (in terms of hidden units):
        in_size -> h -> out_size , where  1 <= h <= 256
        
        We recommend setting lrate to 0.01 for part 1.

        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        # Add a convolutional layer
        self.channels, self.height, self.width = 3, 31, 31
        assert self.channels * self.height * self.width == in_size, "The channels, height and width should match the in_size"
        outC, kSize = 16, 3
        self.conv1 = nn.Conv2d(self.channels, outC, kernel_size=kSize)
        
        outH, outW = self.height - kSize + 1, self.width - kSize + 1
        self.fc1 = nn.Linear(outC * outH * outW, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, out_size)
        self.optimizer = optim.SGD(self.parameters(), lr=lrate, momentum=0.9)
        self.reg_strength = reg_strength
    

    def forward(self, x):
        """
        Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """
        # Reshape the input for the convolutional layer
        x = x.view(-1, self.channels, self.height, self.width)
        x = F.leaky_relu(self.conv1(x))
        # Flatten the output for the linear layers
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return x
    
    def l2_regularization(self):
        l2_loss = 0.0
        for param in self.parameters():
            l2_loss += torch.norm(param, p=2)  # L2 norm of each parameter
        return self.reg_strength * l2_loss

    def step(self, x, y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) for this batch as a float (scalar)
        """
        # Forward pass
        yhat = self.forward(x)

        # Compute the loss
        loss = self.loss_fn(yhat, y)

        # Add L2 regularization to the loss
        loss += self.l2_regularization()

        # Backpropagation and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()  # Convert the loss to a float scalar



def fit(train_set,train_labels,dev_set,epochs,batch_size=100):
    """ 
    Make NeuralNet object 'net'. Use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) Tensor
    @param train_labels: an (N,) Tensor
    @param dev_set: an (M,) Tensor
    @param epochs: an int, the number of epochs of training
    @param batch_size: size of each batch to train on. (default 100)

    This method *must* work for arbitrary M and N.

    The model's performance could be sensitive to the choice of learning rate.
    We recommend trying different values in case your first choice does not seem to work well.

    @return losses: list of floats containing the total loss at the beginning and after each epoch.
        Ensure that len(losses) == epochs.
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    """
    in_size = train_set.shape[1]  # Get the input size
    out_size = 4  # Classifying into four categories

    losses, yhats = [], []
    
    # Standardize the training and development data
    train_mean = train_set.mean(axis=0)
    train_std = train_set.std(axis=0)

    train_set = (train_set - train_mean) / train_std
    dev_set = (dev_set - train_mean) / train_std  # Use the same mean and std as training data for dev set

    net = NeuralNet(0.01, torch.nn.CrossEntropyLoss(), in_size, out_size)

    train_data = get_dataset_from_arrays(train_set, train_labels)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)

    # training
    for epoch in range(epochs):  # loop over the dataset multiple times
        total_loss = 0.0
        for batch in train_loader:
            batch_x, batch_y = batch['features'], batch['labels']  # Load data from the DataLoader
            loss = net.step(batch_x, batch_y)
            total_loss += loss

        losses.append(total_loss / len(train_loader))

    # Evaluate the model on the dev_set
    dev_set = torch.Tensor(dev_set)
    yhats = torch.argmax(net(dev_set), dim=1).numpy()
    yhats = yhats.astype(int)
    
    return losses, yhats, net