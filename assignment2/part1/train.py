################################################################################
# MIT License
#
# Copyright (c) 2022 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2022
# Date Created: 2022-11-14
################################################################################

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.models as models
import torch.optim as optim

from copy import deepcopy

from cifar100_utils import get_train_validation_set, get_test_set, set_dataset


def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(num_classes=100):
    """
    Returns a pretrained ResNet18 on ImageNet with the last layer
    replaced by a linear layer with num_classes outputs.
    Args:
        num_classes: Number of classes for the final layer (for CIFAR100 by default 100)
    Returns:
        model: nn.Module object representing the model architecture.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # Get the pretrained ResNet18 model on ImageNet from torchvision.models
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    # Randomly initialize and modify the model's last layer for CIFAR100.
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    nn.init.normal_(model.fc.weight, mean=0, std=0.01)
    nn.init.constant_(model.fc.bias, 0)

    #######################
    # END OF YOUR CODE    #
    #######################

    return model


def train_model(model, lr, batch_size, epochs, data_dir, checkpoint_name, device, augmentation_name=None):
    """
    Trains a given model architecture for the specified hyperparameters.

    Args:
        model: Model to train.
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the dataset should be loaded from or downloaded to.
        checkpoint_name: Filename to save the best model on validation.
        device: Device to use.
        augmentation_name: Augmentation to use for training.
    Returns:
        model: Model that has performed best on the validation set.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Load the datasets

    # Loading the datase
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_set, val_set = get_train_validation_set(data_dir=data_dir, augmentation_name=augmentation_name)
    train_loader = data.DataLoader(dataset=train_set, batch_size=batch_size, **kwargs)
    val_loader = data.DataLoader(dataset=val_set, batch_size=batch_size, **kwargs)
    # Initialize the optimizer (Adam) to train the last layer of the model.
    optimizer = optim.Adam(model.fc.parameters(), lr=lr)

    # Training loop with validation after each epoch. Save the best model.
    loss_module = nn.CrossEntropyLoss()
    losses = []
    prev_acc = 0
    for ep in range(epochs):
      model.train()
      total_loss = 0
      total_batch = 0
      for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        total_batch += 1
        y_pred = model(x)
        loss = loss_module(y_pred, y)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
      acc = evaluate_model(model, val_loader, device)
      print(f'epoch: {ep} acc: {acc}')
      if acc > prev_acc:
        torch.save(model, checkpoint_name)
      losses.append(total_loss / total_batch)

    # Load the best model on val accuracy and return it.
    best_model = torch.load(checkpoint_name)
    #######################
    # END OF YOUR CODE    #
    #######################

    return best_model


def evaluate_model(model, data_loader, device):
    """
    Evaluates a trained model on a given dataset.

    Args:
        model: Model architecture to evaluate.
        data_loader: The data loader of the dataset to evaluate on.
        device: Device to use for training.
    Returns:
        accuracy: The accuracy on the dataset.

    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # Set model to evaluation mode (Remember to set it back to training mode in the training loop)
    model.eval()
    # Loop over the dataset and compute the accuracy. Return the accuracy
    # Remember to use torch.no_grad().
    correct = 0
    total = 0
    for batch_x, batch_y in data_loader:
        with torch.no_grad():
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_pred = np.argmax(model(batch_x).cpu().numpy(), axis=1)
            correct += np.sum(np.equal(batch_pred, batch_y.cpu().numpy()))
            total += batch_y.shape[0]
    accuracy = correct / total
    #######################
    # END OF YOUR CODE    #
    #######################

    return accuracy


def main(lr, batch_size, epochs, data_dir, seed, augmentation_name, test_noise):
    """
    Main function for training and testing the model.

    Args:
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        seed: Seed for reproducibility.
        augmentation_name: Name of the augmentation to use.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # Set the seed for reproducibility
    set_seed(seed)

    # Set the device to use for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load the model
    model = get_model()
    model = model.to(device)

    # Get the augmentation to use
    

    # Train the model
    best_model = train_model(model=model, lr=lr, batch_size=batch_size, epochs=epochs, data_dir=data_dir, checkpoint_name="best_model", device=device, augmentation_name=augmentation_name)

    # Evaluate the model on the test set
    test_set = get_test_set(data_dir=data_dir, test_noise=test_noise)
    test_loader = data.DataLoader(test_set, batch_size=batch_size)
    acc = evaluate_model(model=best_model, data_loader=test_loader, device=device)

    # default setting 30 epochs acc=0.5728. 
    print('test acc: ', acc)

    #######################
    # END OF YOUR CODE    #
    #######################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Feel free to add more arguments or change the setup

    parser.add_argument('--lr', default=0.001, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')
    parser.add_argument('--epochs', default=30, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=123, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR100 dataset.')
    parser.add_argument('--dataset', default='cifar100', type=str, choices=['cifar100', 'cifar10'],
                        help='Dataset to use.')
    parser.add_argument('--augmentation_name', default=None, type=str,
                        help='Augmentation to use.')
    parser.add_argument('--test_noise', default=False, action="store_true",
                        help='Whether to test the model on noisy images or not.')

    args = parser.parse_args()
    kwargs = vars(args)
    set_dataset(kwargs.pop('dataset'))
    main(**kwargs)
