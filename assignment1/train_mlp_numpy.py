################################################################################
# MIT License
#
# Copyright (c) 2023 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2023
# Date Created: 2023-11-01
################################################################################
"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from tqdm.auto import tqdm
from copy import deepcopy
from mlp_numpy import MLP
from modules import CrossEntropyModule, LinearModule
import cifar10_utils
import matplotlib.pyplot as plt
import seaborn as sns

import torch


def confusion_matrix(predictions, targets):
    """
    Computes the confusion matrix, i.e. the number of true positives, false positives, true negatives and false negatives.

    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      labels: 1D int array of size [batch_size]. Ground truth labels for
              each sample in the batch
    Returns:
      confusion_matrix: confusion matrix per class, 2D float array of size [n_classes, n_classes]
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    pred_y = np.argmax(predictions, axis=1)
    num_class = predictions.shape[1]
    # row represents prediction and col represents ground truth
    conf_mat = np.zeros((num_class, num_class))
    for ith_sample, pred in enumerate(pred_y):
        truth = targets[ith_sample]
        conf_mat[pred][truth] += 1
    #######################
    # END OF YOUR CODE    #
    #######################
    return conf_mat

def _fscore(precision, recall, beta):
    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)

def confusion_matrix_to_metrics(confusion_matrix, beta=1.):
    """
    Converts a confusion matrix to accuracy, precision, recall and f1 scores.
    Args:
        confusion_matrix: 2D float array of size [n_classes, n_classes], the confusion matrix to convert
    Returns: a dictionary with the following keys:
        accuracy: scalar float, the accuracy of the confusion matrix
        precision: 1D float array of size [n_classes], the precision for each class
        recall: 1D float array of size [n_classes], the recall for each clas
        f1_beta: 1D float array of size [n_classes], the f1_beta scores for each class
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    row_sum = np.sum(confusion_matrix, axis=1)
    col_sum = np.sum(confusion_matrix, axis=0)
    true_pred = np.diag(confusion_matrix)
    total_samples = np.sum(confusion_matrix)
    metrics = {}
    metrics['precision'] = true_pred / row_sum
    metrics['recall'] = true_pred / col_sum
    metrics['accuracy'] = np.sum(true_pred) / total_samples 
    metrics['f1_beta'] = _fscore(metrics['precision'], metrics['recall'], beta)
    #######################
    # END OF YOUR CODE    #
    #######################
    return metrics

def evaluate_model(model, data_loader, num_classes=10, plot=False):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
        metrics: A dictionary calculated using the conversion of the confusion matrix to metrics.

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset,
          independent of batch sizes (not all batches might be the same size).
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    all_conf_mat = np.zeros((num_classes, num_classes))
    for batch_x, batch_y in data_loader:
        batch_x = np.reshape(batch_x, (batch_x.shape[0], -1))
        batch_pred = model.forward(batch_x)
        all_conf_mat += confusion_matrix(batch_pred, batch_y)
    if plot:
      _plot_confusion_matrix(all_conf_mat)
      for beta in [.1, 1, 10]:
        metrics = confusion_matrix_to_metrics(all_conf_mat, beta)
        print(f'beta {beta} fcore: {metrics["f1_beta"]}' )
    metrics = confusion_matrix_to_metrics(all_conf_mat)
        
    #######################
    # END OF YOUR CODE    #
    #######################
    return metrics

def _plot_confusion_matrix(confusion_matrix):
  plt.figure(figsize=(8, 6))
  sns.heatmap(confusion_matrix, annot=True, fmt="f", cmap='Blues')
  plt.xlabel('Predicted Class')
  plt.ylabel('Actual Class')
  plt.title('Confusion Matrix')
  plt.show()

def _plot_losses(losses):
  plt.plot(losses)
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.title('Training Loss')
  plt.show()

    

def train(hidden_dims, lr, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation. Between 0.0 and 1.0
      logging_info: An arbitrary object containing logging information. This is for you to 
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model. 
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set, 
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    ## Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=True)

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # TODO: Initialize model and loss module
    n_input = 32*32*3
    n_hidden = hidden_dims
    n_classes = 10
    model = MLP(n_input, n_hidden, n_classes)
    loss_module = CrossEntropyModule()
    # TODO: Training loop including validation
    X = []
    Y = []
    for i, batch in enumerate(cifar10_loader['train']):
          batch_x = batch[0]
          batch_x = batch_x.reshape(batch_x.shape[0], -1)
          X.append(batch_x)
          Y.append(batch[1])
    best_models = []
    losses = []
    for ep in range(epochs):
        total_loss = 0
        num_batch = 0
        for batch_idx in range(len(X)):
            batch_x = X[batch_idx]
            batch_y = Y[batch_idx]
            sample_size = len(batch_y)
            num_batch += 1
            out = model.forward(batch_x)
            loss = loss_module.forward(out, batch_y)
            total_loss += loss
            dout = loss_module.backward(out, batch_y)
            model.backward(dout)
            for l in model.layers:
              if isinstance(l, LinearModule):
                l.params['weight'] -= lr * l.grads['weight'] 
                l.params['bias'] -= lr * l.grads['bias'] 
            model.clear_cache()
        losses.append(total_loss/num_batch)
        print(f'Epoch {ep+1}/{epochs}: loss = {total_loss/num_batch}')

        metrics = evaluate_model(model, cifar10_loader['validation'])
        best_models.append((metrics['accuracy'], deepcopy(model)))        
    # _plot_losses(losses)
    val_accuracies = [acc for acc, _ in best_models]
    # TODO: Test best model
    best_model = best_models[np.argmax(val_accuracies)][1]
    metrics = evaluate_model(best_model, cifar10_loader['test'])
    # TODO: Add any information you might want to save for plotting
    logging_dict = {}
    test_accuracy = metrics['accuracy']
    print("best accuracy", test_accuracy)
    #######################
    # END OF YOUR CODE    #
    #######################

    return model, val_accuracies, test_accuracy, logging_dict


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    
    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)

    train(**kwargs)
    # Feel free to add any additional functions, such as plotting of the loss curve here
    