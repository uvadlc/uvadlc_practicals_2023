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

import torch

from torchvision.datasets import CIFAR100
from torch.utils.data import random_split
from torchvision import transforms

class AddGaussianNoise(torch.nn.Module):
    def __init__(self, mean=0., std=0.1, always_apply=False):
        self.mean = mean
        self.std = std
        self.always_apply = always_apply

    def __call__(self, img):
        #######################
        # PUT YOUR CODE HERE  #
        #######################

        # TODO: Add Gaussian noise to an image.

        # Hints:
        # - You can use torch.randn() to sample z ~ N(0, 1).
        # - Then, you can transform z s.t. it is sampled from N(self.mean, self.std)
        # - Finally, you can add the noise to the image.

        raise NotImplementedError
        #######################
        # END OF YOUR CODE    #
        #######################


def add_augmentation(augmentation_name, transform_list):
    """
    Adds an augmentation transform to the list.
    Args:
        augmentation_name: Name of the augmentation to use.
        transform_list: List of transforms to add the augmentation to.

    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Create a new transformation based on the augmentation_name.
    pass

    # Add the new transformation to the list.
    pass

    #######################
    # END OF YOUR CODE    #
    #######################


def get_train_validation_set(data_dir, validation_size=5000, augmentation_name=None):
    """
    Returns the training and validation set of CIFAR100.

    Args:
        data_dir: Directory where the data should be stored.
        validation_size: Size of the validation size
        augmentation_name: The name of the augmentation to use.

    Returns:
        train_dataset: Training dataset of CIFAR100
        val_dataset: Validation dataset of CIFAR100
    """

    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    train_transform = [transforms.Resize((224, 224)),
                       transforms.ToTensor(),
                       transforms.Normalize(mean, std)]
    if augmentation_name is not None:
        add_augmentation(augmentation_name, train_transform)
    train_transform = transforms.Compose(train_transform)

    val_transform = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std)])

    # We need to load the dataset twice because we want to use them with different transformations
    train_dataset = CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
    val_dataset = CIFAR100(root=data_dir, train=True, download=True, transform=val_transform)

    # Subsample the validation set from the train set
    if not 0 <= validation_size <= len(train_dataset):
        raise ValueError("Validation size should be between 0 and {0}. Received: {1}.".format(
            len(train_dataset), validation_size))

    train_dataset, _ = random_split(train_dataset,
                                    lengths=[len(train_dataset) - validation_size, validation_size],
                                    generator=torch.Generator().manual_seed(42))
    _, val_dataset = random_split(val_dataset,
                                  lengths=[len(val_dataset) - validation_size, validation_size],
                                  generator=torch.Generator().manual_seed(42))

    return train_dataset, val_dataset


def get_test_set(data_dir, test_noise):
    """
    Returns the test dataset of CIFAR100.

    Args:
        data_dir: Directory where the data should be stored
        test_noise: Whether to add Gaussian noise to the test set.
    Returns:
        test_dataset: The test dataset of CIFAR100.
    """

    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    test_transform = [transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std)]
    if test_noise:
        add_augmentation('test_noise', test_transform)
    test_transform = transforms.Compose(test_transform)

    test_dataset = CIFAR100(root=data_dir, train=False, download=True, transform=test_transform)
    return test_dataset
