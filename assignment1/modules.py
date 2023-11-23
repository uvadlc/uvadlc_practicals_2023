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
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """

    def __init__(self, in_features, out_features, input_layer=False):
        """
        Initializes the parameters of the module.

        Args:
          in_features: size of each input sample
          out_features: size of each output sample
          input_layer: boolean, True if this is the first layer after the input, else False.

        TODO:
        Initialize weight parameters using Kaiming initialization. 
        Initialize biases with zeros.
        Hint: the input_layer argument might be needed for the initialization

        Also, initialize gradients with zeros.
        """

        # Note: For the sake of this assignment, please store the parameters
        # and gradients in this format, otherwise some unit tests might fail.
        self.params = {'weight': None, 'bias': None} # Model parameters
        self.grads = {'weight': None, 'bias': None} # Gradients
        self.cache = {}
        self.input_layer = input_layer
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        def _kaiming(shape, first_layer):
          std = np.sqrt(2.0 / shape[1])
          if first_layer:
            std = 1 / np.sqrt(shape[1])
          w = np.random.normal(0, std, size=shape)
          b = np.zeros((1, shape[0]))
          return w, b
        shape = (out_features, in_features)
        w, b = _kaiming(shape, input_layer)
        self.params['weight'] = w
        self.params['bias'] = b
        self.grads['weight'] = np.zeros(shape)
        self.grads['bias'] = np.zeros((1, out_features))
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        w = self.params['weight']
        b = self.params['bias']
        out = x @ w.T + b
        self.cache['input'] = x
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to
        layer parameters in self.grads['weight'] and self.grads['bias'].
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # W = R^(N x M), input = R^(S x M), dout = R^(S x N)
        # R^(N x M) = Transpose(R^(M x S) * R^(S x N)) 
        d_w = (self.cache['input'].T @ dout).T
        d_b = np.sum(dout, axis=0)
        d_b = np.reshape(d_b, (1, d_b.shape[0]))
        # R^(S x M) = R^(S x N) * R^(N x M)
        dx = dout @ self.params['weight']
        self.grads['weight'] = d_w 
        self.grads['bias'] = d_b
        #######################
        # END OF YOUR CODE    #
        #######################
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.cache = {}
        #######################
        # END OF YOUR CODE    #
        #######################


class ELUModule(object):
    """
    ELU activation module.
    """

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        
        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        def _ELU(x):
          return np.where(x >= 0, x, np.exp(x) - 1)
        self.cache = {'input': x} 
        # X = R^(S, M)
        out = _ELU(x)
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        def _ELU_grad(x):
          return np.where(x >= 0, 1, np.exp(x))
        dx = _ELU_grad(self.cache['input']) * dout
        #######################
        # END OF YOUR CODE    #
        #######################
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.cache = {}
        #######################
        # END OF YOUR CODE    #
        #######################


class SoftMaxModule(object):
    """
    Softmax activation module.
    """
    def __init__(self):
        self.cache = {}

    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        def _softmax_stablized(x):
            e_z = np.exp(x - np.max(x, axis=1, keepdims=True))
            return e_z / e_z.sum(axis=1, keepdims=True)
        out = _softmax_stablized(x)
        self.cache['out'] = out
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous modul
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # out = R^(S x M)
        out = self.cache['out']
        # jacobian = R^(M x M)
        # jacobian = diag(out) - out*out.T
        # algo 1
        # diag_batch = np.einsum('ij,jk->jk', out, np.eye(out.shape[1]))
        # outter_batch = np.einsum('ij,ik->jk', out, out)
        # jacobian_batch = diag_batch - outter_batch
        # dx_0 = dout @ jacobian_batch
        # algo 2
        # diag_batch = np.einsum('ij,jk->ijk', out, np.eye(out.shape[1]))
        # outter_batch = np.einsum('ij,ik->ijk', out, out)
        # jacobian_batch = diag_batch - outter_batch
        # dx = np.zeros(dout.shape)
        # for i in range(out.shape[0]):
        #     dx += jacobian_batch[i] @ dout[i]
        # dout = R^(S x M)
        # dx = dout * jacobian
        # dx = R^(S x M) * R^(M x M) = R^(S x M)
        # ones = np.ones(out.shape[1])
        # v = dout * out @ ones.T
        # v = np.reshape(v, (dout.shape[0], 1))
        # dx_2 = out * dout - v
        # print(dx_0[0], '\n', dx[0], '\n',  dx_2[0])
        dx = out * (dout - (dout * out).sum(axis=1)[:, None])
        #######################
        # END OF YOUR CODE    #
        #######################

        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.cache = {}
        #######################
        # END OF YOUR CODE    #
        #######################


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """

    def _one_hot_encoded(self, x, y):
        ohe = np.zeros(x.shape)
        ohe[np.arange(x.shape[0]), y] = 1
        return ohe

    def forward(self, x, y):
        """
        Forward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss

        TODO:
        Implement forward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # X=R^(SxN) Y=R^(S)
        one_hot_encoded = self._one_hot_encoded(x, y)
        out = np.mean(-np.einsum('ik,ik->i', one_hot_encoded, np.log(x)))
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, x, y):
        """
        Backward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # dL/dx = -1/S*Y/X
        one_hot_encoded_labels = self._one_hot_encoded(x, y)
        sample_size = len(y)
        dx = one_hot_encoded_labels/x
        dx /= -sample_size
        #######################
        # END OF YOUR CODE    #
        #######################

        return dx