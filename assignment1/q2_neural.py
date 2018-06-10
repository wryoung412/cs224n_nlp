#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax, softmax_grad
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def forward_backward_prop(X, labels, params, dimensions, debug=False):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    the backward propagation for the gradients for all parameters.

    Notice the gradients computed here are different from the gradients in
    the assignment sheet: they are w.r.t. weights, not inputs.

    Arguments:
    X -- M x Dx matrix, where each row is a training example x.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    # Note: compute cost based on `sum` not `mean`.
    ### YOUR CODE HERE: forward propagation
    z1 = np.matmul(X, W1) + b1
    h = sigmoid(z1)
    # This is misleading. b2 is only a row vector, Broadcasting did the magic.
    # The real formula is
    # z2 = np.matmul(h, W2) + ones(n, 1) * b2
    #
    # When pretending z2 = np.matmul(h, W2) + eye(n) * b2, dim(gradb2) = M * dim(b2).
    # So gradW1 is correct, but gradb1 messed up the padding...
    z2 = np.matmul(h, W2) + b2
    y = softmax(z2)
    cost = -np.sum(labels * np.log(y))
    ### END YOUR CODE

    ### YOUR CODE HERE: backward propagation
    grady = - labels / y
    gradz2 = softmax_grad(y, grady)
    # softmax_grad is tricky and requires grady.
    # When grady is from the cross entropy loss with unique labels, the gradient
    # has a scalar form.
    assert np.all(np.sum(labels, 1) == 1)
    assert np.allclose(gradz2, y - labels, rtol=1e-05, atol=1e-06)
    # Suppose z = f(y), y = g(x). Then dz/dx = dy/dx * dz/dy.
    # Notice the multiplication order.
    gradW2 = np.matmul(h.T, gradz2)
    gradb2 = np.sum(gradz2, 0)
    
    gradh = np.matmul(gradz2, W2.T)
    gradz1 = sigmoid_grad(h) * gradh
    gradW1 = np.matmul(X.T, gradz1)
    gradb1 = np.sum(gradz1, 0)
    ### END YOUR CODE

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    if debug:
        print('params', params.shape, params)
        print('grad', grad.shape, grad)

    return cost, grad

def forward_backward_prop_ref(data, labels, params, dimensions, debug=False):
    """
    Forward and backward propagation for a two-layer sigmoidal network
    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.
    Arguments:
    data -- M x Dx matrix, where each row is a training example.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation
    h = sigmoid(np.dot(data,W1) + b1)
    yhat = softmax(np.dot(h,W2) + b2)
    ### END YOUR CODE

    ### YOUR CODE HERE: backward propagation
    cost = np.sum(-np.log(yhat[labels==1]))

    d3 = (yhat - labels)
    gradW2 = np.dot(h.T, d3)
    gradb2 = np.sum(d3,0,keepdims=True)
    if debug:
        print('ref gradW2', gradW2)
    if debug:
        print('ref gradb2', gradb2)
    

    dh = np.dot(d3,W2.T)
    grad_h = sigmoid_grad(h) * dh

    gradW1 = np.dot(data.T,grad_h)
    gradb1 = np.sum(grad_h,0)
    ### END YOUR CODE

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad



def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    ## debug with small dimensions
    # N = 2
    # dimensions = [3, 1, 2]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params:
                    forward_backward_prop(data, labels, params, dimensions), params)

    # gradcheck_naive(lambda params, debug=False:
    #                 forward_backward_prop_ref(data, labels, params, dimensions, debug), params)



def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    print "No test added..."
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
