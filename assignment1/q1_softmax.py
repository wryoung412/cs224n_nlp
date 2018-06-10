import numpy as np


def softmax(x):
    """Compute the softmax function for each row of the input x.

    It is crucial that this function is optimized for speed because
    it will be used frequently in later code. You might find numpy
    functions np.exp, np.sum, np.reshape, np.max, and numpy
    broadcasting useful for this task.

    Numpy broadcasting documentation:
    http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html

    You should also make sure that your code works for a single
    D-dimensional vector (treat the vector as a single row) and
    for N x D matrices. This may be useful for testing later. Also,
    make sure that the dimensions of the output match the input.

    You must implement the optimization in problem 1(a) of the
    written assignment!

    Arguments:
    x -- A D dimensional vector or N x D dimensional numpy matrix.

    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape

    if len(x.shape) > 1:
        # Matrix
        assert len(x.shape) == 2
        x = x - np.max(x, 1).reshape(-1, 1)
        ### YOUR CODE HERE
        z = np.exp(x)
        s = np.sum(z, 1).reshape(-1, 1)
        ### END YOUR CODE
    else:
        # Vector
        x = x - np.max(x)
        ### YOUR CODE HERE
        z = np.exp(x)
        s = np.sum(z)
        ### END YOUR CODE

    # assign back
    x = z / s

    assert x.shape == orig_shape
    return x

# TODO
# 1. Add tests.
# 2. Make the iteration more efficient. 
def softmax_grad(y, delta):
    """Compute the gradient for the softmax function. 

    Note that for the implementation, the input y should be the softmax value 
    of your original input x. 
    
    Each row represents one sample. For each row y_k and delta_k, 
    dy_kj / dx_ki = -y_ki * y_kj
    dy_ki / dx_ki = y_ki * sum_{j != i} y_kj
    gradx_k = dy_k / dx_k * delta_k

    This is a trick case. Let y = softmax(x). Note that softmax applies to each row.
    1. x to y is an all-to-all relationship.
    2. The gradient depends on x. 

    sigmoid(x): one-to-one.
    Wx: the gradient does not depend on x.
    """
    assert len(y.shape) == 2
    n = y.shape[0]
    d = y.shape[1]

    L = []
    for i in xrange(n):
        D = -np.outer(y[i], y[i])
        # Assume v is an 1-d array. The following are equivalent:
        # np.outer(v, v)
        # np.matmul(v.reshape(-1, 1), v.reshape(1, -1))
        # np.dot(v.reshape(-1, 1), v.reshape(1, -1))
        #
        # The following is the inner product:
        # np.matmul(np.transpose(v), v) 
        row, col = np.diag_indices(d)
        D[row, col] = np.zeros(d)
        D[row, col] = -np.sum(D, 1)
        
        L.append(np.matmul(delta[i], D).reshape(1, -1))

    return np.concatenate(L)

def test_softmax_basic():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print "Running basic tests..."
    test1 = softmax(np.array([1,2]))
    print test1
    ans1 = np.array([0.26894142,  0.73105858])
    assert np.allclose(test1, ans1, rtol=1e-05, atol=1e-06)

    test2 = softmax(np.array([[1001,1002],[3,4]]))
    print test2
    ans2 = np.array([
        [0.26894142, 0.73105858],
        [0.26894142, 0.73105858]])
    assert np.allclose(test2, ans2, rtol=1e-05, atol=1e-06)

    test3 = softmax(np.array([[-1001,-1002]]))
    print test3
    ans3 = np.array([0.73105858, 0.26894142])
    assert np.allclose(test3, ans3, rtol=1e-05, atol=1e-06)

    print "You should be able to verify these results by hand!\n"


def test_softmax():
    """
    Use this space to test your softmax implementation by running:
        python q1_softmax.py
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print "Running your tests..."
    ### YOUR CODE HERE
    test1 = softmax(np.array([[1, 2], [3, 5], [6, 9]]))
    print test1
    ans1 = np.array([[ 0.26894142, 0.73105858],
                     [ 0.11920292, 0.88079708],
                     [ 0.04742587, 0.95257413]])
    assert np.allclose(test1, ans1, rtol=1e-05, atol=1e-06)
    ### END YOUR CODE


if __name__ == "__main__":
    test_softmax_basic()
    test_softmax()
