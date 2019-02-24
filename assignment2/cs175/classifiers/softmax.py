import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  N = X.shape[0]
  C = W.shape[1]
  for i in range(N): # for i in range(N)
    # logistic loss
    scores = X[i].dot(W)
    # shift the value of scores so that the highest number is 0
    scores -= scores.max()
    correct_score = np.exp(scores[y[i]])
    sum_score = np.sum(np.exp(scores))
    loss += - np.log(correct_score/sum_score)

    # gradient with respect to weights W
    for j in range(C):
      if j == y[i]: 
        dW[:, j] += (-1 + correct_score/sum_score) * X[i]
      else:
        # incorrect prediction
        dW[:, j] += np.exp(scores[j])/sum_score * X[i]

  loss /= N
  loss += reg*np.sum(W*W)
  dW /= N
  dW += 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  N = X.shape[0]
  C = W.shape[1]

  # logistic loss function
  scores = X.dot(W)
  scores -= scores.max()
  scores = np.exp(scores)
  scores_sum_array = np.sum(scores, axis=1)
  correct_prediction_array = scores[range(N), y]
  f = correct_prediction_array / scores_sum_array
  loss = -np.sum(np.log(f))/N + reg*np.sum(W*W)
    
  # gradient
  gradient_matrix = np.divide(scores, scores_sum_array.reshape(N, 1))
  gradient_matrix[range(N), y] = -1 + f
  dW = X.T.dot(gradient_matrix)/N + 2*reg*W
    

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

