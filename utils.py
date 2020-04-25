"""GANITE Codebase.

Reference: Jinsung Yoon, James Jordon, Mihaela van der Schaar, 
"GANITE: Estimation of Individualized Treatment Effects using Generative Adversarial Nets", 
International Conference on Learning Representations (ICLR), 2018.

Paper link: https://openreview.net/forum?id=ByKWUeWA-

Last updated Date: April 25th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

utils.py

Note: Utility functions for GANITE.

(1) xavier_init: Xavier initialization function
(2) batch_generator: generate mini-batch with x, t, and y
"""

# Necessary packages
import tensorflow as tf
import numpy as np


def xavier_init(size):
  """Xavier initialization function.
  
  Args:
    - size: input data dimension
  """
  in_dim = size[0]
  xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
  return tf.random_normal(shape = size, stddev = xavier_stddev)


# Mini-batch generation
def batch_generator(x, t, y, size):
  """ Generate mini-batch with x, t, and y.
  
  Args:
    - x: features
    - t: treatments
    - y: observed labels
    - size: mini batch size
    
  Returns:
    - X_mb: mini-batch features
    - T_mb: mini-batch treatments
    - Y_mb: mini-batch observed labels
  """
  batch_idx = np.random.randint(0, x.shape[0], size)
  
  X_mb = x[batch_idx, :]
  T_mb = np.reshape(t[batch_idx], [size,1]) 
  Y_mb = np.reshape(y[batch_idx], [size,1])   
  return X_mb, T_mb, Y_mb