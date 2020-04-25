"""GANITE Codebase.

Reference: Jinsung Yoon, James Jordon, Mihaela van der Schaar, 
"GANITE: Estimation of Individualized Treatment Effects using Generative Adversarial Nets", 
International Conference on Learning Representations (ICLR), 2018.

Paper link: https://openreview.net/forum?id=ByKWUeWA-

Last updated Date: April 25th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

data_loading.py

Note: Load real-world individualized treatment effects estimation datasets

(1) data_loading_twin: Load twins data.
  - Reference: http://data.nber.org/data/linked-birth-infant-death-data-vital-statistics-data.html
"""

# Necessary packages
import numpy as np
from scipy.special import expit


def data_loading_twin(train_rate = 0.8):
  """Load twins data.
  
  Args:
    - train_rate: the ratio of training data
    
  Returns:
    - train_x: features in training data
    - train_t: treatments in training data
    - train_y: observed outcomes in training data
    - train_potential_y: potential outcomes in training data
    - test_x: features in testing data
    - test_potential_y: potential outcomes in testing data      
  """
  
  # Load original data (11400 patients, 30 features, 2 dimensional potential outcomes)
  ori_data = np.loadtxt("data/Twin_data.csv", delimiter=",",skiprows=1)

  # Define features
  x = ori_data[:,:30]
  no, dim = x.shape
    
  # Define potential outcomes
  potential_y = ori_data[:, 30:]
  # Die within 1 year = 1, otherwise = 0
  potential_y = np.array(potential_y < 9999,dtype=float)    
            
  ## Assign treatment
  coef = np.random.uniform(-0.01, 0.01, size = [dim,1])
  prob_temp = expit(np.matmul(x, coef) + np.random.normal(0,0.01, size = [no,1]))
    
  prob_t = prob_temp/(2*np.mean(prob_temp))
  prob_t[prob_t>1] = 1
    
  t = np.random.binomial(1,prob_t,[no,1])
  t = t.reshape([no,])

  ## Define observable outcomes
  y = np.zeros([no,1])
  y = np.transpose(t) * potential_y[:,1] + np.transpose(1-t) * potential_y[:,0]    
  y = np.reshape(np.transpose(y), [no, ])

  ## Train/test division
  idx = np.random.permutation(no)
  train_idx = idx[:int(train_rate * no)]
  test_idx = idx[int(train_rate * no):]
    
  train_x = x[train_idx,:]
  train_t = t[train_idx]
  train_y = y[train_idx]
  train_potential_y = potential_y[train_idx,:]

  test_x = x[test_idx,:]
  test_potential_y = potential_y[test_idx,:]
        
  return train_x, train_t, train_y, train_potential_y, test_x, test_potential_y