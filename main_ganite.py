"""GANITE Codebase.

Reference: Jinsung Yoon, James Jordon, Mihaela van der Schaar, 
"GANITE: Estimation of Individualized Treatment Effects using Generative Adversarial Nets", 
International Conference on Learning Representations (ICLR), 2018.

Paper link: https://openreview.net/forum?id=ByKWUeWA-

Last updated Date: April 25th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

main_ganite.py

(1) Import data
(2) Train GANITE & Estimate potential outcomes
(3) Evaluate the performances
  - PEHE
  - ATE
"""

## Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# 1. GANITE model
from ganite import ganite
# 2. Data loading
from data_loading import data_loading_twin
# 3. Metrics
from metrics import PEHE, ATE


def main (args):
  """Main function for GANITE experiments.
  
  Args:
    - data_name: twin
    - train_rate: ratio of training data
    - Network parameters (should be optimized for different datasets)
      - h_dim: hidden dimensions
      - iteration: number of training iterations
      - batch_size: the number of samples in each batch
      - alpha: hyper-parameter to adjust the loss importance
  
  Returns:
    - test_y_hat: estimated potential outcomes
    - metric_results: performance on testing data
  """
  ## Data loading
  train_x, train_t, train_y, train_potential_y, test_x, test_potential_y = \
  data_loading_twin(args.train_rate)
    
  print(args.data_name + ' dataset is ready.')
    
  ## Potential outcome estimations by GANITE
  # Set newtork parameters
  parameters = dict()    
  parameters['h_dim'] = args.h_dim
  parameters['iteration'] = args.iteration
  parameters['batch_size'] = args.batch_size
  parameters['alpha'] = args.alpha
      
  test_y_hat = ganite(train_x, train_t, train_y, test_x, parameters)   
  print('Finish GANITE training and potential outcome estimations')
  
  ## Performance metrics   
  # Output initialization
  metric_results = dict()
  
  # 1. PEHE
  test_PEHE = PEHE(test_potential_y, test_y_hat)
  metric_results['PEHE'] = np.round(test_PEHE, 4)
    
  # 2. ATE
  test_ATE = ATE(test_potential_y, test_y_hat)  
  metric_results['ATE'] = np.round(test_ATE, 4)
  
  ## Print performance metrics on testing data
  print(metric_results)

  return test_y_hat, metric_results


if __name__ == '__main__':  
  
  # Inputs for the main function
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_name',
      choices=['twin'],
      default='twin',
      type=str)
  parser.add_argument(
      '--train_rate',
      help='the ratio of training data',
      default=0.8,
      type=float)
  parser.add_argument(
      '--h_dim',
      help='hidden state dimensions (should be optimized)',
      default=30,
      type=int)
  parser.add_argument(
      '--iteration',
      help='Training iterations (should be optimized)',
      default=10000,
      type=int)
  parser.add_argument(
      '--batch_size',
      help='the number of samples in mini-batch (should be optimized)',
      default=256,
      type=int)
  parser.add_argument(
      '--alpha',
      help='hyper-parameter to adjust the loss importance (should be optimized)',
      default=1,
      type=int)
  
  args = parser.parse_args() 
  
  # Calls main function  
  test_y_hat, metrics = main(args)