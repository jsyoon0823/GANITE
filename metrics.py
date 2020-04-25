"""GANITE Codebase.

Reference: Jinsung Yoon, James Jordon, Mihaela van der Schaar, 
"GANITE: Estimation of Individualized Treatment Effects using Generative Adversarial Nets", 
International Conference on Learning Representations (ICLR), 2018.

Paper link: https://openreview.net/forum?id=ByKWUeWA-

Last updated Date: April 25th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

metrics.py

Note: Metric functions for GANITE.
Reference: Jennifer L Hill, "Bayesian nonparametric modeling for causal inference", Journal of Computational and Graphical Statistics, 2011.

(1) PEHE: Precision in Estimation of Heterogeneous Effect
(2) ATE: Average Treatment Effect
"""

# Necessary packages
import numpy as np


def PEHE(y, y_hat):
  """Compute Precision in Estimation of Heterogeneous Effect.
  
  Args:
    - y: potential outcomes
    - y_hat: estimated potential outcomes
    
  Returns:
    - PEHE_val: computed PEHE
  """
  PEHE_val = np.mean( np.abs( (y[:,1] - y[:,0]) - (y_hat[:,1] - y_hat[:,0]) ))
  return PEHE_val

def ATE(y, y_hat):
  """Compute Average Treatment Effect.
  
  Args:
    - y: potential outcomes
    - y_hat: estimated potential outcomes
    
  Returns:
    - ATE_val: computed ATE
  """
  ATE_val = np.abs(np.mean(y[:,1] - y[:,0]) - np.mean(y_hat[:,1] - y_hat[:,0]))
  return ATE_val