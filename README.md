# Codebase for "GANITE"

Authors: Jinsung Yoon, James Jordon, Mihaela van der Schaar

Reference: Jinsung Yoon, James Jordon, Mihaela van der Schaar, 
"GANITE: Estimation of Individualized Treatment Effects using Generative Adversarial Nets", 
International Conference on Learning Representations (ICLR), 2018.

Paper link: https://openreview.net/forum?id=ByKWUeWA-

Contact: jsyoon0823@gmail.com

This directory contains implementations of GANITE framework for individualized treatment effect estimations
using a real-world dataset.

-   Twin data: http://data.nber.org/data/linked-birth-infant-death-data-vital-statistics-data.html

To run the pipeline for training and evaluation on GANITE framwork, simply run 
python3 -m main_ganite.py.


### Code explanation

(1) data_loading.py
- Transform raw twins data to preprocessed ITE data (X, T, Y, Potential Y)

(2) metrics.py
  (a) PEHE
  - Precision in Estimation of Heterogeneous Effect
  (b) ATE
  - Average Treatment Effect

(3) ganite.py
- Use observed features, treatments and factual outcomes to estimate the potential outcomes

(4) main_ganite.py
- Report PEHE and ATI for the twin dataset with GANITE

(5) utils.py
- Some utility functions for GANITE.

### Command inputs:

-   data_name: twin
-   train_rate: the ratio of training data
-   h_dim: hidden dimensions
-   iterations: number of training iterations
-   batch_size: the number of samples in each batch
-   alpha: hyper-parameter to adjust the loss importance

Note that network parameters should be optimized.

### Example command

```shell
$ python3 main_ganite.py --data_name twin --train_rate 0.8 
--h_dim 30 --iteration 10000 --batch_size 256 --alpha 1 
```

### Outputs

-   test_y_hat: estimated potential outcomes
-   metric_results: PEHE and ATE