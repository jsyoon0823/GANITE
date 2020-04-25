"""GANITE Codebase.

Reference: Jinsung Yoon, James Jordon, Mihaela van der Schaar, 
"GANITE: Estimation of Individualized Treatment Effects using Generative Adversarial Nets", 
International Conference on Learning Representations (ICLR), 2018.

Paper link: https://openreview.net/forum?id=ByKWUeWA-

Last updated Date: April 25th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

ganite.py

Note: GANITE module.
"""

# Necessary packages
import tensorflow as tf
import numpy as np
from utils import xavier_init, batch_generator


def ganite (train_x, train_t, train_y, test_x, parameters):
  """GANITE module.
  
  Args:
    - train_x: features in training data
    - train_t: treatments in training data
    - train_y: observed outcomes in training data
    - test_x: features in testing data
    - parameters: GANITE network parameters
      - h_dim: hidden dimensions
      - batch_size: the number of samples in each batch
      - iterations: the number of iterations for training
      - alpha: hyper-parameter to adjust the loss importance
      
  Returns:
    - test_y_hat: estimated potential outcome for testing set
  """
  # Parameters 
  h_dim = parameters['h_dim']
  batch_size = parameters['batch_size']
  iterations = parameters['iteration']
  alpha = parameters['alpha']
  
  no, dim = train_x.shape

  # Reset graph
  tf.reset_default_graph()

  ## 1. Placeholder
  # 1.1. Feature (X)
  X = tf.placeholder(tf.float32, shape = [None, dim])
  # 1.2. Treatment (T)
  T = tf.placeholder(tf.float32, shape = [None, 1])
  # 1.3. Outcome (Y)
  Y = tf.placeholder(tf.float32, shape = [None, 1])

  ## 2. Variables
  # 2.1 Generator
  G_W1 = tf.Variable(xavier_init([(dim+2), h_dim])) # Inputs: X + Treatment + Factual outcome
  G_b1 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  G_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
  G_b2 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  # Multi-task outputs for increasing the flexibility of the generator 
  G_W31 = tf.Variable(xavier_init([h_dim, h_dim]))
  G_b31 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  G_W32 = tf.Variable(xavier_init([h_dim, 1]))
  G_b32 = tf.Variable(tf.zeros(shape = [1])) # Output: Estimated outcome when t = 0
  
  G_W41 = tf.Variable(xavier_init([h_dim, h_dim]))
  G_b41 = tf.Variable(tf.zeros(shape = [h_dim])) 
  
  G_W42 = tf.Variable(xavier_init([h_dim, 1]))
  G_b42 = tf.Variable(tf.zeros(shape = [1])) # Output: Estimated outcome when t = 1
  
  # Generator variables
  theta_G = [G_W1, G_W2, G_W31, G_W32, G_W41, G_W42, G_b1, G_b2, G_b31, G_b32, G_b41, G_b42]
  
  # 2.2 Discriminator
  D_W1 = tf.Variable(xavier_init([(dim+2), h_dim])) # Inputs: X + Factual outcomes + Estimated counterfactual outcomes
  D_b1 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  D_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
  D_b2 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  D_W3 = tf.Variable(xavier_init([h_dim, 1]))
  D_b3 = tf.Variable(tf.zeros(shape = [1]))
  
  # Discriminator variables
  theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]
  
  # 2.3 Inference network
  I_W1 = tf.Variable(xavier_init([(dim), h_dim])) # Inputs: X
  I_b1 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  I_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
  I_b2 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  # Multi-task outputs for increasing the flexibility of the inference network
  I_W31 = tf.Variable(xavier_init([h_dim, h_dim]))
  I_b31 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  I_W32 = tf.Variable(xavier_init([h_dim, 1]))
  I_b32 = tf.Variable(tf.zeros(shape = [1])) # Output: Estimated outcome when t = 0
  
  I_W41 = tf.Variable(xavier_init([h_dim, h_dim]))
  I_b41 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  I_W42 = tf.Variable(xavier_init([h_dim, 1]))
  I_b42 = tf.Variable(tf.zeros(shape = [1])) # Output: Estimated outcome when t = 1
  
  # Inference network variables
  theta_I = [I_W1, I_W2, I_W31, I_W32, I_W41, I_W42, I_b1, I_b2, I_b31, I_b32, I_b41, I_b42]

  ## 3. Definitions of generator, discriminator and inference networks
  # 3.1 Generator
  def generator(x, t, y):
    """Generator function.
    
    Args:
      - x: features
      - t: treatments
      - y: observed labels
      
    Returns:
      - G_logit: estimated potential outcomes
    """
    # Concatenate feature, treatments, and observed labels as input
    inputs = tf.concat(axis = 1, values = [x,t,y])
    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
      
    # Estimated outcome if t = 0
    G_h31 = tf.nn.relu(tf.matmul(G_h2, G_W31) + G_b31)
    G_logit1 = tf.matmul(G_h31, G_W32) + G_b32
      
    # Estimated outcome if t = 1
    G_h41 = tf.nn.relu(tf.matmul(G_h2, G_W41) + G_b41)
    G_logit2 = tf.matmul(G_h41, G_W42) + G_b42
      
    G_logit = tf.concat(axis = 1, values = [G_logit1, G_logit2])
    return G_logit
      
  # 3.2. Discriminator
  def discriminator(x, t, y, hat_y):
    """Discriminator function.
    
    Args:
      - x: features
      - t: treatments
      - y: observed labels
      - hat_y: estimated counterfactuals
      
    Returns:
      - D_logit: estimated potential outcomes
    """
    # Concatenate factual & counterfactual outcomes
    input0 = (1.-t) * y + t * tf.reshape(hat_y[:,0], [-1,1]) # if t = 0
    input1 = t * y + (1.-t) * tf.reshape(hat_y[:,1], [-1,1]) # if t = 1
      
    inputs = tf.concat(axis = 1, values = [x, input0,input1])
      
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
    D_logit = tf.matmul(D_h2, D_W3) + D_b3
    return D_logit
  
  # 3.3. Inference Nets
  def inference(x):
    """Inference function.
    
    Args:
      - x: features
      
    Returns:
      - I_logit: estimated potential outcomes
    """
    I_h1 = tf.nn.relu(tf.matmul(x, I_W1) + I_b1)
    I_h2 = tf.nn.relu(tf.matmul(I_h1, I_W2) + I_b2)
      
    # Estimated outcome if t = 0
    I_h31 = tf.nn.relu(tf.matmul(I_h2, I_W31) + I_b31)
    I_logit1 = tf.matmul(I_h31, I_W32) + I_b32
      
    # Estimated outcome if t = 1
    I_h41 = tf.nn.relu(tf.matmul(I_h2, I_W41) + I_b41)
    I_logit2 = tf.matmul(I_h41, I_W42) + I_b42
      
    I_logit = tf.concat(axis = 1, values = [I_logit1, I_logit2])
    return I_logit

  ## Structure
  # 1. Generator
  Y_tilde_logit = generator(X, T, Y)
  Y_tilde = tf.nn.sigmoid(Y_tilde_logit)
  
  # 2. Discriminator
  D_logit = discriminator(X,T,Y,Y_tilde)

  # 3. Inference network
  Y_hat_logit = inference(X)
  Y_hat = tf.nn.sigmoid(Y_hat_logit)

  ## Loss functions
  # 1. Discriminator loss
  D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = T, logits = D_logit )) 
  
  # 2. Generator loss
  G_loss_GAN = -D_loss 
  G_loss_Factual = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
      labels = Y, logits = (T * tf.reshape(Y_tilde_logit[:,1],[-1,1]) + \
                            (1. - T) * tf.reshape(Y_tilde_logit[:,0],[-1,1]) ))) 
  
  G_loss = G_loss_Factual + alpha * G_loss_GAN
  
  # 3. Inference loss
  I_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
      labels = (T) * Y + (1-T) * tf.reshape(Y_tilde[:,1],[-1,1]), logits = tf.reshape(Y_hat_logit[:,1],[-1,1]) )) 
  I_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
      labels = (1-T) * Y +  (T) * tf.reshape(Y_tilde[:,0],[-1,1]), logits = tf.reshape(Y_hat_logit[:,0],[-1,1]) )) 
  
  I_loss = I_loss1 + I_loss2

  ## Solver
  G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)
  D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
  I_solver = tf.train.AdamOptimizer().minimize(I_loss, var_list=theta_I)

  ## GANITE training
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
      
  print('Start training Generator and Discriminator')
  # 1. Train Generator and Discriminator
  for it in range(iterations):
    
    for _ in range(2):
      # Discriminator training
      X_mb, T_mb, Y_mb = batch_generator(train_x, train_t, train_y, batch_size)      
      _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict = {X: X_mb, T: T_mb, Y: Y_mb})
      
    # Generator traininig
    X_mb, T_mb, Y_mb = batch_generator(train_x, train_t, train_y, batch_size)        
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict = {X: X_mb, T: T_mb, Y: Y_mb})
          
    # Check point
    if it % 1000 == 0:
      print('Iteration: ' + str(it) + '/' + str(iterations) + ', D loss: ' + \
            str(np.round(D_loss_curr, 4)) + ', G loss: ' + str(np.round(G_loss_curr, 4)))
      
  print('Start training Inference network')
  # 2. Train Inference network
  for it in range(iterations):
    
    X_mb, T_mb, Y_mb = batch_generator(train_x, train_t, train_y, batch_size) 
    _, I_loss_curr = sess.run([I_solver, I_loss], feed_dict = {X: X_mb, T: T_mb, Y: Y_mb})    
  
    # Check point
    if it % 1000 == 0:      
      print('Iteration: ' + str(it) + '/' + str(iterations) + 
            ', I loss: ' + str(np.round(I_loss_curr, 4)))
            
  ## Generate the potential outcomes
  test_y_hat = sess.run(Y_hat, feed_dict = {X: test_x})
  
  return test_y_hat        