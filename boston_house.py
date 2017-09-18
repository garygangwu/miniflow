import numpy as np
from sklearn.datasets import load_boston
from sklearn.utils import shuffle, resample
from miniflow import *

def fit(X_train, y_train):
  n_features = X_train.shape[1]
  n_hidden = 10
  W1_ = np.random.randn(n_features, n_hidden)
  b1_ = np.zeros(n_hidden)
  W2_ = np.random.randn(n_hidden, 1)
  b2_ = np.zeros(1)

  # Neural network
  X, y = Input(), Input()
  W1, b1 = Input(), Input()
  W2, b2 = Input(), Input()

  l1 = Linear(X, W1, b1)
  s1 = Sigmoid(l1)
  l2 = Linear(s1, W2, b2)
  cost = MSE(y, l2)

  feed_dict = {
    X: X_train,
    y: y_train,
    W1: W1_,
    b1: b1_,
    W2: W2_,
    b2: b2_
  }

  epochs = 1000
  # Total number of examples
  m = X_train.shape[0]
  batch_size = 11
  steps_per_epoch = m // batch_size

  graph = topological_sort(feed_dict)
  trainables = [W1, b1, W2, b2]

  print("Total number of examples = {}".format(m))

  # Step 4
  for i in range(epochs):
    mse = 0
    me = 0
    for j in range(steps_per_epoch):
      # Step 1
      # Randomly sample a batch of examples
      X_batch, y_batch = resample(X_train, y_train, n_samples=batch_size)

      # Reset value of X and y Inputs
      X.value = X_batch
      y.value = y_batch

      # Step 2
      forward_and_backward(graph)

      # Step 3
      sgd_update(trainables)

      mse += cost.value
      me += np.mean(abs(l2.value.reshape(1, -1)[0] - y_batch))

    print("Epoch: {}, MSE: {:.3f}, ME: {:.3f}".format(i+1, mse/steps_per_epoch, me/steps_per_epoch))

  return {
    'W1': W1.value,
    'b1': b1.value,
    'W2': W2.value,
    'b2': b2.value
  }


# Load data
data = load_boston()
X = data['data'][:-100]
y_train = data['target'][:-100]
X_test = data['data'][-100:]
y_test = data['target'][-100:]

# Normalize data
X_train = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Training
flow_params = fit(X_train, y_train)

# Scoring based on the neural network
X_node = Input()
W1, b1 = Input(), Input()
W2, b2 = Input(), Input()

l1 = Linear(X_node, W1, b1)
s1 = Sigmoid(l1)
l2 = Linear(s1, W2, b2)

# Normalize data
X_test_input = (X_test - np.mean(X, axis=0)) / np.std(X, axis=0)
feed_dict = {
  X_node: X_test_input,
  W1: flow_params['W1'],
  b1: flow_params['b1'],
  W2: flow_params['W2'],
  b2: flow_params['b2']
}
graph = topological_sort(feed_dict)
forward(graph)
results = l2.value.reshape(1, -1)[0]

print "Estimated results:"
print results
print "Targets:"
print y_test
print "ME: %f" % (np.mean(abs(results - y_test)))
print "MSE: %f" % (np.mean((results - y_test) ** 2))

