import numpy as np
from neural_net import Layer, Model
from activations import Relu, Sigmoid, Linear, Costs

X_train = np.array([[1, 2], [5, 3], [6, 1]])
Y_train = np.array([[1], [0], [1]])

X_test = np.array([[2, 4], [1, 3], [6, 3]])
Y_test = np.array([[0], [1], [1]])

model = Model([Layer(3, Relu), Layer(2, Relu), Layer(1, Sigmoid)])

alpha = 3
epochs = 10
seed = 5

reg_rate = 0.1

train_hist, test_hist, train_acc_hist, test_acc_hist = model.fit(X_train, Y_train, X_test, Y_test, alpha, epochs, seed, reg_rate)

print(f"Training error: {train_hist[epochs-1]}")
print(f"Train accuracy: {train_acc_hist[epochs-1]}")
print(f"Test error: {test_hist[epochs-1]}")
print(f"Test accuracy: {test_acc_hist[epochs-1]}")