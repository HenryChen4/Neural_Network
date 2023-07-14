import numpy as np
from activations import Sigmoid, Linear, Relu, Costs
import matplotlib.pyplot as plt

class Layer:
    def __init__(self, units, activation):
        self.activation = activation
        self.units = units
        self.neurons = np.array([])
        self.W_l = np.array([])
        self.B_l = np.array([])
    
    def initialize(self, prev_layer_units, seed=10):
        np.random.seed(seed)
        W_shape = (self.units, prev_layer_units)
        B_shape = (self.units, 1)
        self.W_l = np.random.uniform(-1., 1., W_shape)
        self.B_l = np.random.uniform(-1., 1., B_shape)

    def feed_forward(self, a_in):
        Z = np.matmul(self.W_l, a_in) + self.B_l
        self.neurons = self.activation.g(Z)
    
    def back_prop(self, del_J_z, prev_layer, alpha):
        del_J_w = np.matmul(prev_layer.neurons, del_J_z.T)
        del_J_b = del_J_z

        self.W_l -= alpha * del_J_w.T
        self.B_l -= alpha * del_J_b

        return np.multiply(np.matmul(self.W_l.T, del_J_z), prev_layer.activation.del_g_z(prev_layer.neurons))

    def summarize(self):
        print("Weights:")
        print(self.W_l)
        print("===============================")
        print("Biases:")
        print(self.B_l)
        print("===============================")
        print("Activations:")
        print(self.neurons)

class Model:
    def __init__(self, layer_arr):
        self.layers = np.array(layer_arr) 

    def initialize(self, n, seed=10):
        for l in range(self.layers.shape[0]-1, -1, -1):
            prev_units = n if l == 0 else self.layers[l-1].units
            self.layers[l].initialize(prev_units, seed)

    def forward_propagate(self, x_in):
        a_in = x_in
        for l in range(self.layers.shape[0]):
            self.layers[l].feed_forward(a_in)
            a_in = self.layers[l].neurons
        return a_in

    def back_propagate(self, m, x_in, y_out, alpha):
        a_out = self.layers[-1].neurons
        del_J_g = self.layers[-1].activation.del_J_g(a_out, y_out, m)
        del_g_z = self.layers[-1].activation.del_g_z(a_out)
        del_J_z = np.multiply(del_J_g, del_g_z)

        x_in_layer = Layer(x_in.shape[0], Linear)
        x_in_layer.neurons = x_in

        for l in range(self.layers.shape[0]-1, -1, -1):
            prev_layer = x_in_layer if l == 0 else self.layers[l-1]
            del_J_z = self.layers[l].back_prop(del_J_z, prev_layer, alpha)

    def fit(self, X_train, Y_train, alpha, epochs, seed=10):
        n = X_train.shape[1]
        m = X_train.shape[0]
        self.initialize(n, seed)
    
        J_hist = []

        for c in range(epochs):
            all_y_hat = []
            J = 0
            for i in range(X_train.shape[0]):
                y_hat = self.forward_propagate(X_train[i])
                all_y_hat.append(y_hat)
                self.back_propagate(m, X_train[i], Y_train[i], alpha)
            J = Costs.sigmoid_cost(all_y_hat, Y_train)
            J_hist.append(J)
            print(f"Epoch {c} complete!")
        
        return J_hist

    def summarize(self):
        for layer in self.layers:
            layer.summarize()
            print('\n')