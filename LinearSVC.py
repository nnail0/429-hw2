# LinearSVC

"""
process:
init weights to low and bias to 0. 
start with an initial separation strip?



"""


import numpy as np 
import pandas as pd

class LinearSVC:

    def __init__(self, eta = 0.01, n_iter = 50, random_state = 1):
        self.eta = eta
        self.n_iter = n_iter 
        self.random_state = random_state


    def fit(self, X, y, C):

        # absorb the bias
        n = X.shape[0] #number of samples
        m = X.shape[1] #number of features 
        x_0 = np.ones((n,1)) 
        X = np.hstack((x_0,X))

        rand_gen = np.random.RandomState(self.random_state)
        self.w_ = rand_gen.normal(loc = 0.0, scale = 0.01, size = 1 + m)

        self.losses_ = []

        for _ in range(self.n_iter):
            # get the input
            indiv_loss = []
            for x_i, target in zip(X, y):
                # compute individual hinge losses
                loss = np.max(0, 1 - (x_i * target))
            
            sum = 0
            for i in range(n):
                sum += indiv_loss[i] + (0.5 * np.dot(self.w_, self.w_))

            total_loss = C * (1 / n) * sum
            self.losses_.append(total_loss)
            self.w_ += self.eta * 
            



    def predict(self, X):
                

                
            





    def net_input(X):
        return np.dot(X, self.w_)
    





