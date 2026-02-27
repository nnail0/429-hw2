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


    """
    So we perform the gradient for each sample, sum those up, 
    *then* after performing it for all samples, doing eta * this sum
    """

    def fit(self, X, y, C):

        # absorb the bias
        n = X.shape[0] #number of samples
        m = X.shape[1] #number of features 
        x_0 = np.ones((n,1)) 
        X = np.hstack((x_0,X))

        rand_gen = np.random.RandomState(self.random_state)
        self.w_ = rand_gen.normal(loc = 0.0, scale = 0.01, size = 1 + m)

        self.losses_ = []    # this is for objective, not gradient

        for _ in range(self.n_iter):
            # get the input
            epoch_loss = []
            grad_losses = 0    
            for x_i, target in zip(X, y):
                # compute individual hinge losses
                output = self.predict(x_i)

                init_hinge_loss = np.max(0, 1 - (target * output))
                epoch_loss.append(init_hinge_loss)

                grad_hinge_loss = (-(output) * target) * x_i
                grad_losses += grad_hinge_loss
            
            # calculate what is inside the sum term. 
            sum = 0
            for i in range(n):
                sum += epoch_loss[i] * (1/2) * np.dot(self.w_, self.w_)

            total_loss = (C / n) * sum
            self.losses_.append(total_loss)

            # multiply by 1/n times the gradient
            self.w_ += (1 / n) * self.eta * grad_losses
            

    def net_input(self, X):
        return np.dot(X, self.w_)


    # check the sign of the output. 
    def predict(self, X):
        return int(np.sign(self.net_input(X)))

def main():
    data = pd.read_csv("data/iris.data")
    X_iris = data.iloc[0:100, 0:3].to_numpy()
    y_iris = data.iloc[0:100, 4].to_numpy()

    for i in range(y_iris.size):
        if y_iris[i] == "Iris-setosa":
            y_iris[i] = int(-1)
        elif y_iris[i] == "Iris-versicolor":
            y_iris[i] = int(1)


    svc1 = LinearSVC(0.01, 50, 1)
    svc1.fit(X_iris, y_iris, 1)

if __name__ == "__main__":
    main()

