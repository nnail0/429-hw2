# LinearSVC

"""
process:
init weights to low and bias to 0. 
start with an initial separation strip?
"""


import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt

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
        self.w_ = rand_gen.normal(loc = 0.0, scale = 0.001, size = 1 + m)

        self.losses_ = []    # this is for objective, not gradient

        for _ in range(self.n_iter):
            # get the input
            epoch_loss = []
            grad_losses = 0    
            for x_i, target in zip(X, y):
                # compute individual hinge losses
                
                init_hinge_loss = max(0, 1 - (target * self.net_input(x_i)))
                epoch_loss.append(init_hinge_loss)

                # grad_hinge_loss = (-(output) * target) * x_i
                # grad_losses += grad_hinge_loss
                if target * self.net_input(x_i) < 1:
                    grad_losses += -target * x_i
                # else it is 0. 

            # compute non-grad loss for convergence
            total_loss = (1/2) * np.dot(self.w_, self.w_) + (C / n) * sum(epoch_loss)
            self.losses_.append(total_loss)

            grad_loss = self.w_ + (C / n) * grad_losses 

            # multiply by 1/n times the gradient
            self.w_ -= self.eta * grad_loss
        
        print("Model fit complete")
            

    def net_input(self, X):
        return np.dot(X, self.w_)


    # check the sign of the output. 
    def predict(self, X):
        return int(np.sign(self.net_input(X)))

def main():
    data = pd.read_csv("data/iris.data")
    X_iris = data.iloc[0:99, 2:4].to_numpy()
    print("X_iris: ", X_iris)
    y_iris = data.iloc[0:99, 4].to_numpy()

    for i in range(y_iris.size):
        if y_iris[i] == "Iris-setosa":
            y_iris[i] = int(1)
        elif y_iris[i] == "Iris-versicolor":
            y_iris[i] = int(-1)

    y_iris = y_iris.astype(int)

    print("Y_iris: ", y_iris)


    svc1 = LinearSVC(eta = 0.01, n_iter= 50)
    svc1.fit(X_iris, y_iris, 0.1)
    print("svc1 fit results: ", svc1.w_)

    plot1 = plt.plot()
    plt.scatter(X_iris[:, 0], X_iris[:, 1])

    x_vals = np.linspace(min(X_iris[:,0]), max(X_iris[:,0]), 100)
    y_vals = -(svc1.w_[1] * x_vals + svc1.w_[0]) / svc1.w_[2]

    plt.plot(x_vals, y_vals)
    plt.show()


if __name__ == "__main__":
    main()

