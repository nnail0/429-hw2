# linearly seperable data generator. 

import numpy as np 
from matplotlib import pyplot as plt


class DataGenerator:

    def __init__(self, random_state = 1):
        self.random_state = random_state
    

    def generate(self, n_samples = 100, n_dims = 2, center = 0, u_range = 1):
        random_state = np.random.RandomState(self.random_state)
        
        # part a: generate random vector a
        array_a = random_state.randint(low = 0, high = 10, size = n_dims)

        #print("array_a: \n", array_a)

        matrix_x = np.ndarray((n_samples, n_dims))
        # part b: generate n random samples of size d. 
        for i in range(n_samples):
            matrix_x[i] = random_state.normal(loc = center, scale = u_range, size = n_dims)
            # matrix_x[i] = random_state.uniform(-u_range, u_range, n_dims)

        #print("matrix x: \n", matrix_x)

        split_idx = int(np.ceil(n_samples * 0.7))
        X_train = matrix_x[0:split_idx, :]
        X_test = matrix_x[split_idx:, :]

        # part c: label the training data. 
        labels = np.ndarray((X_train.shape[0], 1))
        for i in range(split_idx):
            result = array_a.T.dot(X_train[i])    
            if result < 0:
                labels[i] = -1
            else :
                labels[i] = 1


        print(np.shape(X_train))
        X_train = np.append(X_train, labels, axis=1)
        print(np.shape(X_train))
        # print(X_test)
        return (X_train, X_test)


def __main__():
    gen = DataGenerator(100)
    result_tuple = gen.generate(n_dims = 2, center= 0, u_range= 1)
    train = result_tuple[0]
    test = result_tuple[1]

    plot1 = plt.plot()
    plt.scatter(x = train[:, 0], y = train[:, 1])
    plt.show()
    
        

if __name__ == "__main__":
    __main__()

        


