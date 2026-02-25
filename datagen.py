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

        #print("matrix x: \n", matrix_x)

        # part c: label the data. 
        labels = np.ndarray((n_samples, 1))
        for i in range(n_samples):
            result = array_a.T.dot(matrix_x[i])    
            if result < 0:
                labels[i] = -1
            else :
                labels[i] = 1

        print("shape of x: ", np.shape(matrix_x),  ", shape of labels: ", np.shape(labels))
        result = np.append(matrix_x, labels, axis=1)
        # print(result)
        return result


def __main__():
    gen = DataGenerator()
    result = gen.generate(n_dims = 1, center= 0, u_range= 1)

    plot1 = plt.plot()
    plt.scatter(x = result[:, 0], y = result[:, 1])
    plt.show()
    
        

if __name__ == "__main__":
    __main__()

        


