# linearly seperable data generator. 

import numpy as np 


class DataGenerator:

    def __init__(self, random_state = 1):
        self.random_state = random_state
    

    def generate(self, n_samples = 500, n_dims = 10, center = 0, u_range = 1):
        random_state = np.random.RandomState(self.random_state)
        generator = np.random.default_rng(random_state)
        # part a
        array_a = generator.random(size=n_dims)

        matrix_x = np.ndarray((n_samples, n_dims))
        # part b: generate n random samples of size d. 
        for _ in range(n_samples):
            np.append(matrix_x, generator.normal(loc = center, scale = u_range, size = n_dims))

        # part c: label the data. 
        labels = np.ndarray((n_samples, 1))
        for i in range(n_samples):
            result = array_a.T.dot(matrix_x[i])    
            if result < 0:
                np.append(labels, -1)
            else :
                np.append(labels, 1)

        print("shape of x: ", np.shape(matrix_x),  ", shape of labels: ", np.shape(labels))
        result = np.append(matrix_x, labels, axis=1)
        return result


def __main__():
    gen = DataGenerator()
    result = gen.generate()
    print(result)
        

if __name__ == "__main__":
    __main__()

        


