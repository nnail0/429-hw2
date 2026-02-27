# 429-hw2
Miles Nordwall
Assignment 2 completed as part of CS 429 - Intro to Machine Learning at UNM. 

## Task 1

```
class LinearSVC(object):
    """
    Linear Support Vector Classifier
    Params: 
    eta (float) : Learning rate (between 0.0 and 1.0)
    n_iter (int) : Number of passes over the training dataset.
    random_state (int) : Random number generator seed for random weight initialization.

    Attrs:
    w_ (1d-array) : Weights after fitting with bias absorbed at X[0].
    losses_ (list) : Hinge Loss with L2-regularization loss function values in each epoch.
    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.rand_gen = []
        self.w_initialized = False

    def fit(self, X, y, C=0.1):
        """
        Learns parameters from the training data
        Uses SGD
        Params:
        X {array-like}, shape = [n_examples, n_features]
        Training vectors, where n_examples is the number of examples and n_features is the number of features.
        y (array-like) ,shape = [n_examples]
        Target values.
        C {float} = regularization hyperparameter 

        Returns:
        self : Instance of LinearSVC
        """
        self.losses_ = []
        X = self._initialize_weights(X,X.shape[1])
        c_n = C /X.shape[0]
        
        for _ in range(self.n_iter):
            
            X, y = self._shuffle(X, y)
            epoch_losses = []
            
            for xi, yi in zip(X, y):
                
                y_hat = self.net_input(xi)
                fn_margin = 1 - (yi*y_hat)
                Li = np.maximum(0,fn_margin)

                #gd = self.w_ + c_n * (-yi*xi) #dL/dw
                b_update = 0
                if fn_margin > 0: #case 1
                    gd = self.w_ + c_n * (-yi*xi) #dL/dw
                    b_update = -yi
                else: #case 2
                    gd = self.w_
                epoch_losses.append(Li)
                self.w_ += self.eta* (-gd)
                self.b_ += self.eta* b_update
            avg_loss = np.mean(epoch_losses)
            self.losses_.append(avg_loss)
        return self
        

    
    def _initialize_weights(self, X, m):
        """Initialize weights to small random numbers
        """
        self.b_ = np.float_(0.)
        self.rand_gen = np.random.RandomState(self.random_state)
        self.w_ = self.rand_gen.normal(loc=0.0, scale=0.01,
                                  size=m)
        self.w_initialized = True
        return X
        

    def _update_weights(self, xi, y):
        """Calculate hinge loss then apply GD to L wrt w_"""
        y_hat = self.net_input(xi)
        Li = np.max(0, 1 - (y*y_hat))
        loss = c_n * (Li + (0.5 * (self.w_.shape[0])^2))
        self.w_ += self.w_ + self.eta - loss
        self.b_ += gd
        return loss
        
    def _shuffle(self, X, y):
        """Shuffle training data"""
        r = self.rand_gen.permutation(len(y))
        return X[r], y[r]

    def net_input(self, X): #TODO1 the net_input for svc is w^Tx-b right? unabsorb b?
        """Calculate net input"""
        return np.dot(X, self.w_) - self.b_

    def predict(self, X):
        """Return class label after unit step"""
        return np.sign(self.net_input(X))
```

To implement a Linear SVC in Python, we can start with a basic perceptron template and modify it to suit the needs of the model. The main major changes needed were to the loss calculation and the model update. This implementation of the SVC uses hinge loss with L2 regularization. 

- For the loss update, we use the following equation: 

$$ L_i = \max(0, 1 - (y_i * \hat{y}_i)) $$

- To update the model, we take the gradient of this loss:

$$ w = w + \eta \frac{\partial L}{\partial w} $$
$$ b = b + \eta \frac{\partial L}{\partial b} $$

To compute the total loss, we loop over pairs of observations `for x_i, target in zip(X, y)` similar to the perceptron. In this loop, we perform two calculations. One operation calculates the standard hinge loss, which we can use to evaluate how quickly the model converges with each epoch. Another operation computes the gradient of the hinge loss, which will be used in a later calculation to determine how to update the model. 

# Task 2

```
lass DataGenerator:

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

```

Task 2 has us implement a generator that produces linearly seperable sets of data using knowledge from linear algebra. To demonstrate the efficacy of the code, we can set the number of dimensions to 2 with 100 data points. 

![Scatter plot of linearly seperable data.](image.png)

With seed 100 on this specific RNG, we can see a clear split between two clouds of data. This data would make for a clean seed on which to train a model. 

