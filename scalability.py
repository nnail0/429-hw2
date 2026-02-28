# 

from LinearSVC import LinearSVC as mySVC
from sklearn.svm import LinearSVC as skSVC
import time
import numpy as np
import pandas as pd
import csv

from matplotlib import pyplot as plt 

"""
Investigate the scalability of the LinearSVC class you have implemented. Using the dataset
generator developed in the previous task, you may produce random datasets regarding to the 9 combinations
of the following scales: d = 10, 50, 100 and n = 500, 5000, 50000. You may assign a large constant such
as 100 to u. (Please feel free to slightly adjust the scales according to your computer’s hardware.) Evaluate
the time cost and loss convergence of your linear SVC on the 9 datasets. The comparison should be given
by tables along with explanations.
"""

from datagen import DataGenerator
import itertools

n_dims = [10, 50, 100]
n_samples = [500, 5000, 50000]

dg1 = DataGenerator(100)

sets = []
errors = []

my_svc = mySVC(0.001, 50, 1)

def plot_losses(loss_values : list):
    x_vals = np.linspace(0, 49, 50)
    plot = plt.plot()
    plt.scatter(x = x_vals, y = loss_values)
    plt.show()

for dims, samples in itertools.product(n_dims, n_samples):
    print(dims, samples)
    data = dg1.generate(samples, dims, 0, u_range = 10)
    train_name = str(dims) + "d_" + str(samples) + "n_train.csv"
    test_name = str(dims) + "d_" + str(samples) + "n_test.csv"
    with open("data/" + train_name, mode = "w", newline = '') as f:
        writer = csv.writer(f)
        writer.writerows(data[0])
    with open("data/" + test_name, mode = "w", newline = '') as f:
        writer = csv.writer(f)
        writer.writerows(data[1])

    X_data = (data[0])[:, 0:dims - 1]
    y_data = (data[0])[:, dims - 1]
    fit_start = time.clock_gettime(5)
    my_svc.fit(X_data, y_data, 0.1)
    fit_end = time.clock_gettime(5)
    print("%d dims, %d samples: %f" % (dims, samples, fit_end - fit_start))
    end_weight = my_svc.w_
    errors.append(my_svc.losses_)


for l in errors:
    plot_losses(l)




