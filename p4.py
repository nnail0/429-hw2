# comparison. 

from sklearn.svm import LinearSVC as skSVC
from LinearSVC import LinearSVC as mySVC
from datagen import DataGenerator

import numpy as np
import pandas as pd
import time
from matplotlib import pyplot as plt

filenames = [
    "10d_500n_train.csv",
    "10d_500n_test.csv",
    "10d_5000n_train.csv",
    "10d_5000n_test.csv",
    "10d_50000n_train.csv",
    "10d_50000n_test.csv",
    "50d_500n_train.csv",
    "50d_500n_test.csv",
    "50d_5000n_train.csv",
    "50d_5000n_test.csv",
    "50d_50000n_train.csv",
    "50d_50000n_test.csv",
    "100d_500n_train.csv",
    "100d_500n_test.csv",
    "100d_5000n_train.csv",
    "100d_5000n_test.csv",
    "100d_50000n_train.csv",
    "100d_50000n_test.csv"
]

train_filenames = []
test_filenames = []

for name in filenames:
    if "train" in name:
        train_filenames.append(name)
    else:
        test_filenames.append(name)

my_svc = mySVC(eta = 0.001, n_iter = 50, random_state=100)
sk_primal = skSVC(loss = "hinge", dual = False)
sk_dual = skSVC(loss = "hinge", dual = True)

models = {"my_svc" : [], "sk_primal" : [], "sk_dual" : []}

def fit_sklearn_model(X_, y_, n_iter_, dual_ : bool):
    for i in range(1, n_iter_ + 1):
        model = skSVC(loss = "hinge", dual = dual_)
        model.fit(X = X_, y = y_, n_iter = i)

        result = model.predict(X_)
        accuracy = model.score(X_, y_)

        if dual_ == False:
            models.get("sk_primal").append(accuracy)
        else:
            models.get("sk_dual").append(accuracy)

train_datasets = []
test_datasets = []
for name in train_filenames:
    train_data = np.genfromtxt(name, delimiter=',')
    train_datasets.append(train_data)

for name in test_filenames:
    test_data = np.genfromtxt(name, delimiter = ",")
    test_datasets.append(test_data)

for model in models.keys():
    print(str(model))
    print("\n")
    for data in train_datasets:
        n_samples = data.shape[0]
        n_dims = data.shape[1]
        X_data = data[:, 0:n_dims - 1]
        y_data = data[:, n_dims - 1]
        fit_start = time.clock_gettime(5)

        
        if model == "my_svc":
            mySVC.fit(X_data, y_data)
            models.get(model).update({"my_svc": model.errors_})
        elif model == "sk_primal":
            result = fit_sklearn_model(X_data, y_data, n_iter = 50, dual = False)
            models.get(model).update({"sk_primal" : result})
        elif model == "sk_dual":
            result = fit_sklearn_model(X_data, y_data, n_iter = 50, dual = True)
            models.get(model).update({"sk_dual" : result})

        fit_end = time.clock_gettime(5)
        print("%d dims, %d samples: %f" % (n_dims, n_samples, fit_end - fit_start))
        # end_weight = my_svc.w_

