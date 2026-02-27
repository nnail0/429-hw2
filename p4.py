# comparison. 

from sklearn.svm import LinearSVC as skSVC
from LinearSVC import LinearSVC as mySVC
from datagen import DataGenerator as dg

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


generator = dg(100)