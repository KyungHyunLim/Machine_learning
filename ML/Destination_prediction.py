import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import itertools
import seaborn as sns
import math
from mpl_toolkits.mplot3d import axes3d
import time
from time import strftime

from keras.utils import np_utils
from keras.layers import Input, BatchNormalization, LeakyReLU
from keras.layers.core import Dense, Dropout, Flatten, Reshape
from keras.layers.convolutional import Convolution1D, MaxPooling1D, Convolution2D, MaxPooling2D
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping
from keras.metrics import top_k_categorical_accuracy
import keras.backend as K

from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.manifold import TSNE
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

def reset_weights(model):
    session = K.get_session()
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)

def top1(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=1)
def top3(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)
def top5(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)
  
#POI 
data = pd.read_pickle('all_5poi_non_personal_sequence_dnn_data.pkl')
