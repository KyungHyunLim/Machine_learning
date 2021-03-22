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

from sklearn.preprocessing import StandardScaler, normalize, OneHotEncoder
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
  
#POI 임베딩 데이터 로드
data = pd.read_pickle('all_5poi_non_personal_sequence_dnn_data.pkl')
uid = pd.unique(data['User_Type'])

#MLP 
start_time = time.time()

result = {'model_num':[], 'top1':[], 'top3':[], 'top5':[]}
num = 0
big = []
why = []
for id in uid:
#     if id==-1 or id==4:
#         continue
    scaler = StandardScaler()
    
    print("")
    print(id)
    print("Model num: " + str(num))
    sub_data = data[data['User_Type']==id]
    
    
    if len(pd.unique(sub_data['REP_LAST_STN_ID'])) > 10000:
        print("too big" + str(id))
        big.append(id)
        del sub_data
        continue
    
    ohe = OneHotEncoder()
    y = pd.unique(sub_data['REP_LAST_STN_ID'])
    y_len = len(y)
    y =  np.array(sub_data['REP_LAST_STN_ID']).reshape(-1, 1)
    ohe.fit(y)
    one_hot = ohe.transform(y).toarray()
    sub_data = sub_data.drop('REP_LAST_STN_ID', axis=1)
    sub_data = sub_data.drop('User_Type', axis=1) 
    sub_data = sub_data.drop('TRCR_NO', axis=1) 
    
    if y_len != len(one_hot[0]):
        print("why?" + str(id))
        why.append(id)
        del sub_data
        del y
        del one_hot
        continue
        
    scaler.fit(sub_data)
    sub_data = scaler.transform(sub_data)
   
    train_x, test_x = train_test_split(sub_data, train_size=0.85, random_state=30)
    train_y, test_y = train_test_split(one_hot, train_size=0.85, random_state=30)

    del sub_data
    del y
    del one_hot
    
    c_input = Input((train_x.shape[1],))
    H = Dense(y_len//2+50)(c_input)
    H = BatchNormalization()(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(0.2)(H)
    H = Dense(y_len//2+30)(H)
    H = BatchNormalization()(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(0.2)(H)
    H = Dense(y_len//2+10)(H)
    H = LeakyReLU(0.2)(H)
    c_output = Dense(y_len, activation='softmax')(H)
    model_mlp = Model(c_input, c_output)
    model_mlp.summary()
    model_mlp.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc',top1,top3,top5])
    
    es = EarlyStopping(monitor='val_loss', patience=7)
    histoy = model_mlp.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=1000, batch_size=256, verbose=2, shuffle=True, callbacks=[es])
   
    model_mlp.save('ws3/'+str(id)+'.h5')

    result['model_num'].append(id)
  
    result['top1'].append(histoy.history['val_top1'][len(histoy.epoch)-1])
    result['top3'].append(histoy.history['val_top3'][len(histoy.epoch)-1])
    result['top5'].append(histoy.history['val_top5'][len(histoy.epoch)-1])
    
    
#     model_name = 'model/model_'+str(num)+'.h5' 
#     model_mlp.save(model_name)
    
    del train_x
    del train_y
    del test_x
    del test_y
    
    num = num + 1
    reset_keras()
    
    
print("\n\n")
print("---{}s seconds---".format(time.time()-start_time))
