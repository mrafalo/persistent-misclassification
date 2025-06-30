from keras.initializers import Constant
from keras.layers import Input, Conv2D, Flatten, Activation, MaxPool2D, Dropout
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import tensorflow as tf
from keras import backend as K
from tensorflow.keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Conv2D,  MaxPool2D, Flatten, GlobalAveragePooling2D,  BatchNormalization, Layer, Add, concatenate
import logging
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn import metrics
import numpy as np
import work.custom_logger as cl
import work.globals as g
import work.data as d
import work.sidc as sidc
import work.didc as didc

from sklearn.linear_model import LogisticRegression
from keras.utils import custom_object_scope
import keras
from tensorflow.keras.applications import ResNet50, ResNet101, ResNet152
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import classification_report
from sklearn import metrics
from scipy.stats import chi2_contingency
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn import tree
import pandas as pd
from sklearn.metrics import confusion_matrix
from scipy.stats import norm
import ast
from sklearn.preprocessing import OneHotEncoder
import os
from datetime import datetime

logger = cl.get_logger()


def result_found(_df, _c):
    
    founded = _df[
        (_df['loss_function'] == _c['loss_function']) & 
        (_df['model_name'] == _c['model_name']) &
        (_df['learning_rate'] == _c['learning_rate']) &
        (_df['optimizer'] == _c['optimizer']) &
        (_df['img_size'] == _c['img_size']) &
        (_df['batch_size'] == _c['batch_size']) &
        (not _df['status'].isna().any())
        ]
    
    if len(founded)>0: 
        return True
    else:
        return False


def models_variance(_models, _latex=True):
    df = pd.read_csv(g.RESULT_FILE, sep=';')
    cols = ['model_name', 'optimizer', 'loss_function', 'auc_mean', 'sensitivity_mean', 'precision_mean']
    df = df[cols]
   
    # variance_df = df.groupby('model_name')[['auc_mean', 'sensitivity_mean', 'precision_mean']].var()
    std_dev_df = round(df.groupby('model_name')[['auc_mean', 'sensitivity_mean', 'precision_mean']].std(),2)



def model_summary(_model_name):
    df = pd.read_csv(g.RESULT_FILE, sep=';')
    res = df.loc[(df['model_name'] == _model_name) & 
                 (df['optimizer']=='SGD') &
                 (df['learning_rate']==0.005),:].groupby('loss_function').agg({
        'auc_mean': 'max',
        'sensitivity_mean': 'max',
        'precision_mean': 'max'
        })
    
    res = res.round(2)
    print(res.to_latex())
    return res
    
         
def models_summary(_models, _latex=True):
    df = pd.read_csv(g.RESULT_FILE, sep=';')

    losses = ['focal_loss', 'kl_divergence', 'categorical_crossentropy', 'squared_hinge' ]
    
    res = df.loc[df['loss_function'].isin(losses),:].pivot_table(
        index='model_name', 
        columns='loss_function', 
        values='auc_mean', 
        aggfunc=np.max
    )

    
    res = res.round(2)
    
    if _latex:
        print(res.to_latex())
    else:
        print(res.to_string())
        
def get_optimal_models(_latex=True):
    df = pd.read_csv(g.RESULT_FILE, sep=';')
    cols = ['model_name', 'optimizer', 'loss_function', 'auc_mean', 'sensitivity_mean', 'precision_mean']
    df = df[cols]
    idx_auc_mean = df.groupby('model_name')['auc_mean'].idxmax()

    res = df.loc[idx_auc_mean]
    
    res = res.round(2)

    if _latex:
        print(res.to_latex(index=False))
    else:
        print(res.to_string(index=False))
        
def chi2(_data):
    stat, p_val, dof, expected = chi2_contingency(_data)
    
    return p_val
      

def random_forest_cv(_iters):
    
    # _iters = 2
    df = d.load_base_data_file()    

    X = df[g.FEATURES_PL] 
    y = df.rak 
    results = []
    
    np.random.seed(g.SEED)

    for i in range(0, _iters):
      x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
      
      m1 = RandomForestClassifier()
      m1 = m1.fit(x_train,y_train)
      
      res = model_predictor_scikit(m1, x_test, y_test)

      
      importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': m1.feature_importances_
        }).sort_values(by='importance', ascending=False)
    
      res['f1'] = importance_df.iloc[0]['feature']  
      res['f2'] = importance_df.iloc[1]['feature']  
      res['f3'] = importance_df.iloc[2]['feature']  
      res['f4'] = importance_df.iloc[3]['feature']
      res['f5'] = importance_df.iloc[4]['feature']
      
      results.append(res) 
      
    results = pd.DataFrame(results)
    
    value_counts = pd.Series(results[['f1', 'f2', 'f3', 'f4', 'f5']].values.flatten()).value_counts()
    # print(value_counts)

    res = pd.DataFrame(value_counts)
    res = res.reset_index()
    res.columns = ['feature', 'cnt']
    
    
    return res, importance_df

def log_reg_cv(_iters):
    # _iters = 2
    df = d.load_base_data_file()    

    X = df[g.FEATURES_PL] 
    y = df.rak 
    results = []
    
    np.random.seed(g.SEED)
    for i in range(0, _iters):
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
        
        m1 = LogisticRegression(max_iter=1000)
        m1 = m1.fit(x_train,y_train)
        res = model_predictor_scikit(m1, x_test, y_test)
        coefficients = m1.coef_[0]
        importance_df = pd.DataFrame({'feature': g.FEATURES_PL, 'importance': np.abs(coefficients)}).sort_values(by='importance', ascending=False)
        res['f1'] = importance_df.iloc[0]['feature']  
        res['f2'] = importance_df.iloc[1]['feature']  
        res['f3'] = importance_df.iloc[2]['feature']  
        res['f4'] = importance_df.iloc[3]['feature'] 
        res['f5'] = importance_df.iloc[4]['feature']
      
        results.append(res) 
      
    res = pd.DataFrame(results)
    
    value_counts = pd.Series(res[['f1', 'f2', 'f3', 'f4', 'f5']].values.flatten()).value_counts()

    return value_counts, importance_df
    

def focal_loss(y_true, y_pred, gamma=2, alpha=2):
    pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    return -K.sum(alpha * K.pow(1. - pt, gamma) * K.log(pt + 1e-6), axis=-1)

def model_cnn_base(_img_width, _img_height):
    # 160x160x1
    input_tensor = Input(shape=(_img_height, _img_width, 1), name="thyroid_input")
    # 160x160x8
    x = Conv2D(8, (3, 3), padding="same", activation="relu")(input_tensor)
    # 80x80x8
    x = MaxPool2D((2, 2), strides=(2, 2))(x)
    # 80x80x12
    x = Conv2D(12, (3, 3), padding="same", activation="relu")(x)
    # 40x40x12
    x = MaxPool2D((2, 2), strides=(2, 2))(x)
    # 40x40x16
    x = Conv2D(16, (3, 3), padding="same", activation="relu")(x)
    # 20x20x16
    x = MaxPool2D((2, 2), strides=(2, 2))(x)
    # 20x20x24
   
    x = Conv2D(24, (3, 3), padding="same", activation="relu")(x)
    # 10x10x24
    x = MaxPool2D((2, 2), strides=(2, 2))(x)
    # 10x10x32
    x = Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    # 5x5x32
    x = MaxPool2D((2, 2), strides=(2, 2))(x)
    # 5x5x48
    x = Conv2D(48, (3, 3), padding="same", activation="relu")(x)
    # 5x5x48
    x = Dropout(0.5)(x)

    y_cancer = Conv2D(
        filters=1,
        kernel_size=(5, 5),
        kernel_initializer="glorot_normal",
        bias_initializer=Constant(value=-0.9),
    )(x)

    y_cancer = Flatten()(y_cancer)
    y_cancer = Activation("sigmoid", name="out_cancer")(y_cancer)

    return Model(
        inputs=input_tensor,
        outputs=y_cancer,
    )

def model_cnn1(_img_width, _img_height, _channels):
    model = Sequential()
    model.add(Input(shape=(_img_height, _img_width, _channels)))
    model.add(Conv2D(8, (3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(1, 1)))
    model.add(Conv2D(12, (3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(1, 1)))
    model.add(Conv2D(16, (5, 5), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(1, 1)))
    model.add(Conv2D(24, (5, 5), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(32, (7, 7), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(48, (7, 7), padding="same", activation="relu"))
    model.add(Dropout(0.5))
    model.add(Conv2D(filters=1,kernel_size=(5, 5),kernel_initializer="glorot_normal",bias_initializer=Constant(value=-0.9)))
    model.add(Flatten())
    model.add(Dense(2, activation='sigmoid'))
    
    return model

def model_cnn1_ext(_img_width, _img_height, _channels, _num_features):
    
    
    
    image_input = Input(shape=(_img_width, _img_height, _channels), name="image_input")
    x = Conv2D(8, (3, 3), activation="relu")(image_input)
    x = MaxPooling2D((2, 2), strides=(1, 1))(x)
    
    x = Conv2D(12, (3, 3), activation="relu")(x)
    x = MaxPooling2D((2, 2), strides=(1, 1))(x)
    
    x = Conv2D(16, (5, 5), activation="relu")(x)
    x = MaxPooling2D((2, 2), strides=(1, 1))(x)
    
    x = Conv2D(24, (5, 5), activation="relu")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(32, (7, 7), activation="relu")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(48, (7, 7), activation="relu")(x)
    # x = MaxPooling2D((2, 2))(x)
    
    x = Dropout(0.5)(x)
    x = Conv2D(filters=1,kernel_size=(5, 5),kernel_initializer="glorot_normal",bias_initializer=Constant(value=-0.9))(x)
    
    x = Flatten()(x)
    
    ext_input = Input(shape=(_num_features,), name="ext_input")
    y = Dense(16, activation="relu")(ext_input)
    
    merged = concatenate([x, y])
    z = Dense(16, activation="relu")(merged)
    z = Dropout(0.5)(z)
    z = Dense(2, activation="sigmoid")(z)
    
    model = Model(inputs=[image_input, ext_input], outputs=z)
    
    return model

def model_cnn2_4grad(_img_width, _img_height, _channels):
    model = Sequential()
    model.add(Input(shape=(_img_height, _img_width, _channels)))
    model.add(Conv2D(8, (3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(1, 1)))
    model.add(Conv2D(12, (3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(1, 1)))
    model.add(Conv2D(16, (5, 5), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(1, 1)))
    model.add(Conv2D(24, (5, 5), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(32, (7, 7), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(48, (7, 7), padding="same", activation="relu"))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(2, activation='sigmoid'))
    
    return model

def model_cnn2_4grad(x):
    # build CNN model
    with tf.variable_scope('model_cnn', reuse = False) as scope:
        x_t = tf.transpose(x, [0, 3, 1, 2]) # NHWC to NCHW

       # Input layer
        # inputs = Input(shape=(_img_height, _img_width, _channels), name='input')
        
        # Block 1
        conv1 = Conv2D(8, (3, 3), padding="same", activation='relu', name='conv1')(x_t)
        pool1 = MaxPooling2D((2, 2), strides=(1, 1), name='pool1')(conv1)
        
        # Block 2
        conv2 = Conv2D(12, (3, 3), padding="same", activation='relu', name='conv2')(pool1)
        pool2 = MaxPooling2D((2, 2), strides=(1, 1), name='pool2')(conv2)
        
        # Block 3
        conv3 = Conv2D(16, (5, 5), padding="same", activation='relu', name='conv3')(pool2)
        pool3 = MaxPooling2D((2, 2), strides=(1, 1), name='pool3')(conv3)
        
        # Block 4
        conv4 = Conv2D(24, (5, 5), padding="same", activation='relu', name='conv4')(pool3)
        pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(conv4)
        
        # Block 5
        conv5 = Conv2D(32, (7, 7), padding="same", activation='relu', name='conv5')(pool4)
        pool5 = MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(conv5)
        
        # Block 6
        conv6 = Conv2D(48, (7, 7), padding="same", activation='relu', name='conv6')(pool5)
        dropout = Dropout(0.5, name='dropout')(conv6)
        
        # Fully connected layer
        flatten = Flatten(name='flatten')(dropout)
        output = Dense(2, activation='sigmoid', name='output')(flatten)

    return [conv1, conv2, conv3, conv4, conv5, conv6], flatten, output

def model_cnn2(_img_width, _img_height, _channels):
    model = Sequential()
    model.add(Input(shape=(_img_height, _img_width, _channels)))
    model.add(Conv2D(8, (3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(1, 1)))
    model.add(Conv2D(12, (3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(1, 1)))
    model.add(Conv2D(16, (5, 5), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(1, 1)))
    model.add(Conv2D(24, (5, 5), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(32, (7, 7), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(48, (7, 7), padding="same", activation="relu"))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(2, activation='sigmoid'))
    
    return model

def model_cnn2_ext(_img_width, _img_height, _channels, _num_features):
    image_input = Input(shape=(_img_width, _img_height, _channels), name="image_input")
    x = Conv2D(8, (3, 3), activation="relu")(image_input)
    x = MaxPooling2D((2, 2), strides=(1, 1))(x)
    
    x = Conv2D(12, (3, 3), activation="relu")(x)
    x = MaxPooling2D((2, 2), strides=(1, 1))(x)
    
    x = Conv2D(16, (3, 3), activation="relu")(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(24, (3, 3), activation="relu")(x)
    x = MaxPooling2D((2, 2), strides=(1, 1))(x)

    x = Conv2D(32, (7, 7), activation="relu")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(48, (7, 7), activation="relu")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    
    x = Dropout(0.5)(x)
    # x = Conv2D(filters=1,kernel_size=(5, 5),kernel_initializer="glorot_normal",bias_initializer=Constant(value=-0.9))(x)
    
    x = Flatten()(x)
    
    ext_input = Input(shape=(_num_features,), name="ext_input")
    y = Dense(16, activation="relu")(ext_input)
    
    merged = concatenate([x, y])
    z = Dense(16, activation="relu")(merged)
    z = Dropout(0.5)(z)
    z = Dense(2, activation="sigmoid")(z)
    
    model = Model(inputs=[image_input, ext_input], outputs=z)
    
    return model

def model_cnn3(_img_width, _img_height, _channels):
    model = Sequential()
    model.add(Input(shape=(_img_height, _img_width, _channels)))
    model.add(Conv2D(8, (3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(1, 1)))
    model.add(Conv2D(12, (3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(1, 1)))
    model.add(Conv2D(16, (5, 5), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(1, 1)))
    model.add(Conv2D(24, (5, 5), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(filters=1,kernel_size=(5, 5),kernel_initializer="glorot_normal",bias_initializer=Constant(value=-0.9)))
    model.add(Flatten())
    model.add(Dense(2, activation='sigmoid'))
    
    return model


def model_cnn3_ext(_img_width, _img_height, _channels, _num_features):
    
    # print(_img_width, _img_height, _channels, _num_features)
    
    image_input = Input(shape=(_img_width, _img_height, _channels), name="image_input")
    x = Conv2D(8, (3, 3), activation="relu")(image_input)
    x = MaxPooling2D((2, 2), strides=(1, 1))(x)
    
    x = Conv2D(12, (3, 3), activation="relu")(x)
    x = MaxPooling2D((2, 2), strides=(1, 1))(x)
    
    x = Conv2D(16, (3, 3), activation="relu")(x)
    x = MaxPooling2D((2, 2), strides=(1, 1))(x)
    
    x = Conv2D(24, (3, 3), activation="relu")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Dropout(0.5)(x)
    x = Conv2D(filters=1,kernel_size=(5, 5),kernel_initializer="glorot_normal",bias_initializer=Constant(value=-0.9))(x)
    
    x = Flatten()(x)
    
    ext_input = Input(shape=(_num_features,), name="ext_input")
    y = Dense(16, activation="relu")(ext_input)
    
    merged = concatenate([x, y])
    # z = Dense(16, activation="relu")(merged)
    z = Dropout(0.5)(merged)
    z = Dense(2, activation="sigmoid")(z)
    
    model = Model(inputs=[image_input, ext_input], outputs=z)
    
    return model



def model_cnn4(_img_width, _img_height, _channels):
    model = Sequential()
    model.add(Input(shape=(_img_height, _img_width, _channels)))
    model.add(Conv2D(8, (3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(1, 1)))
    model.add(Conv2D(12, (3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(1, 1)))
    model.add(Conv2D(16, (5, 5), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(1, 1)))
    model.add(Conv2D(24, (5, 5), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(32, (7, 7), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(48, (7, 7), padding="same", activation="relu"))
    model.add(Dropout(0.8))
    model.add(Conv2D(filters=1,kernel_size=(5, 5),kernel_initializer="glorot_normal",bias_initializer=Constant(value=-0.9)))
    model.add(Flatten())
    model.add(Dense(2, activation='sigmoid'))
    
    return model

def model_cnn4_ext(_img_width, _img_height, _channels, _num_features):
    image_input = Input(shape=(_img_width, _img_height, _channels), name="image_input")
    x = Conv2D(8, (3, 3), activation="relu")(image_input)
    x = MaxPooling2D((2, 2), strides=(1, 1))(x)
    
    x = Conv2D(12, (3, 3), activation="relu")(x)
    x = MaxPooling2D((2, 2), strides=(1, 1))(x)
    
    x = Conv2D(16, (5, 5), activation="relu")(x)
    x = MaxPooling2D((2, 2), strides=(1, 1))(x)
    
    x = Conv2D(24, (5, 5), activation="relu")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    
    x = Conv2D(32, (7, 7), activation="relu")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(48, (7, 7), activation="relu")(x)
    # x = MaxPooling2D((2, 2))(x)
    
    x = Dropout(0.8)(x)
    x = Conv2D(filters=1,kernel_size=(5, 5),kernel_initializer="glorot_normal",bias_initializer=Constant(value=-0.9))(x)
    
    x = Flatten()(x)
    
    ext_input = Input(shape=(_num_features,), name="ext_input")
    y = Dense(16, activation="relu")(ext_input)
    
    merged = concatenate([x, y])
    z = Dense(16, activation="sigmoid")(merged)
    z = Dropout(0.5)(z)
    z = Dense(2, activation="sigmoid")(z)
    
    model = Model(inputs=[image_input, ext_input], outputs=z)
    
    return model


def model_cnn5(_img_width, _img_height, _channels):
    model = Sequential()
    model.add(Input(shape=(_img_height, _img_width, _channels)))
    model.add(Conv2D(8, (3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(1, 1)))
    model.add(Conv2D(12, (3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(1, 1)))
    model.add(Conv2D(16, (5, 5), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(1, 1)))
    model.add(Conv2D(24, (5, 5), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(32, (7, 7), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(48, (7, 7), padding="same", activation="relu"))
    model.add(Dropout(0.5))
    model.add(Conv2D(1, (7, 7), padding="same", activation="relu"))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))
    
    return model

def model_cnn6(_img_width, _img_height, _channels):
    model = Sequential()
    model.add(Input(shape=(_img_height, _img_width, _channels)))
    model.add(Conv2D(8, (3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(1, 1)))
    model.add(Conv2D(12, (3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(1, 1)))
    model.add(Conv2D(16, (5, 5), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(1, 1)))
    model.add(Conv2D(24, (5, 5), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(32, (7, 7), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(48, (7, 7), padding="same", activation="relu"))
    model.add(Dropout(0.8))
    model.add(Conv2D(filters=1,kernel_size=(5, 5),activation='softmax'))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))
    
    return model

def model_cnn7(_img_width, _img_height, _channels):
    model = Sequential()
    model.add(Input(shape=(_img_height, _img_width, _channels)))
    model.add(Conv2D(8, (3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(1, 1)))
    model.add(Conv2D(12, (3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(1, 1)))
    model.add(Conv2D(16, (5, 5), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(1, 1)))
    model.add(Conv2D(24, (5, 5), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(32, (7, 7), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(48, (7, 7), padding="same", activation="relu"))
    model.add(Dropout(0.5))
    model.add(Conv2D(filters=1,kernel_size=(5, 5),activation='softmax'))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))
    
    return model

def model_densenet201(_img_width, _img_height, _channels):
    
    model = tf.keras.applications.DenseNet201(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=(_img_width,_img_height, _channels),
        pooling=None,
        classes=2 )
    
    return model

def model_densenet121(_img_width, _img_height, _channels):
    
    model = tf.keras.applications.DenseNet121(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=(_img_width,_img_height, _channels),
        pooling=None,
        classes=2 )
    
    return model

def model_VGG16(_img_width, _img_height, _channels):
    
    model = tf.keras.applications.VGG16(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=(_img_width,_img_height, _channels),
        pooling=None,
        classes=2,
        classifier_activation="softmax")
 
    return model

def model_VGG19(_img_width, _img_height, _channels):
    
    model = tf.keras.applications.VGG19(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=(_img_width,_img_height, _channels),
        pooling=None,
        classes=2,
        classifier_activation="softmax")
  
 
    return model


def model_sidc(_img_width, _img_height, _channels):
    
    inputs = Input(shape=(_img_width,_img_height,_channels))
    outputs_x1, outputs_y1 = sidc.entry_flow(inputs)
    outputs_x2, outputs_y2 = sidc.middle_flow(outputs_x1, outputs_y1)
    outputs = sidc.exit_flow(outputs_x2, outputs_y2)
    return Model(inputs, outputs)

def model_didc(_img_width, _img_height, _channels):
    
    inputs = Input(shape=(_img_width,_img_height,_channels))
    outputs = didc.exit_flow(didc.middle_flow(didc.entry_flow(inputs)))

    return Model(inputs, outputs)


def model_ResNet50(_img_width, _img_height, _channels):
    
    
    base_model = ResNet50(weights=None, include_top=False, input_shape=(_img_width, _img_height, _channels))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)      
    model = Model(inputs=base_model.input, outputs=predictions)

    return model

def model_ResNet101(_img_width, _img_height, _channels):
    
    base_model = ResNet101(weights=None, include_top=False, input_shape=(_img_width, _img_height, _channels))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)      
    model = Model(inputs=base_model.input, outputs=predictions)
     
    return model

def model_ResNet152(_img_width, _img_height, _channels):
    
    base_model = ResNet152(weights=None, include_top=False, input_shape=(_img_width, _img_height, _channels))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)      
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model

def find_cutoff(target, predicted):
    fpr, tpr, t = metrics.roc_curve(target, predicted)
    tnr = 1 - fpr
    g = np.sqrt(tpr*tnr)
    pos = np.argmax(g)

    return t[pos]    


def get_model_by_name(_cfg):

    features_cnt = len(ast.literal_eval(_cfg['features']))
    
    if _cfg['model_name'] == 'ResNet101': return model_ResNet101(_cfg['img_size'], _cfg['img_size'], _cfg['channels'])
    if _cfg['model_name'] == 'ResNet152': return model_ResNet152(_cfg['img_size'], _cfg['img_size'], _cfg['channels'])
    if _cfg['model_name'] == 'ResNet50': return model_ResNet50(_cfg['img_size'], _cfg['img_size'], _cfg['channels'])
    if _cfg['model_name'] == 'VGG16': return model_VGG16(_cfg['img_size'], _cfg['img_size'], _cfg['channels'])
    if _cfg['model_name'] == 'VGG19': return model_VGG19(_cfg['img_size'], _cfg['img_size'], _cfg['channels'])
    if _cfg['model_name'] == 'denseNet121': return model_densenet121(_cfg['img_size'], _cfg['img_size'], _cfg['channels'])
    if _cfg['model_name'] == 'denseNet201': return model_densenet201(_cfg['img_size'], _cfg['img_size'], _cfg['channels'])
    if _cfg['model_name'] == 'cnn1': return model_cnn1(_cfg['img_size'], _cfg['img_size'], _cfg['channels'])    
    if _cfg['model_name'] == 'cnn2': return model_cnn2(_cfg['img_size'], _cfg['img_size'], _cfg['channels'])    
    if _cfg['model_name'] == 'cnn3': return model_cnn3(_cfg['img_size'], _cfg['img_size'], _cfg['channels'])    
    if _cfg['model_name'] == 'cnn4': return model_cnn4(_cfg['img_size'], _cfg['img_size'], _cfg['channels'])
    if _cfg['model_name'] == 'cnn5': return model_cnn5(_cfg['img_size'], _cfg['img_size'], _cfg['channels'])
    if _cfg['model_name'] == 'sidc': return model_sidc(_cfg['img_size'], _cfg['img_size'], _cfg['channels'])
    if _cfg['model_name'] == 'didc': return model_didc(_cfg['img_size'], _cfg['img_size'], _cfg['channels'])
    
    if _cfg['model_name'] == 'cnn1_ext': return model_cnn1_ext(_cfg['img_size'], _cfg['img_size'], _cfg['channels'], features_cnt)
    if _cfg['model_name'] == 'cnn2_ext': return model_cnn2_ext(_cfg['img_size'], _cfg['img_size'], _cfg['channels'], features_cnt)
    if _cfg['model_name'] == 'cnn3_ext': return model_cnn3_ext(_cfg['img_size'], _cfg['img_size'], _cfg['channels'], features_cnt)
    if _cfg['model_name'] == 'cnn4_ext': return model_cnn4_ext(_cfg['img_size'], _cfg['img_size'], _cfg['channels'], features_cnt)

def model_predictor(_model, _X_test, _y_test):
    
    y_base = _y_test
    _y_test = _y_test[:,1]
    
    res_details = pd.DataFrame()
    
    test_cases = len(_y_test)
    test_positives = np.sum(_y_test)
    
    if test_positives==0:
        return {
        'accuracy': -1,
        'sensitivity': -1,
        'specificity': -1,
        'precision': -1,
        'f1': -1,
        'auc': -1,
        'threshold': -1,
        'status': 'NO TARGET'
        }
        
    y_predict_base = _model.predict(_X_test, verbose=0)
    m_opt_predict = y_predict_base[:,1]
    
    contains_nan = np.isnan(m_opt_predict).any()
    
    if not contains_nan:
        t = find_cutoff(_y_test,m_opt_predict)
        m_opt_predict_binary = [1 if x >= t else 0 for x in m_opt_predict]

        res_details = pd.DataFrame({
            "actual": _y_test,
            "threshold":  [t for x in range(len(_y_test))],
            "predict": m_opt_predict,
            "predict_binary": m_opt_predict_binary,
            "status": ["OK" for x in range(len(_y_test))]
        })
        
        conf_matrix = np.round(metrics.confusion_matrix(_y_test, m_opt_predict_binary),2)
        
        accuracy = np.round(metrics.accuracy_score(_y_test, m_opt_predict_binary),2)
        sensitivity = np.round(metrics.recall_score(_y_test, m_opt_predict_binary),2)
        specificity = np.round(conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1]),2)
        precision = np.round(metrics.precision_score(_y_test, m_opt_predict_binary),2)
        auc = np.round(metrics.roc_auc_score(_y_test, m_opt_predict),2)
        f1 = np.round(metrics.f1_score(_y_test, m_opt_predict_binary),2)
        test_cases = len(_y_test)
        test_positives = np.sum(_y_test)
        
        res = {
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1': f1,
            'auc': auc,
            'threshold': t,
            'status': 'OK'
        }
    else:
        res = {
            'accuracy': 0,
            'sensitivity': 0,
            'specificity': 0,
            'precision': 0,
            'f1': 0,
            'auc': 0,
            'threshold': 0,
            'status': 'NULL VALUES'
            }
        
        res_details = pd.DataFrame({
            "actual": _y_test,
            "threshold":  [-1 for x in range(len(_y_test))],
            "predict": [-1 for x in range(len(_y_test))],
            "predict_binary": [-1 for x in range(len(_y_test))],
            "status": ["ERR" for x in range(len(_y_test))]
        })
                
    
    return res, res_details


def model_predictor_scikit(_model, _X_test, _y_test):
    
    test_cases = len(_y_test)
    test_positives = np.sum(_y_test)
    
    if test_positives==0:
        return {
        'accuracy': -1,
        'sensitivity': -1,
        'specificity': -1,
        'precision': -1,
        'f1': -1,
        'auc': -1,
        'threshold': -1,
        'test_cases': test_cases,
        'test_positives': test_positives
        }
    
    y_predict_base = _model.predict_proba(_X_test)
    
    m_opt_predict = y_predict_base[:,1]
    
    t = find_cutoff(_y_test,m_opt_predict)

    m_opt_predict_binary = [1 if x >= t else 0 for x in m_opt_predict]
 
    conf_matrix = np.round(metrics.confusion_matrix(_y_test, m_opt_predict_binary),2)
    
   
    accuracy = np.round(metrics.accuracy_score(_y_test, m_opt_predict_binary),2)
    sensitivity = np.round(metrics.recall_score(_y_test, m_opt_predict_binary),2)
    specificity = np.round(conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1]),2)
    precision = np.round(metrics.precision_score(_y_test, m_opt_predict_binary),2)
    auc = np.round(metrics.roc_auc_score(_y_test, m_opt_predict),2)
    f1 = np.round(metrics.f1_score(_y_test, m_opt_predict_binary),2)
    test_cases = len(_y_test)
    test_positives = np.sum(_y_test)
    
    res = {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1': f1,
        'auc': auc,
        'threshold': t,
        'test_cases': test_cases,
        'test_positives': test_positives
    }
    
    return res

def model_load(_path):

    with custom_object_scope({'focal_loss': focal_loss}):
        m1 = load_model(_path)
    
    return m1    

def model_fitter(_model, _X_train, _y_train, _X_val, _y_val, _X_test, _y_test, _id_coi_list_test, _images_list, _epochs, _learning_rate, _batch_size, _optimizer, _loss, _model_name):
    
    
    if _optimizer == 'Adam':
        opt = Adam(learning_rate=_learning_rate)
    else:
        opt = SGD(learning_rate=_learning_rate)
    
    if _loss != 'focal_loss':
        _model.compile(optimizer = opt, loss=_loss, metrics=["accuracy"]) 
    else:
        _model.compile(optimizer = opt, loss=focal_loss, metrics=["accuracy"]) 
        
    es = EarlyStopping(monitor='val_accuracy', mode='max', patience=30, restore_best_weights=True)
    
    # checkpoint_path = "checkpoint/cp.ckpt"
    # checkpoint_dir = os.path.dirname(checkpoint_path)

    # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
    #                                                  save_weights_only=True,
    #                                                  verbose=1)

    hist = _model.fit(_X_train, _y_train, 
                      validation_data=(_X_val, _y_val), 
                      callbacks=[es], 
                      batch_size=_batch_size, 
                      epochs=_epochs, 
                      verbose=False)
    
    
    if es.stopped_epoch > 0:
        logger.info("Early stopped at epoch: " + str(es.stopped_epoch) + ' of ' + str(_epochs));

    #ev = _model.evaluate(_X_test, _y_test, verbose=False)

    res, res_details = model_predictor(_model, _X_test, _y_test)

    res_details['image'] = _images_list
    res_details["id_coi"] = _id_coi_list_test
    curr_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    # _model.save(f"{_stock}_{_model_name}_tf_{str(res['auc'])}", save_format='tf')
    _model.save(f"models/{curr_date}_{_model_name}_h5_{str(res['auc'])}", save_format='h5')
    return res, res_details



def compute_cm_metrics():
    def confidence_interval(prop, n, confidence):
        if n == 0:
            return (0, 0)
        z = norm.ppf(1 - (1 - confidence) / 2)
        se = np.sqrt((prop * (1 - prop)) / n)
        return (round(max(0, prop - z * se),3), round(min(1, prop + z * se),3))

    
    _confidence: float = 0.95
    specifities = []
    specifities_ci = []
    sensitivities = []
    sensitivities_ci = []
    precisions = []
    precisions_ci = []
    features = []
    feature_ratios = []
    
    df = d.load_base_data_file()  
    for feature in g.FEATURES_PL:
        tn, fp, fn, tp = confusion_matrix(df['rak'], df[feature]).ravel()
        
        # Compute metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True negative rate
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0    # Positive predictive value
        
        sens_ci = confidence_interval(sensitivity, tp + fn, _confidence)
        spec_ci = confidence_interval(specificity, tn + fp, _confidence)
        prec_ci = confidence_interval(precision, tp + fp, _confidence)
        
        features.append(feature)
        sensitivities.append(round(sensitivity,3))
        sensitivities_ci.append(sens_ci)
        specifities.append(round(specificity,3))
        specifities_ci.append(spec_ci)
        precisions.append(round(precision,3))
        precisions_ci.append(prec_ci)
        
        feature_ratios.append(round(len(df[df[feature]==1])/len(df),2))
    
    
    res = pd.DataFrame({
        'feature': features,
        'feature_ratio': feature_ratios,
        'sensitivity': sensitivities,
        'sensitivity_ci': sensitivities_ci,
        'specificity': specifities,
        'specificity_ci': specifities_ci,
        'precision': precisions,
        'precision_ci': precisions_ci,        
        })
    
    # res.to_csv('res.csv', sep=';')

    return res
  
    
  
    
# # #grad cam

# def loss_accuracy(prob, logits, labels):
#     # softmax loss and accurary
#     with tf.variable_scope('Loss_Acc', reuse = False) as scope:
#         loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits,
#                                                                       labels = labels))
#         correct_pred = tf.equal(tf.argmax(prob, 1), tf.argmax(labels, 1))
#         acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#         return loss, acc

# def get_one_hot(label):
#     # convert label to one-hot encoding
#     label = label.reshape(-1, 1)
#     encoder = OneHotEncoder(categories = [range(2)])
#     encoder.fit(label)
#     return encoder.transform(label).toarray()

# def get_placeholders_tensors(_cfg, target_layer_index = -1):
#     # get model's placeholders and tensors
#     # target_layer_index is the index of the conv layer
#     x = tf.placeholder(tf.float32, name = 'x', shape = [None,
#                                                         _cfg['img_size'],
#                                                         _cfg['img_size'],
#                                                         _cfg['channels']])
                                                        
#     y = tf.placeholder(tf.float32, name = 'label', shape = [None, 2])
#     dropout_rate = tf.placeholder(tf.float32, name = 'dropout_rate')
#     optim = SGD(learning_rate=_cfg['learning_rate'])
    
#     connvs, logits, probs = model_cnn2_4grad(_cfg['img_size'], _cfg['img_size'], _cfg['channels'])
#     loss, acc  = loss_accuracy(probs, logits, y)

    
#     grad_cam = Grad_CAM(connvs[target_layer_index], logits, x, y)

#     placeholders_tensors = {'x': x,
#                             'y': y,
#                             'dropout_rate': dropout_rate,
#                             'optimizer': optim,
#                             'probs': probs,
#                             'loss': loss,
#                             'acc': acc,
#                             'grad_cam': grad_cam}
#     return placeholders_tensors

# def Grad_CAM(conv_layer, logits, x, y):
#     # gradient-weighted activation mapping (Grad_CAM) for visualisation
#     with tf.variable_scope('Grad_CAM', reuse = False) as scope:
#         y_c = tf.reduce_sum(tf.multiply(logits, y), axis = 1)
#         conv_layer_grad = tf.gradients(y_c, conv_layer)[0] # 0: weight, 1: bias
#         alpha = tf.reduce_mean(conv_layer_grad, axis = (2, 3)) # feature map importance
#         linear_combination = tf.multiply(tf.reshape(alpha, [-1,
#                                                             alpha.get_shape().as_list()[1],
#                                                             1, 1]), conv_layer)
#         grad_cam = tf.nn.relu(tf.reduce_sum(linear_combination, axis = 1))
#         return grad_cam
    
    
# def get_results_for_visualization(sess, placeholders_tensors, dataset, count):
#     # get images, grad_cams, and predicated probabilites
#     iterator = dataset.make_one_shot_iterator()
#     next_element = iterator.get_next()
#     batch = sess.run(next_element)

#     feed_dictionary = {placeholders_tensors['x']: np.array(batch['image'][:count]),
#                         placeholders_tensors['y']: np.array(get_one_hot(batch['label'][:count])),
#                         placeholders_tensors['dropout_rate']: [0, 0]}
#     probs = sess.run(placeholders_tensors['probs'], feed_dict = feed_dictionary)
#     predicted_label = np.argmax(probs, 1)
#     feed_dictionary = {placeholders_tensors['x']: np.array(batch['image'][:count]),
#                         placeholders_tensors['y']: np.array(get_one_hot(predicted_label)),
#                         placeholders_tensors['dropout_rate']: [0, 0]}
#     grad_cam = sess.run(placeholders_tensors['grad_cam'], feed_dict = feed_dictionary)

#     return np.array(batch['image']), grad_cam, probs, np.array(batch['label'])



# # Replace vanila relu to guided relu to get guided backpropagation.
# import tensorflow as tf

# from tensorflow.python.framework import ops
# from tensorflow.python.ops import gen_nn_ops

# @ops.RegisterGradient("GuidedRelu")
# def _GuidedReluGrad(op, grad):
#     return tf.where(0. < grad, gen_nn_ops._relu_grad(grad, op.outputs[0]), tf.zeros(grad.get_shape()))


# from nets import resnet_v1_101
# latest_checkpoint = "model/resnet_v1_101.ckpt"
# ## Optimistic restore.
# reader = tf.train.NewCheckpointReader(latest_checkpoint)
# saved_shapes = reader.get_variable_to_shape_map()
# variables_to_restore = tf.global_variables()
# for var in variables_to_restore:
#   if not var.name.split(':')[0] in saved_shapes:
#     print("WARNING. Saved weight not exists in checkpoint. Init var:", var.name)
#   else:
#     # print("Load saved weight:", var.name)
#     pass
        