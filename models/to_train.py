import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dropout, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.optimizers import Adam, Nadam, SGD, Adagrad
from keras.activations import softmax, relu
from keras.losses import binary_crossentropy
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

comments = pd.read_csv('x_no_index.csv')
annotations = pd.read_csv('y_more_no_df_clean.csv')

x_train, x_test, y_train, y_test = train_test_split(comments, annotations, test_size=.8)

def create_model(first_neuron=64, second_neuron=32, second_activation='relu', 
                 last_neuron=1, last_activation='relu', loss='binary_crossentropy', 
                 optimizer='adam', dropout=0.2):
    model = Sequential()
    model.add(Dense(first_neuron,input_dim=63,activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(second_neuron,activation=second_activation))
    model.add(Dropout(dropout))
    model.add(Dense(last_neuron,activation=last_activation))
    #need to optimize beta_1, beta_2, epsilon, decay, amsgrad
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model
param_grid = {
     'clf__first_neuron':[32, 64, 128, 256],
     'clf__second_neuron':[32, 64, 128],
    'clf__last_neuron':[1, 32],
     'clf__batch_size': [64, 128],
     'clf__epochs': [12],
     'clf__dropout': [0, 0.2],
     'clf__optimizer': [Adam],
     'clf__loss': [binary_crossentropy],
          'clf__second_activation': [relu],
     'clf__last_activation': [relu]}
clf = KerasClassifier(build_fn=create_model, verbose=0)
scaler = StandardScaler()
pipeline = Pipeline([('preprocess',scaler), ('clf',clf)])
grid = GridSearchCV(pipeline, cv=3, param_grid=param_grid, verbose=10)
grid.fit(x_train, y_train)
pd.DataFrame(grid.cv_results_).to_csv('test_optimal.csv')