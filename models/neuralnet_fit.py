#supress warnings (especially from sklearn)
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
import csv
from scipy import interp
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.metrics import binary_accuracy
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.activations import softmax, relu
from sklearn.preprocessing import StandardScaler
from keras import backend as K
from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

def preprocess():
    data = pd.read_csv('../data/labeled_clustered_data.csv')
    df_0 = data[(data[['cluster_num']] == 0).any(axis=1)]
    df_0_label=df_0.pop('label')
    df_1 = data[(data[['cluster_num']] == 1).any(axis=1)]
    df_1_label=df_1.pop('label')
    df_2 = data[(data[['cluster_num']] == 2).any(axis=1)]
    df_2_label=df_2.pop('label')
    dfs = [df_0, df_1, df_2]
    dfs_labels = [df_0_label, df_1_label, df_2_label]
    return dfs, dfs_labels

def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

def create_model(first_neuron=64, second_neuron=32, second_activation='relu', 
                 last_neuron=1, last_activation='relu', loss='binary_crossentropy', 
                 optimizer='adam', lr=0.01, dropout=0.2):
    model = Sequential()
    model.add(Dense(first_neuron,input_dim=71,activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(second_neuron,activation=second_activation))
    model.add(Dense(last_neuron,activation=last_activation))
    model.compile(loss=loss, optimizer=optimizer(lr=lr), metrics=["accuracy"])
    return model

def create_pipeline():
    param_grid = {'clf__lr': [0.001, 0.01],
     'clf__first_neuron':[8, 32, 6, 128],
     'clf__second_neuron':[1, 8, 32, 64],
    'clf__last_neuron':[1],
     'clf__batch_size': [128],
     'clf__epochs': [20],
     'clf__dropout': [0, 0.1, 0.2],
     'clf__optimizer': [Adam],
     'clf__loss': [binary_crossentropy],
          'clf__second_activation': [relu],
     'clf__last_activation': [relu]}
    clf = KerasClassifier(build_fn=create_model, verbose=0)
    scaler = StandardScaler()
    pipeline = Pipeline([
        ('preprocess',scaler),
        ('clf',clf)
    ])
    grid = GridSearchCV(pipeline, cv=5, param_grid=param_grid)
    return grid

def run_pipeline():
    dfs, dfs_labels = preprocess()
    print('finished preprocess')
    grid = create_pipeline()
    print('created pipeline and running ...')
    for i, df in enumerate(dfs):
        print('CLUSTER: '+str(i))
        labels = dfs_labels[i]
        reduced_df = df
        x_train, x_test, y_train, y_test = train_test_split(reduced_df, labels, test_size=.2)
        clear_session()
        grid.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)
        clear_session()
        print("Best: %f using %s" % (grid.best_score_, grid.best_params_))
        pd.DataFrame(grid.cv_results_).to_csv('neuralnet_optimal.csv')

if __name__ == '__main__':
    run_pipeline()
