
#supress warnings (especially from sklearn)
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, LeaveOneOut
from sklearn.pipeline import Pipeline
import csv
from scipy import interp
from keras.wrappers.scikit_learn import KerasClassifier
from keras.metrics import binary_accuracy
from sklearn.metrics import classification_report, average_precision_score, precision_recall_curve, roc_curve, auc, precision_score, roc_curve, confusion_matrix, precision_recall_fscore_support, f1_score, precision_score, recall_score
from keras.losses import binary_crossentropy
from keras.activations import softmax, relu
from sklearn.preprocessing import StandardScaler
from keras import backend as K
from keras.models import Sequential
import re
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from io import StringIO

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

def create_model_cluster0():
    dropout = 0
    first_neuron = 32
    last_neuron = 1
    second_neuron = 8
    model = Sequential()
    model.add(Dense(first_neuron,input_dim=71,activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(second_neuron,activation='relu'))
    model.add(Dense(last_neuron,activation='relu'))
    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['balanced_accuracy_score'])
    return model

def create_model_cluster2():

    dropout = 0
    first_neuron = 8
    last_neuron = 1
    second_neuron = 1

    model = Sequential()
    model.add(Dense(first_neuron,input_dim=71,activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(second_neuron,activation='relu'))
    model.add(Dense(last_neuron,activation='relu'))
    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['balanced_accuracy_score'])
    return model

def create_model_cluster1():

    dropout = 0
    first_neuron = 8
    last_neuron = 1
    second_neuron = 8
    model = Sequential()
    model.add(Dense(first_neuron,input_dim=71,activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(second_neuron,activation='relu'))
    model.add(Dense(last_neuron,activation='relu'))
    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['balanced_accuracy_score'])
    return model

def classification_report_csv(ground_truth,predictions,full_path="test_pandas.csv"):
    # ground_truth = [item for sublist in ground_truth for item in sublist]
    # predictions = [item for sublist in predictions for item in sublist]
    # ground_truth = [1 if i >= 0.5 else 0 for i in ground_truth]
    # predictions = [1 if i >= 0.5 else 0 for i in predictions]
    # print(ground_truth)
    # print(predictions)
    f_score = f1_score(ground_truth, predictions, average="macro")
    precision = precision_score(ground_truth, predictions, average="macro")
    recall = recall_score(ground_truth, predictions, average="macro")
    results_pd = pd.DataFrame({"class": labels,
                               "precision": precision,
                               "recall": recall,
                               "f_score": f_score
                               })
    results_pd.to_csv(full_path, index=False)

def classification_report_csv(report, full_path):
    report = re.sub(r" +", " ", report).replace("avg / total", "avg/total").replace("\n ", "\n")
    report_df = pd.read_csv(StringIO("Classes" + report), sep=' ', index_col=0)        
    report_df.to_csv(full_path, index = False)

def run_pipeline():
    dfs, dfs_labels = preprocess()
    filepath = 'results/figures/'
    for cluster, x_df in enumerate(dfs):
        if cluster == 0:
            model = KerasClassifier(build_fn=create_model_cluster0, epochs=20, batch_size=128, verbose=0)
        elif cluster == 1:
            model = KerasClassifier(build_fn=create_model_cluster1, epochs=20, batch_size=64, verbose=0)
        else:
            model = KerasClassifier(build_fn=create_model_cluster2, epochs=20, batch_size=64, verbose=0)
        y_df = dfs_labels[cluster]
        print('RUN Neural Network CLUSTER: '+str(cluster))  
        K = 5
        eval_size = int(np.round(1./K))
        skf = StratifiedKFold(n_splits=K)
        y_predications = []
        
        name = 'NeuralNet'
        fold = 0
        prediction = np.array([])
        reals = np.array([])
        X_train, X_valid, y_train, y_valid = train_test_split(x_df, y_df, test_size=.2)
        model.fit(X_train,y_train)
        prediction = model.predict(X_valid)
        full_path = 'results/'+name+'_cluster_'+str(cluster)+'_withoutPCA.csv'
        report = classification_report(y_valid, prediction)
        classification_report_csv(report, full_path)
    
    

if __name__ == '__main__':
    run_pipeline()
