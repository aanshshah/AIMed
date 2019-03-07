#supress warnings (especially from sklearn)
def warn(*args, **kwargs):
    pass 
import warnings 
warnings.warn = warn 
import os 
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split, GridSearchCV,StratifiedKFold, cross_validate 
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.pipeline import Pipeline 
import csv 
from scipy import interp 
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier 
from sklearn.metrics import classification_report, average_precision_score, precision_recall_curve, roc_curve, auc, precision_score, roc_curve, confusion_matrix, precision_recall_fscore_support, f1_score, precision_score, recall_score, accuracy_score 
from tensorflow.keras.losses import binary_crossentropy 
from tensorflow.keras.activations import softmax, relu 
from sklearn.preprocessing import StandardScaler 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout 
from tensorflow.keras.optimizers import Adam 
from sklearn.metrics.scorer import make_scorer 
import xgboost as xgb 
from xgboost.sklearn import XGBClassifier 
from tensorflow.keras.backend import clear_session 
import tensorflow 
from tensorflow.compat.v1 import ConfigProto 
from tensorflow.compat.v1 import InteractiveSession 
from tensorflow.keras.backend import set_session 

def classification_report_csv(y_test, y_preds, name, full_path, new_file=True):
    report = classification_report(y_test, y_preds)
    report = [x.strip().split() for x in report.strip().split('\n') if x]
    report[0] = ['model','class'] + report[0]
    report[3] = [''.join(report[3][0:3])] + report[3][3:]
    data = np.array([[name] + report[i] for i in range(1,len(report))])
    report_df = pd.DataFrame(data, columns=np.array(report)[0])
    if new_file: report_df.to_csv(full_path, index = False)
    else: report_df.to_csv(open(full_path, 'a'), index=False, header=False) 

def store_cluster_info(y_preds, y_real, name, new_file=True):
    filename = 'results/neuralnet_alldata_results.csv'
    classification_report_csv(y_real, y_preds, name, filename, new_file) 

def preprocess():
    df = pd.read_csv('../data/x_lace_df.csv', index_col=0)
    df = df.drop(['subject_id', 'hadm_id'], axis=1)
    y = pd.read_csv('../data/y_more_no_df_clean.csv')
    return df, y 

def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon()) 

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon()) 

def specificity_at_sensitivity(sensitivity, **kwargs):
    def metric(labels, predictions):
        # any tensorflow metric
        value, update_op = tensorflow.metrics.specificity_at_sensitivity(labels, predictions, sensitivity, **kwargs)

        # find all variables created for this metric
        metric_vars = [i for i in tensorflow.local_variables() if 'specificity_at_sensitivity' in i.name.split('/')[2]]

        # Add metric variables to GLOBAL_VARIABLES collection.
        # They will be initialized for new session.
        for v in metric_vars:
            tensorflow.add_to_collection(tensorflow.GraphKeys.GLOBAL_VARIABLES, v)

        # force to update metric values
        with tensorflow.control_dependencies([update_op]):
            value = tensorflow.identity(value)
            return value
    return metric

def create_model(first_neuron=64, second_neuron=32, second_activation='relu',
                 last_neuron=1, last_activation='relu', loss='binary_crossentropy',
                 optimizer='adam', lr=0.01, dropout=0.2):
    model = Sequential()
    model.add(Dense(first_neuron,input_dim=63,activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(second_neuron,activation=second_activation))
    model.add(Dense(last_neuron,activation=last_activation))
    model.compile(loss=loss, optimizer=optimizer(lr=lr), metrics=[ 'acc'])
    return model 

def create_pipeline_baseline():
    pipeline = []
    skf = StratifiedKFold(n_splits=3)
    lr = GridSearchCV(LogisticRegression(random_state = 0), cv=skf, verbose=0, param_grid={})
    rf = GridSearchCV(RandomForestClassifier(random_state=0), cv=skf, verbose=0, param_grid={})
    pipeline = [['LogisticRegression', lr], ['RandomForest',rf]]
    return pipeline 

def create_pipeline_nn():
    param_grid = {'clf__lr': [0.01],
     'clf__first_neuron':[1, 8, 16, 32, 64, 128, 256],
     'clf__second_neuron':[1, 8, 16, 32, 64, 128],
    'clf__last_neuron':[1],
     'clf__batch_size': [64, 128],
     'clf__epochs': [50],
     'clf__dropout': [0, 0.2],
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
    scorers = {
    'precision_score': make_scorer(precision_score)
    # 'recall_score': make_scorer(recall_score), 'accuracy_score': make_scorer(accuracy_score)
    }
    refit_score='precision_score'
    skf = StratifiedKFold(n_splits=3)
    grid = GridSearchCV(pipeline, cv=skf, param_grid=param_grid, verbose=3, scoring=scorers,refit=refit_score)
    return grid 

def run_pipeline():
    df, labels = preprocess()
    print('finished preprocessing')
    opt_df, _, labels, _ = train_test_split(df, labels, test_size=0)
    nn_grid = create_pipeline_nn()
    print('created pipeline and running ...')
    nn_grid.fit(opt_df, labels)
    nn_opt = nn_grid.best_estimator_
    print(nn_opt)
    pd.DataFrame(nn_grid.cv_results_).to_csv('neuralnet_optimal_all_data.csv')
    print('fitting based on optimal parameters')
    X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.15)
    nn_opt.fit(X_train, y_train)
    y_pred = nn_opt.predict(X_test)
    name = 'Neural_Network'
    store_cluster_info(y_pred, y_test, name, new_file=True) 

config = ConfigProto(allow_soft_placement=True) 
config.gpu_options.per_process_gpu_memory_fraction = 1.0 
session = InteractiveSession(config=config) 
set_session(session)
run_pipeline()

