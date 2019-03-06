#supress warnings (especially from sklearn)
import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV,StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import csv
from scipy import interp
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import classification_report, average_precision_score, precision_recall_curve, roc_curve, auc, precision_score, roc_curve, confusion_matrix, precision_recall_fscore_support, f1_score, precision_score, recall_score, accuracy_score
from keras.losses import binary_crossentropy
from keras.activations import softmax, relu
from sklearn.preprocessing import StandardScaler
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from sklearn.metrics.scorer import make_scorer
import xgboost as xgb
from xgboost.sklearn import XGBClassifier

def classification_report_csv(y_test, y_preds, name, full_path, new_file=True):
    report = classification_report(y_test, y_preds)
    report = [x.strip().split() for x in report.strip().split('\n') if x]
    report[0] = ['model','class'] + report[0]
    report[3] = [''.join(report[3][0:3])] + report[3][3:]
    data = np.array([[name] + report[i] for i in range(1,len(report))])
    report_df = pd.DataFrame(data, columns=np.array(report)[0])
    if new_file: report_df.to_csv(full_path, index = False)
    else: report_df.to_csv(open(full_path, 'a'), index=False, header=False)

def store_cluster_info(y_preds, y_real, name, cluster, new_file=True):
    filename = 'results/'+str(cluster)+'_cluster_results.csv'
    classification_report_csv(y_real, y_preds, name, filename, new_file)

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
    model.compile(loss=loss, optimizer=optimizer(lr=lr), metrics=[sensitivity])
    return model

def create_pipeline_baseline():
    pipeline = []
    skf = StratifiedKFold(n_splits=2)
    lr = GridSearchCV(LogisticRegression(random_state = 0), cv=skf, verbose=0, param_grid={})
    rf = GridSearchCV(RandomForestClassifier(random_state=0), cv=skf, verbose=0, param_grid={})
    pipeline = [['LogisticRegression', lr], ['RandomForest',rf]]
    return pipeline

def create_pipeline_nn():
    param_grid = {'clf__lr': [0.01],
     'clf__first_neuron':[8, 16, 32, 64, 128, 256],
     'clf__second_neuron':[1, 8, 32, 64, 128],
    'clf__last_neuron':[1],
     'clf__batch_size': [64, 128],
     'clf__epochs': [20],
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
    'precision_score': make_scorer(precision_score),
    'recall_score': make_scorer(recall_score),
    'accuracy_score': make_scorer(accuracy_score)
    }
    refit_score='precision_score'
    skf = StratifiedKFold(n_splits=3)

    grid = GridSearchCV(pipeline, cv=skf, param_grid=param_grid, verbose=3, scoring=scorers,refit=refit_score)
    return grid

def run_pipeline():
    dfs, dfs_labels = preprocess()
    print('finished preprocess')
    nn_grid = create_pipeline_nn()
    baseline_grid = create_pipeline_baseline()
    print('created pipeline and running ...')
    for i, df in enumerate(dfs):
        print('CLUSTER: '+str(i))
        labels = dfs_labels[i]

        opt_df, _, labels, _ = train_test_split(df, labels, test_size=0)
        
        K.clear_session()
        nn_grid.fit(opt_df, labels)
        K.clear_session()

        nn_opt = nn_grid.best_estimator_

        pd.DataFrame(nn_grid.cv_results_).to_csv('neuralnet_optimal_final_cluster_'+str(i)+'.csv')

        X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.25)
        nn_opt.fit(X_train, y_train)
        y_pred = nn_opt.predict(X_test)
        name = 'Neural_Network'
        store_cluster_info(y_pred, y_test, name, i, new_file=True)
        
        for name, grid in baseline_grid:
            grid.fit(X_train, y_train)
            y_pred = grid.predict(X_test)
            store_cluster_info(y_pred, y_test, name, i, new_file=False)
        
        print('RUN_XGBOOST CLUSTER: '+str(i))
        if i == 0:
            xgb_opt = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
               colsample_bytree=0.8, gamma=0.0, learning_rate=0.1,
               max_delta_step=0, max_depth=2, min_child_weight=3, missing=None,
               n_estimators=50, n_jobs=1, nthread=4, objective='binary:logistic',
               random_state=0, reg_alpha=0, reg_lambda=1,
               scale_pos_weight=10.186375321336762, seed=1, silent=True,
               subsample=0.4)
        elif i == 1:
            xgb_opt = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
               colsample_bytree=0.7, gamma=0.0, learning_rate=0.1,
               max_delta_step=0, max_depth=3, min_child_weight=3, missing=None,
               n_estimators=60, n_jobs=1, nthread=4, objective='binary:logistic',
               random_state=0, reg_alpha=0.01, reg_lambda=1,
               scale_pos_weight=8.653391412570006, seed=1, silent=True,
               subsample=0.8)
        else:
            xgb_opt = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
               colsample_bytree=0.7, gamma=0.0, learning_rate=0.1,
               max_delta_step=0, max_depth=3, min_child_weight=5, missing=None,
               n_estimators=50, n_jobs=1, nthread=4, objective='binary:logistic',
               random_state=0, reg_alpha=10, reg_lambda=1,
               scale_pos_weight=5.418508287292818, seed=1, silent=True,
               subsample=0.9)
        
        skf = StratifiedKFold(n_splits=2)        
        name = 'XGBoost'
        fold = 0
        prediction = np.array([])
        reals = np.array([])
        for train_indices, test_indices in skf.split(df, labels):
            print('FOLD: ' + str(fold))
            X_train, y_train = x_df.iloc[train_indices], y_df.iloc[train_indices]
            X_valid, y_valid = x_df.iloc[test_indices], y_df.iloc[test_indices]

            class_weight_scale = 1.*y_train.value_counts()[0]/y_train.value_counts()[1]
            xgb_opt.set_params(**{'scale_pos_weight' : class_weight_scale})
            xgb_opt.fit(X_train,y_train)
            xgb_opt_pred_prob = xgb_opt.predict_proba(X_valid)[:, 1]
            xgb_opt_pred_prob = np.around(xgb_opt_pred_prob)

            y_valid = y_valid.tolist()
            
            reals = np.append(reals,y_valid)
            reals = reals.astype(int)
            
            prediction = np.append(prediction, xgb_opt_pred_prob)
            prediction = prediction.astype(int)
            fold += 1
        store_cluster_info(prediction, reals, name, i, new_file=False)

if __name__ == '__main__':
    run_pipeline()
