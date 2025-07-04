#supress warnings (especially from sklearn)
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, LeaveOneOut
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, average_precision_score, precision_recall_curve, roc_curve, auc, precision_score, roc_curve, confusion_matrix, precision_recall_fscore_support
import csv
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from xgboost import plot_importance
from scipy import interp

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

def run_xgboost(optimize=True):
    dfs, dfs_labels = preprocess()
    filepath = 'results/figures/'
    for cluster, x_df in enumerate(dfs):
        y_df = dfs_labels[cluster]
        print('RUN_XGBOOST CLUSTER: '+str(cluster))  
        xgb_opt = opt_xgboost(cluster, x_df, y_df, optimize)
        print('OPTIMAL: ')
        print(xgb_opt)
        
def opt_xgboost(cluster, x_df, y_df, optimize=True):
    # Define the class weight scale (a hyperparameter) as the ration of negative labels to positive labels.
    # This instructs the classifier to address the class imbalance.
    class_weight_scale = 1.*y_df.value_counts()[0]/y_df.value_counts()[1]
    filepath = 'results/figures/'
    # Setting minimal required initial hyperparameters
    param={
        'objective':'binary:logistic',
        'nthread':4,
        'scale_pos_weight':class_weight_scale,
        'seed' : 1   
    }
    xgb1 = XGBClassifier()
    xgb1.set_params(**param)
    K = 5
    eval_size = int(np.round(1./K))
    skf = StratifiedKFold(n_splits=K)

    for train_indices, test_indices in skf.split(x_df, y_df):
        X_train, y_train = x_df.iloc[train_indices], y_df.iloc[train_indices]
        X_valid, y_valid = x_df.iloc[test_indices], y_df.iloc[test_indices]
        class_weight_scale = 1.*y_train.value_counts()[0]/y_train.value_counts()[1]
        xgb1.set_params(**{'scale_pos_weight' : class_weight_scale})
        xgb1.fit(X_train,y_train)
        xgb1_pred_prob = xgb1.predict_proba(X_valid)

    X_train = x_df
    y_train = y_df

    if optimize:

        param_test0 = {
         'n_estimators':range(50,250,10)
        }
        print('performing hyperparamter optimization step 0')
        gsearch0 = GridSearchCV(estimator = xgb1, param_grid = param_test0, scoring='roc_auc',n_jobs=-1,iid=False, cv=5)
        gsearch0.fit(X_train,y_train)
        print(gsearch0.best_params_, gsearch0.best_score_)

        param_test1 = {
         'max_depth':range(1,10),
         'min_child_weight':range(1,10)
        }
        print('performing hyperparamter optimization step 1')
        gsearch1 = GridSearchCV(estimator = gsearch0.best_estimator_,
         param_grid = param_test1, scoring='roc_auc',n_jobs=-1,iid=False, cv=5)
        gsearch1.fit(X_train,y_train)
        print(gsearch1.best_params_, gsearch1.best_score_)

        max_d = gsearch1.best_params_['max_depth']
        min_c = gsearch1.best_params_['min_child_weight']

        param_test2 = {
         'gamma':[i/10. for i in range(0,5)]
        }
        print('performing hyperparamter optimization step 2')
        gsearch2 = GridSearchCV(estimator = gsearch1.best_estimator_, 
         param_grid = param_test2, scoring='roc_auc',n_jobs=-1,iid=False, cv=5)
        gsearch2.fit(X_train,y_train)
        print(gsearch2.best_params_, gsearch2.best_score_)

        param_test3 = {
            'subsample':[i/10.0 for i in range(1,10)],
            'colsample_bytree':[i/10.0 for i in range(1,10)]
        }
        print('performing hyperparamter optimization step 3')
        gsearch3 = GridSearchCV(estimator = gsearch2.best_estimator_, 
         param_grid = param_test3, scoring='roc_auc',n_jobs=-1,iid=False, cv=5)
        gsearch3.fit(X_train,y_train)
        print(gsearch3.best_params_, gsearch3.best_score_)

        param_test4 = {
            'reg_alpha':[0, 1e-5, 1e-3, 0.1, 10]
        }
        print('performing hyperparamter optimization step 4')
        gsearch4 = GridSearchCV(estimator = gsearch3.best_estimator_, 
         param_grid = param_test4, scoring='roc_auc',n_jobs=-1,iid=False, cv=5)
        gsearch4.fit(X_train,y_train)
        print(gsearch4.best_params_, gsearch4.best_score_)

        alpha = gsearch4.best_params_['reg_alpha']
        if alpha != 0:
            param_test4b = {
                'reg_alpha':[0.1*alpha, 0.25*alpha, 0.5*alpha, alpha, 2.5*alpha, 5*alpha, 10*alpha]
            }
            print('performing hyperparamter optimization step 4b')
            gsearch4b = GridSearchCV(estimator = gsearch4.best_estimator_, 
             param_grid = param_test4b, scoring='roc_auc',n_jobs=-1,iid=False, cv=5)
            gsearch4b.fit(X_train,y_train)
            print(gsearch4b.best_params_, gsearch4.best_score_)
            print('\nParameter optimization finished!')
            xgb_opt = gsearch4b.best_estimator_
            xgb_opt
        else:
            xgb_opt = gsearch4.best_estimator_
            xgb_opt
    else: 
        # Pre-optimized settings
        xgb_opt = XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=0.7,
           gamma=0.1, learning_rate=0.1, max_delta_step=0, max_depth=3,
           min_child_weight=5, missing=None, n_estimators=70, nthread=4,
           objective='binary:logistic', reg_alpha=25.0, reg_lambda=1,
           scale_pos_weight=7.0909090909090908, seed=1, silent=True,
           subsample=0.6)

    #save model
    return xgb_opt

if __name__ == '__main__':
    run_xgboost()


