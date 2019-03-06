#supress warnings (especially from sklearn)
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings("ignore", category=DeprecationWarning)
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
from sklearn.metrics import classification_report, average_precision_score, precision_recall_curve, roc_curve, auc, precision_score, roc_curve, confusion_matrix, precision_recall_fscore_support, f1_score, precision_score, recall_score, accuracy_score
import csv
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from xgboost import plot_importance
from scipy import interp
from sklearn.metrics.scorer import make_scorer
# import joblib

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
    filename = 'results/alldata_results_xgboost.csv'
    classification_report_csv(y_real, y_preds, name, filename, new_file)

def preprocess():
    df = pd.read_csv('../data/x_with_lacefeatures.csv')
    df = df.drop(['subject_id', 'hadm_id'], axis=1)
    y = pd.read_csv('../data/y_more_no_df_clean.csv')
    return df, y

def run_xgboost(optimize=True):
    x_df, y_df = preprocess()
    if optimize:
        xgb_opt = opt_xgboost(x_df, y_df, optimize)
    else:
        xgb_opt = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
        colsample_bytree=0.9, gamma=0.0, learning_rate=0.1,
        max_delta_step=0, max_depth=3, min_child_weight=7, missing=None,
        n_estimators=210, n_jobs=1, nthread=4, objective='binary:logistic',
        random_state=0, reg_alpha=25.0, reg_lambda=1,
        scale_pos_weight=8.285971685971687, seed=1, silent=True,
        subsample=0.9)
    K = 5
    eval_size = int(np.round(1./K))
    skf = StratifiedKFold(n_splits=K)

    prediction = np.array([])
    reals = np.array([])
    name = 'XGBoost'
    for train_indices, test_indices in skf.split(x_df, y_df):
        X_train, y_train = x_df.iloc[train_indices], y_df.iloc[train_indices]
        X_valid, y_valid = x_df.iloc[test_indices], y_df.iloc[test_indices]
        class_weight_scale = 1.*y_df['label'].value_counts()[0]/y_df['label'].value_counts()[1]
        print('class weight scale : {}'.format(class_weight_scale))
        xgb_opt.set_params(**{'scale_pos_weight' : class_weight_scale})
        xgb_opt.fit(X_train,y_train)
        xgb_opt_pred_prob = xgb_opt.predict_proba(X_valid)[:, 1]

        y_valid = y_valid.values.tolist()
            
        reals = np.append(reals,y_valid)
        reals = reals.astype(int)
        
        prediction = np.append(prediction, xgb_opt_pred_prob)
        prediction = prediction.astype(int)
    store_cluster_info(prediction, reals, name, new_file=False)
    # store_cluster_info(y_predications, y_real, name, cluster) 

def opt_xgboost(x_df, y_df, optimize=True):
    # Define the class weight scale (a hyperparameter) as the ration of negative labels to positive labels.
    # This instructs the classifier to address the class imbalance.
    class_weight_scale = 1.*y_df['label'].value_counts()[0]/y_df['label'].value_counts()[1]
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
        class_weight_scale = 1.*y_df['label'].value_counts()[0]/y_df['label'].value_counts()[1]
        xgb1.set_params(**{'scale_pos_weight' : class_weight_scale})
        xgb1.fit(X_train,y_train)

    X_train = x_df
    y_train = y_df

    if optimize:
        scorers = {
        'precision_score': make_scorer(precision_score)
        }

        param_test0 = {
         'n_estimators':range(50,250,10)
        }
        print('performing hyperparamter optimization step 0')
        gsearch0 = GridSearchCV(estimator = xgb1, param_grid = param_test0, scoring=scorers,iid=False, cv=5, refit='precision_score')
        gsearch0.fit(X_train,y_train)
        print(gsearch0.best_params_, gsearch0.best_score_)

        param_test1 = {
         'max_depth':range(1,10),
         'min_child_weight':range(1,10)
        }
        print('performing hyperparamter optimization step 1')
        gsearch1 = GridSearchCV(estimator = gsearch0.best_estimator_,
         param_grid = param_test1, scoring=scorers, iid=False, cv=5, refit='precision_score')
        gsearch1.fit(X_train,y_train)
        print(gsearch1.best_params_, gsearch1.best_score_)

        max_d = gsearch1.best_params_['max_depth']
        min_c = gsearch1.best_params_['min_child_weight']

        param_test2 = {
         'gamma':[i/10. for i in range(0,5)]
        }
        print('performing hyperparamter optimization step 2')
        gsearch2 = GridSearchCV(estimator = gsearch1.best_estimator_, 
         param_grid = param_test2, scoring=scorers, iid=False, cv=5, refit='precision_score')
        gsearch2.fit(X_train,y_train)
        print(gsearch2.best_params_, gsearch2.best_score_)

        param_test3 = {
            'subsample':[i/10.0 for i in range(1,10)],
            'colsample_bytree':[i/10.0 for i in range(1,10)]
        }
        print('performing hyperparamter optimization step 3')
        gsearch3 = GridSearchCV(estimator = gsearch2.best_estimator_, 
         param_grid = param_test3, scoring=scorers, iid=False, cv=5, refit='precision_score')
        gsearch3.fit(X_train,y_train)
        print(gsearch3.best_params_, gsearch3.best_score_)

        param_test4 = {
            'reg_alpha':[0, 1e-5, 1e-3, 0.1, 10]
        }
        print('performing hyperparamter optimization step 4')
        gsearch4 = GridSearchCV(estimator = gsearch3.best_estimator_, 
         param_grid = param_test4, scoring=scorers, iid=False, cv=5, refit='precision_score')
        gsearch4.fit(X_train,y_train)
        print(gsearch4.best_params_, gsearch4.best_score_)

        alpha = gsearch4.best_params_['reg_alpha']
        if alpha != 0:
            param_test4b = {
                'reg_alpha':[0.1*alpha, 0.25*alpha, 0.5*alpha, alpha, 2.5*alpha, 5*alpha, 10*alpha]
            }
            print('performing hyperparamter optimization step 4b')
            gsearch4b = GridSearchCV(estimator = gsearch4.best_estimator_, 
             param_grid = param_test4b, scoring=scorers,iid=False, cv=5, refit='precision_score')
            gsearch4b.fit(X_train,y_train)
            print(gsearch4b.best_params_, gsearch4.best_score_)
            print('\nParameter optimization finished!')
            xgb_opt = gsearch4b.best_estimator_
        else:
            xgb_opt = gsearch4.best_estimator_
    else: 
        # Pre-optimized settings
        xgb_opt = XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=0.7,
           gamma=0.1, learning_rate=0.1, max_delta_step=0, max_depth=3,
           min_child_weight=5, missing=None, n_estimators=70, nthread=4,
           objective='binary:logistic', reg_alpha=25.0, reg_lambda=1,
           scale_pos_weight=7.0909090909090908, seed=1, silent=True,
           subsample=0.6)
    print('OPTIMAL FOR ALL DATA:')
    print(xgb_opt)
    #save model
    return xgb_opt

if __name__ == '__main__':
    run_xgboost(optimize=True)

