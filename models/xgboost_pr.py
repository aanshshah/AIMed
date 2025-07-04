#supress warnings (especially from sklearn)
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import os
from tqdm import tqdm
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, LeaveOneOut
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, average_precision_score, precision_recall_curve, roc_curve, auc, precision_score, roc_curve, confusion_matrix, precision_recall_fscore_support, f1_score, precision_score, recall_score
import csv
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from xgboost import plot_importance
from scipy import interp
import re
from io import StringIO


# import joblib

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


def store_cluster_info(y_pred, y_real, name, cluster):
    filename = 'results/'+name+'_cluster_'+str(cluster)+'_withoutPCA.csv'
    y_test = np.array(y_real).flatten()
    y_preds = np.array(y_pred).flatten()
    # np.around(y_preds.astype(np.float))
    classification_report_csv(y_test.any(), y_preds.any(), filename)

def run_xgboost(optimize=True):
    dfs, dfs_labels = preprocess()
    filepath = 'results/figures/'
    xgb_opt = XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=0.7,
           gamma=0.1, learning_rate=0.1, max_delta_step=0, max_depth=3,
           min_child_weight=5, missing=None, n_estimators=70, nthread=4,
           objective='binary:logistic', reg_alpha=25.0, reg_lambda=1,
           scale_pos_weight=7.0909090909090908, seed=1, silent=True,
           subsample=0.6)
    for cluster, x_df in enumerate(dfs):
        if cluster == 0:
            xgb_opt = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
               colsample_bytree=0.8, gamma=0.0, learning_rate=0.1,
               max_delta_step=0, max_depth=2, min_child_weight=3, missing=None,
               n_estimators=50, n_jobs=1, nthread=4, objective='binary:logistic',
               random_state=0, reg_alpha=0, reg_lambda=1,
               scale_pos_weight=10.186375321336762, seed=1, silent=True,
               subsample=0.4)
        elif cluster == 1:
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
        y_df = dfs_labels[cluster]
        print('RUN_XGBOOST CLUSTER: '+str(cluster))  
        K = 5
        eval_size = int(np.round(1./K))
        skf = StratifiedKFold(n_splits=K)
        fig = plt.figure(figsize=(7,7))
        y_predications = []
        
        name = 'XGBoost'
        fold = 0

        # entries = {'avg_precision':[],
        #            'avg_recall': [],
        #            'avg_f_score': [],
        #             'avg_support': [] }
        r = {'real': [],
            'pred': []}
        prediction = np.array([])
        reals = np.array([])
        for train_indices, test_indices in skf.split(x_df, y_df):
            X_train, y_train = x_df.iloc[train_indices], y_df.iloc[train_indices]
            X_valid, y_valid = x_df.iloc[test_indices], y_df.iloc[test_indices]
            # print(X_valid.shape)/
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
            # precision, recall, f_score, support = precision_recall_fscore_support(y_valid, xgb_opt_pred_prob)
            # exists = entries['avg_precision']
            # exists.append(precision)
            # entries['avg_precision'] = exists
            # exists = entries['avg_recall']
            # exists.append(recall)
            # entries['avg_recall'] = exists
            # exists = entries['avg_f_score']
            # exists.append(f_score)
            # entries['avg_f_score'] = exists
            # exists = entries['avg_support']
            # exists.append(support)
            # entries['avg_support'] = exists

            fold += 1
        # results = {}
        # for key, values in entries.items():
        #     results[key] = [np.array(values).mean()]
        # print(results)
        # results_pd = pd.DataFrame(results)
        # for key, value in r.items():r[key] = np.array(value).flatten()
        # real = numpy_fillna(reals)
        # pred = numpy_fillna(prediction)
        # reals = numpy_fillna(reals)
        # prediction = numpy_fillna(prediction)
        # print(reals.shape, prediction.shape)
        full_path = 'results/'+name+'_cluster_'+str(cluster)+'_withoutPCA.csv'
        # print(np.unique(r['pred']))
        report = classification_report(reals, prediction)
        classification_report_csv(report, full_path)
        # results_pd.to_csv(full_path, index=False)
def numpy_fillna(data):
    # Get lengths of each row of data
    # for i in data:
    #     print(i)
    lens = np.array([len(i) for i in data])

    # Mask of valid places in each row
    mask = np.arange(lens.max()) < lens[:,None]

    # Setup output array and put elements from data into masked positions
    out = np.zeros(mask.shape, dtype=data.dtype)
    out[mask] = np.concatenate(data)
    return out
if __name__ == '__main__':
    run_xgboost()
