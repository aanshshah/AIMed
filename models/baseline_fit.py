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
from sklearn.metrics import classification_report, average_precision_score, precision_recall_curve, roc_curve, auc, precision_score, roc_curve, confusion_matrix
import seaborn as sns
import csv
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from xgboost import plot_importance
from scipy import interp
import joblib

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

def classifaction_report_csv(report, filename):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split('      ')
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv(filename, index = False)

def create_pipeline():
    pipeline = []
    lr = LogisticRegressionCV(random_state = 0, n_jobs=-1)
    en = ElasticNetCV(random_state = 0, n_jobs=-1)
    rf = RandomForestClassifier(random_state=0, n_jobs=-1)
    pipeline.append(['logistic_regression', lr]) 
    pipeline.append(['elastic_network', en]) 
    pipeline.append(['random_forest', rf])
    return pipeline

# def pr_curve(pr_map, dimension_reduction=True):
#     for cluster, clf_map in pr_map.items():
#         fig=plt.figure()
#         if dimension_reduction:
#             plt.title('PR Curve for Cluster '+str(cluster))
#             filepath = 'results/figures/'+str(cluster)+'cluster_PR.png'
#         else:
#             plt.title('PR Curve for Cluster '+str(cluster)+' without PCA')
#             filepath = 'results/figures/'+str(cluster)+'cluster_PR_withoutPCA.png'
#         for name, values in clf_map.items():
#                 average_precision = values[0]
#                 precision = values[1]
#                 recall = values[2]
#                 plt.plot(recall, precision, label=name+' (area = {:.3f})'.format(average_precision))
#         plt.xlabel('Recall')
#         plt.ylabel('Precision')
#         plt.legend(loc='best')
#         plt.show()
#         fig.savefig(filepath)
#         plt.close()

def store_cluster_info(y_pred, y_real, name, cluster):
    filename = 'results/'+name+'_cluster_'+str(cluster)+'_withoutPCA.csv'
    y_test = np.concatenate(y_real)
    y_preds = np.concatenate(y_pred)
    average_precision = average_precision_score(y_test, y_preds)
    precision, recall, _ = precision_recall_curve(y_test, y_preds)
    report = classification_report(y_test, y_preds, target_names=['Not Readmitted', 'Readmitted'])
    classifaction_report_csv(report, filename)
    # pr_curve(pr_map, dimension_reduction)

def run_pipeline(dimension_reduction=False):
    K = 5
    skf = StratifiedKFold(n_splits=K)
    dfs, dfs_labels = preprocess()
    pipeline = create_pipeline()


    for i, df in enumerate(dfs):
        labels = dfs_labels[i]
        reduced_df = df
        fold = 0

        for name, clf in pipeline:
            y_real = []
            y_predication = []
            directory = 'results/'
            filename = directory+name+'_cluster_'+str(i)+'_withoutPCA_fold_'+str(fold)+'.csv'
            
            fold_pr = clf_pr.get(name, [])
            fold = 0
            for train_indices, test_indices in skf.split(reduced_df, labels):
                
                Xtrain, Ytrain = reduced_df.iloc[train_indices], labels.iloc[train_indices]
                Xvalid, Yvalid = reduced_df.iloc[test_indices], labels.iloc[test_indices]
                y_real.append(Yvalid)

                clf.fit(Xtrain, Ytrain)
                y_preds = [0 if x < 0.5 else 1 for x in clf.predict(Xvalid)]

                y_predication.append(y_preds)

                
                fold+=1
            store_cluster_info(y_predication, y_real, name, i)
            model_name = name+'_cluster_'+str(i)'.joblib'
            joblib.dump(clf, model_name)
if __name__ == '__main__':
    run_pipeline(dimension_reduction=False)