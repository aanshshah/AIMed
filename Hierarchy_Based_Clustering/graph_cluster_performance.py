import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import json

"""
1. traverse results directory
2. split each file into <depth>, <range>
3. map range to the disease string 
4. plot everything
"""
picd9_to_label = {}
codes_file = open('/Users/Aansh/Documents/Brown /Research/AIMed/icd9/codes.json')
codes = json.load(codes_file)
"""
Maps each disease to parent in ICD9 hierarchy. In absence of a parent, maps to None.
Returns mapping of parents to children and nodes to their parents
"""
def create_graphs(n):
    for i in range(len(codes)):
        grouping = codes[i]
        parent = None
        for j in range(len(grouping)):
            if grouping[j]['depth'] == n and grouping[j].get('descr'): 
                parent = grouping[j]['code']
                picd9_to_label[parent] = grouping[j]['descr']

    results_path = 'results/'
    dirs = ['depth_1_xgb_performance','depth_2_xgb_performance']
    dirs = ['depth_1_xgb_performance']
    do_not_visit = {'mean_results.csv', 'median_results.csv', '.DS_STORE'}
    scores = ['f1_score', 'precision', 'recall', 'support']
    readmit_str = '    Readmitted'
    for d in dirs:
        readmit_dict = {}
        noadmit_dict = {}
        path = os.path.join(results_path, d)
        files = os.listdir(path)
        for file in files:
            if file not in do_not_visit:
                depth, cluster_range = file.split()
                cluster_range = cluster_range[:-4]
                results = pd.read_csv(os.path.join(path, file), names=scores, header=None,skiprows=[0])
                readmit_curr = []
                noadmit_curr = []
                for metric in scores:
                    readmit_curr.append(results[metric][readmit_str])
                    noadmit_curr.append(results[metric]['Not Readmitted'])
                readmit_dict[cluster_range] = readmit_curr+[picd9_to_label[cluster_range]]
                noadmit_dict[cluster_range] = noadmit_curr+[picd9_to_label[cluster_range]]
        readmit_df = pd.DataFrame.from_dict(readmit_dict, columns=['f1_score', 'precision', 'recall', 'support', 'cluster'], orient='index')
        noadmit_df = pd.DataFrame.from_dict(noadmit_dict, columns=['f1_score', 'precision', 'recall', 'support', 'cluster'], orient='index')
    fig = px.scatter(readmit_df, x='precision', y='recall', hover_data=['f1_score', 'cluster'], title="Readmitted Performance with Depth {0}".format(n))
    fig2 = px.scatter(noadmit_df, x='precision', y='recall', hover_data=['f1_score', 'cluster'], title="Not Readmitted Performance with Depth {0}".format(n))
    return fig, fig2
figs = create_graphs(1) + create_graphs(2)

for fig in figs:
    fig.show()




