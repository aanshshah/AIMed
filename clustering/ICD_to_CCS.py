import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfTransformer
from kmodes.kmodes import KModes
from collections import Counter

#import csv files as dataframes
ccs = pd.read_csv('../data/CCS.csv', skiprows=[0,2])
patient_icd9 = pd.read_csv('../data/patient_icd9.csv')

def icd9_dict():
    '''
        Input: None
        Output: Outputs a dictionary mapping icd9 codes to CCS codes
		
        Create mapping of icd9 codes (key) to CCS codes (value) for quick O(1) lookups
	'''
    codes = {}
    for index,row in ccs.iterrows():
        ccs_code = row["'CCS CATEGORY'"].strip(' "\'\t\r\n')
        icd9_code = row["'ICD-9-CM CODE'"].strip(' "\'\t\r\n')
        codes[icd9_code] = ccs_code
    
    return codes

def icd9_to_ccs(icd9_codes, icd9_dict):
    '''
        Input: list of icd9 codes and a dict mapping icd9 code to CCS code
        Output: a sorted tuple of corresponding CCS codes

        Converts list of ICD9 codes to CCS codes
    '''
    #clean input and find matching ICD9 code
    patient_codes = icd9_codes.strip("[]").split(", ")
    ccs_codes = set()

    for patient_code in patient_codes:
        if patient_code[1:-1] not in icd9_dict: #sometimes code is 'nan'
            continue
        code = icd9_dict[patient_code[1:-1]] #strip quotes off of patient_code
        ccs_codes.add(code)

    combo = tuple(sorted(ccs_codes))
    return combo

if __name__ == "__main__":
    icd9_map = icd9_dict()
    # combos = {}
    index_dict, index = {}, 0 #Stores index mapping for BOW
    num_features = 275 #number of unique CCS codes (Goes from 15072 dim to 275)
    bow = np.zeros((patient_icd9.shape[0], num_features))
    ccs_column = [] #create list of CCS codes to add as a df feature 
    #create BOW array
    for i, row in patient_icd9.iterrows():
        patient_codes = row['ICD9_CODES']
        combo = icd9_to_ccs(patient_codes, icd9_map)
        ccs_column.append(combo)
        for c in combo:
            if c not in index_dict:
                index_dict[c] = index
                index += 1
            bow[i, index_dict[c]] = 1

    #34 is the largest number of codes for a single stay
  
    #TruncatedSVD
    # svd = TruncatedSVD(n_components=34, n_iter=20, random_state=42)
    # svd.fit(bow)
    # bow = svd.transform(bow)
    # print(svd.explained_variance_ratio_.sum())  
    # transformer = TfidfTransformer().fit(bow)
    # bow = transformer.transform(bow)
    # print(bow)

    #PCA
    # pca = PCA(n_components=50)
    # pca.fit(bow)
    # bow = pca.transform(bow)
    # print(pca.explained_variance_)

    # bow = TSNE(n_components=2).fit_transform(bow)

    # K means stuff
    # inertias = []
    # k = [x for x in range(1,11)]
    # for i in range(1, 11):
        #K means
        # kmeans = KMeans(n_clusters=i, random_state=0).fit(bow)
        # print(kmeans.inertia_)
        # inertias.append(kmeans.inertia_)

    # K Modes
    km = KModes(n_clusters=3, init='Huang', n_init=100, verbose=1)
    clusters = km.fit_predict(bow)
    print(Counter(clusters))

    #add new features to df
    patient_icd9['cluster_num'] = clusters
    patient_icd9['CCS_codes'] = ccs_column
    patient_icd9.to_csv("patient_ccs_100.csv")

    # print(km.cost_)
    # inertias.append(km.cost_)

    # colors = ['b', 'g', 'r']
    # markers = ['o', 'v', 's']


    # plt.plot(k, inertias, 'bx-')
    # plt.xlabel('k')
    # plt.ylabel('Distortion')
    # plt.title('K Modes')
    # plt.show()

    








