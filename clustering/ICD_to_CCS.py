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
import sompy
from sklearn.metrics import silhouette_samples, silhouette_score

#import csv files as dataframes
ccs = pd.read_csv('../data/CCS.csv', skiprows=[0,2])
patient_icd9 = pd.read_csv('../data/patient_icd9.csv')

def icd9_dict():
    '''
        Create mapping of icd9 codes (key) to CCS codes (value) for quick O(1) lookups

        Args:
            None
        Output: 
            codes: a dictionary mapping icd9 codes to CCS codes		
	'''
    codes = {}
    for index,row in ccs.iterrows():
        ccs_code = row["'CCS CATEGORY'"].strip(' "\'\t\r\n')
        icd9_code = row["'ICD-9-CM CODE'"].strip(' "\'\t\r\n')
        codes[icd9_code] = ccs_code
    
    return codes

def icd9_to_ccs(icd9_codes, icd9_dict):
    '''
        Converts list of ICD9 codes to CCS codes

        Args: 
            icd9_codes: list of icd9 codes 
            icd9_dict: a dict mapping icd9 code to CCS code
        Returns:
            a sorted tuple of corresponding CCS codes
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

def apply_tsvd(data, n_components=50, n_iter=20):
    '''
        Applies Truncated SVD to data and transforms using TFIDF

        Args:
            data: 2D matrix with features
        Returns: 
            data: 2D matrix of tranformed features with dimensionality reduction
    '''
    svd = TruncatedSVD(n_components=n_components, n_iter=n_iter, random_state=42)
    svd.fit(data)
    data = svd.transform(data)
    print(svd.explained_variance_ratio_.sum())  
    transformer = TfidfTransformer().fit(data)
    data = transformer.transform(data)
    return data

def apply_pca(data, n_components=50):
    '''
        Applies PCA to data and transforms using TFIDF

        Args:
            data: 2D matrix with features
        Returns:
            data: 2D matrix of tranformed features with dimensionality reduction
    '''
    pca = PCA(n_components=n_components)
    pca.fit(data)
    data = pca.transform(data)
    print(pca.explained_variance_)
    return data

def kmodes(patient_icd9, bow):

    '''
        Uses kmodes to make clusters for dataset, plots cost graph, and saves clusters to new csv file

        Args:
            patient_icd9: dataframe of patient_icd9 csv
            bow: 2D matrix of bag-of-words of CCS codes
        Returns:
            Saves new data as csv file and graphs the elbow line graph for cost to find best k
    '''  
    

    # K Modes
    km = KModes(n_clusters=3, init='Huang', n_init=100, verbose=1)
    clusters = km.fit_predict(bow)
    print(Counter(clusters))

    #add new features to df
    patient_icd9['cluster_num'] = clusters
    patient_icd9['CCS_codes'] = ccs_column
    patient_icd9.to_csv("patient_ccs_100.csv")

    print(km.cost_)
    inertias.append(km.cost_)

    colors = ['b', 'g', 'r']
    markers = ['o', 'v', 's']


    plt.plot(k, inertias, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('K Modes')
    plt.show()

# def som():

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

    #SOM
    mapsize = [30,30]   #5*sqrt(#row)
    som = sompy.SOMFactory.build(bow, mapsize, mask=None, mapshape='planar', lattice='rect', normalization='var', initialization='pca', neighborhood='gaussian', training='batch', name='sompy')  # this will use the default parameters, but i can change the initialization and neighborhood methods
    # som.train(n_job=1, verbose='info', train_rough_len=50, train_finetune_len=100)  # verbose='debug' will print more, and verbose=None wont print anything
    som.train(n_job=1, train_rough_len=2, train_finetune_len=2)  # verbose='debug' will print more, and verbose=None wont print anything
    map_labels = som.cluster()
    data_labels = np.array([map_labels[int(k)] for k in som._bmu[0]])
    # print(data_labels.shape, np.unique(data_labels))

    #save as csv
    # patient_icd9['cluster_num'] = data_labels
    # patient_icd9['CCS_codes'] = ccs_column
    # patient_icd9.to_csv("../data/patient_ccs_som_30x30.csv")

    
    

    








