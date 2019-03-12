import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.misc import comb

def generate_plot(groups, labels):
    '''
        Generates CCS count bar graphs for each cluster

        Args:
            groups: a 2D array of (clusters, CCS codes)
            labels: a dictionary mapping CCS codes to descriptions
        Returns:
            Saves CCS bar graphs inside graphs folder
    '''
    index = 0
    for group in groups:
        ccs_count = {}
        for codes in group:
            cleaned_codes = codes.strip("()").split(", ")
            for code in cleaned_codes:
                code = code[1:-1]
                if code in ccs_count:
                    ccs_count[code] += 1
                else:
                    ccs_count[code] = 1
        # print(ccs_count)
        ccs_count = Counter(ccs_count)
        common = ccs_count.most_common()[:10]   #10 most common CCS codes
        x, y = [], []
        for i,j in common:
            x.append(labels[i])
            y.append(j)
        
        #Formats labels
        fig, ax = plt.subplots()
        fig.autofmt_xdate()

        x_pos = [i for i, _ in enumerate(x)]
        plt.bar(x_pos, y, color='blue')
        plt.xlabel("CCS")
        plt.ylabel("Number of Patients")
        plt.title("CCS Counts for Group " + str(index))
        plt.xticks(x_pos, x)
        plt.savefig('graphs/Kmodes_CCS_cluster' + str(index) +'.png', bbox_inches='tight')
        index += 1
        plt.clf()
        # plt.show()

def calc_similarity(groups):
    """Calculates the similarity score of each cluster by getting the sum of the set difference

    Args:
        groups: A list of the dictionaries of clusters (key: CCS code, value: # patients)

    Returns:
        score: Similarity score. The higher the number, the better
        cnt_overlap_key: cluster overlap. Counts number of overlapping keys in cluster. The lower, the better

    """
    clusters = [] 
    # create dictionaries for each cluster with key: CCS code, value: # patients
    for group in groups:
        ccs_count = {}
        for codes in group:
            cleaned_codes = codes.strip("()").split(", ")
            for code in cleaned_codes:
                code = code[1:-1]
                if code in ccs_count:
                    ccs_count[code] += 1
                else:
                    ccs_count[code] = 1
        clusters.append(ccs_count)
    
    score = 0 # count number of times 
    cnt_overlap_key = 0
    for i in range(len(clusters)):
        for j in range(i+1, len(clusters)):
            tot_keys = set(list(clusters[i].keys()) + list(clusters[j].keys()))
            for key in tot_keys:
                if key in clusters[i] and key not in clusters[j]:
                    score += clusters[i][key]
                elif key in clusters[j] and key not in clusters[i]:
                    score += clusters[j][key]
                else:
                    score += abs(clusters[i][key] - clusters[j][key])
                    cnt_overlap_key += 1
    return score, cnt_overlap_key
    
def get_labels(df):
    '''
        Maps a dictionary of icd codes to disease

        Args:
            df: ICD dataframe
        Returns: 
            dictionary (code to disease)
    '''
    df['CCS_clean_category'] = df["\'CCS CATEGORY\'"].str.strip(' "\'\t\r\n')
    df['clean_desc'] = df["\'CCS CATEGORY DESCRIPTION\'"].str.strip(' "\'\t\r\n')
    return dict(zip(df["CCS_clean_category"], df['clean_desc']))


if __name__ == "__main__":
    # data = pd.read_csv("../data/patient_ccs_som_30x30.csv")
    icd = pd.read_csv("../data/CCS.csv", skiprows=[0,2])
    labels = get_labels(icd)
    data = pd.read_csv("../data/patient_ccs_100.csv")

    #generate graphs
    groups = data.groupby('cluster_num')['CCS_codes'].apply(list)
    generate_plot(groups, labels)


    #similarity scores
    # score, cnt = calc_similarity(groups)
    # print("Similarity Score:", score)
    # print("Average # of overlapping CCS codes:", cnt/comb(len(groups), 2))
