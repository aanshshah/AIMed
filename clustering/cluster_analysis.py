import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.misc import comb

def generate_plot(groupss):
    index = 1
    for group in groups:
        print ("Group size", len(group))
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
        common = ccs_count.most_common()[:15]
        x, y = [], []
        for i,j in common:
            x.append(i)
            y.append(j)

        print(x)
        print(y)
        x_pos = [i for i, _ in enumerate(x)]
        plt.bar(x_pos, y, color='blue')
        plt.xlabel("CCS")
        plt.ylabel("Number of Patients")
        plt.title("CCS Counts for Group " + str(index))
        # plt.ylim(min(y), max(y))
        plt.xticks(x_pos, x)
        plt.savefig('graphs/SOM_CCS_cluster' + str(index) +'.png', bbox_inches='tight')
        index += 1
        plt.clf()
        # plt.show()

def calc_similarity(groups):
    """Calculates the similarity score of each cluster by getting the sum of the set difference

    Args:
        param1 (groups): A list of the dictionaries of clusters (key: CCS code, value: # patients)

    Returns:
        Similarity score. The higher the number, the better

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
    
#[1,2,3]
#[1,4]
if __name__ == "__main__":
    data = pd.read_csv("../data/patient_ccs_som_30x30.csv")
    # data = pd.read_csv("../data/patient_ccs_100.csv")
    groups = data.groupby('cluster_num')['CCS_codes'].apply(list)
    for i in groups:
        print(len(i))
    # score, cnt = calc_similarity(groups)
    # print("Similarity Score:", score)
    # print("Average # of overlapping CCS codes:", cnt/comb(len(groups), 2))
    # generate_plot(groups)