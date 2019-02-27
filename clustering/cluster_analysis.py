import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def generate_plot():
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
        plt.savefig('CCS_cluster' + str(index) +'.png', bbox_inches='tight')
        index += 1
        plt.clf()
        # plt.show()

if __name__ == "__main__":
    data = pd.read_csv("../data/patient_ccs_100.csv")
    groups = data.groupby('cluster_num')['CCS_codes'].apply(list)
    generate_plot(groups)

