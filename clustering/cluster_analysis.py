import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("../data/patient_ccs_100.csv")
groups = data.groupby('cluster_num')['CCS_codes'].apply(list)

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
    sorted(ccs_count, key=ccs_count.get, reverse=True)[:20]
    print(len(ccs_count))
    # plt.bar(range(len(ccs_count)), list(ccs_count.values()), align='center')
    # plt.xticks(range(len(ccs_count)), list(ccs_count.keys()))
    # plt.show()


