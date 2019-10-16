import requests
import xml.etree.ElementTree as ET
import pandas as pd
from itertools import combinations 
import time

def query_results(cmb):
    '''
    Takes in a tuple of comorbidities and returns the number of articles on pubmed
    that contains those keywords
    '''
    print("Results for", cmb, end='')
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term="
    param = cmb[0].replace(" ", "+") + "+AND+" + cmb[1].replace(" ", "+")
    try:
        res=requests.get(base_url + param)
        xml = (res.text)
        root = ET.fromstring(xml)
        print(root[0].text)
    except:
        print("Error - could not find")
    time.sleep(0.2)

ccs = pd.read_csv('../data/CCS.csv', skiprows=[0,2])
pairs = list(combinations(ccs["'CCS CATEGORY DESCRIPTION'"].unique(),2))
for pair in pairs:
    pair = (pair[0].replace("'", ""), pair[1].replace("'", ""))
    query_results(pair)
