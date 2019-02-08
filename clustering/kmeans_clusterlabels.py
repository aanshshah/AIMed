import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
import collections
from sklearn.preprocessing import StandardScaler
import csv

def preprocess_once():
	'''
		We do standard preprocessing from all the data we have. Then we look at all the icd9 codes from 
		each patient diagnoses. We map each patient and hospital stay to icd9 code attributed to patient
		condition. We then add it to a csv file with the rest of the features. 
	'''
	MICU_admits_clean = pd.read_csv('../data/MICU_admits_clean.csv')
	MICU_admits_clean.fillna(0, inplace=True)
	will_readmit = (MICU_admits_clean.future_readmit == 'Yes')
	y_df = pd.DataFrame(will_readmit.map({True: 1, False: 0}))
	y_df.columns = ['label']
	x_df = MICU_admits_clean.drop(['admittime','dischtime','first_careunit','last_careunit','readmit_dt','readmit_last_careunit', 'future_readmit', 'next_readmit_dt'], axis = 1)
	x_df_ids = x_df[['subject_id', 'hadm_id']].copy()
	x_df = x_df.drop(['subject_id', 'hadm_id'], axis=1)
	x_df_cat = x_df[['gender', 'marital_status', 'insurance']].copy()
	x_df_num = x_df.drop(['gender', 'marital_status', 'insurance'], axis = 1)

	scaled_x_df_num = pd.DataFrame(StandardScaler().fit_transform(x_df_num), columns=x_df_num.keys())
	clean_scaled_x_df_num = scaled_x_df_num
	clean_x_df_cat = x_df_cat#x_df_cat.drop(x_df_cat.index[outliers]).reset_index(drop = True)
	clean_x_df_cat_ohe = pd.get_dummies(clean_x_df_cat, drop_first=True)
	clean_x_df_ids = x_df_ids
	x_df = pd.concat([scaled_x_df_num, clean_x_df_cat_ohe, x_df_ids], axis = 1)
	x_df.to_csv('../data/x_lace_df.csv')
	x = pd.read_csv('../data/x_lace_df.csv')
	all_patients = set()
	for index, row in x.iterrows():
	    subject_id = row['subject_id']
	    hadm_id = row['hadm_id']
	    all_patients.add((subject_id, hadm_id))
	diagnoses = pd.read_csv('../data/DIAGNOSES_ICD.csv')
	patient_icd9 = {}
	all_codes = set()
	for index, row in diagnoses.iterrows():
	    subject_id = row['SUBJECT_ID']
	    hadm_id = row['HADM_ID']
	    if (subject_id, hadm_id) in all_patients:
	        icd9_code = str(row['ICD9_CODE'])
	        all_codes.add(icd9_code)
	        exists = patient_icd9.get((subject_id, hadm_id), [])
	        exists.append(icd9_code)
	        patient_icd9[(subject_id, hadm_id)] = exists

	X = pd.read_csv('../data/x_with_lacefeatures.csv')
	with open('../data/patient_icd9.csv', 'w') as csv_file:
	    writer = csv.writer(csv_file)
	    writer.writerow(['SUBJECT_ID', 'HADM_ID', 'ICD9_CODES'])
	    for key, value in patient_icd9.items():
	       writer.writerow([key[0], key[1], value])
	X['ICD9_CODE'] = 0
	X['ICD9_CODE'] = X['ICD9_CODE'].astype(object)
	for index, row in X.iterrows():	
		subject_id = row['subject_id']
		hadm_id = row['hadm_id']
		exists = patient_icd9.get((subject_id, hadm_id), [])
		if exists:
			print(index, exists)
			X.loc[index, 'ICD9_CODE'] = np.array(exists)
	X.to_csv('../data/x_with_icd9.csv')

def kmeans_cluster():
	X = pd.read_csv('../data/x_with_icd9.csv')
	patient_icd9 = open('../data/patient_icd9.csv')
	k = 4

	estimator = KMeans(n_clusters=k, random_state=0).fit(X)
	
	icd9_map = {}
	for index, row in X.iterrows():	
		subject_id = row['SUBJECT_ID']
		hadm_id = row['HADM_ID']
		label = k_means.labels_[index]

		codes = row['ICD9_CODE']
		icd9_code = icd9_map.get(label, [])
		icd9_code.append(codes)
	return icd9_code

def main():
	preprocess_once()
	# print(kmeans_cluster())

if __name__ == '__main__':
	main()