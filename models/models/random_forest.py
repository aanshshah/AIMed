from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import numpy as np


def run():
	################
	# READING DATA #
	################

	# read comments & annotations from respective csv files (stored locally)
	comments = pd.read_csv('x_more_no_df_clean.csv')
	annotations = pd.read_csv('y_more_no_df_clean.csv')
	comments = comments.reset_index()
	# idx = np.random.permutation(comments.index)
	# comments = comments.reindex(idx)
	# annotations = annotations.reindex(idx)
	print("Comments", comments.shape)
	print("annotations", annotations.shape)
	print("TOT", annotations.sum()/comments.shape[0])
	###########################
	# BUILDING THE CLASSIFIER #
	###########################

	# split into train and test sets
	x_train, x_test, y_train, y_test = train_test_split(comments, annotations, test_size=.8,random_state = 0)

	###########################
	# RANDOM FOREST:          #
	###########################
	rf = RandomForestClassifier(n_estimators=20, oob_score=True, random_state=0)
	rf.fit(x_train, y_train)
	predicted = rf.predict(x_test)
	accuracy = accuracy_score(y_test, predicted)
	print(f'Out-of-bag score estimate: {rf.oob_score_:.3}')
	print(f'Mean accuracy score: {accuracy:.3}')
	###########################
# FEATURE IMPORTANCE:     #
###########################
	# Feature Importance
	features = list(comments.columns)
	importances = rf.feature_importances_
	indices = np.argsort(importances)[-5:]

	plt.title('Feature Importances')
	plt.barh(range(len(indices)), importances[indices], color='b', align='center')
	plt.yticks(range(len(indices)), [features[i] for i in indices])
	plt.xlabel('Relative Importance')
	plt.show()
	plt.savefig('random_forest_importance.png')
	plt.close()
	return accuracy
print(run())


