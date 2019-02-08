import random_forest
import logistic_regression
import mimic_readmission
import matplotlib.pyplot as plt
import numpy as np

def evaluate():
	rf_accuracy = random_forest.run()
	ffn_accuracy = mimic_readmission.run()
	lr_accuracy = logistic_regression.run()
	return rf_accuracy, ffn_accuracy, lr_accuracy

def graph():
	objects = ('Random Forest', 'Feed Forward', 'Logistic Regression')
	y_pos = np.arange(len(objects))
	performance = evaluate()
	 
	plt.bar(y_pos, performance, align='center', alpha=0.5)
	plt.xticks(y_pos, objects)
	plt.ylabel('Accuracy')
	plt.title('Classifier')
	 
	plt.show()
	plt.close()

graph()