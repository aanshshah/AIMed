from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Convolution1D, Flatten, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import TensorBoard


import numpy as np
import pandas as pd
import math
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.utils import to_categorical
from keras.regularizers import L1L2



def run():
	################
	# READING DATA #
	################

	# read comments & annotations from respective csv files (stored locally)
	comments = pd.read_csv('readmission/notebooks/x_df_clean.csv')
	annotations = pd.read_csv('readmission/notebooks/y_df_clean.csv')
	comments = comments.reset_index()

	###########################
	# BUILDING THE CLASSIFIER #
	###########################

	# split into train and test sets
	x_train, x_test, y_train, y_test = train_test_split(comments, annotations, test_size=.3,random_state = 0)
	# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
	embedding_vecor_length = 12
	model = Sequential()

	###########################
	# CLASSIFIER REGRESSION:  #
	###########################
	y_train = to_categorical(y_train)
	y_test = to_categorical(y_test)
	model.add(Dense(2,  # output dim is 2, one score per each class
	                activation='softmax',
	                kernel_regularizer=L1L2(l1=0.0, l2=0.1),
	                input_dim=x_train.shape[1]))  # input dimension = number of features your data has
	model.compile(optimizer='sgd',
	              loss='categorical_crossentropy',
	              metrics=['accuracy'])
	model.fit(x_train, y_train, epochs=100)

	score = model.evaluate(x_test, y_test, verbose=0)

	# print('Test score:', score[0])
	# print('Test accuracy:', score[1])
	return score[1]
# Log to tensorboard

print(run())
