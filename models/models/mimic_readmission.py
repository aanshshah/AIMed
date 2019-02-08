

from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Convolution1D, Flatten, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import TensorBoard

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
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
from keras.wrappers.scikit_learn import KerasClassifier
import eli5

from eli5.sklearn import PermutationImportance

import shap
from IPython.display import display, HTML

def build_model():
    model = Sequential()
    model.add(Dense(1,activation='sigmoid',input_shape=(68,)))
    model.add(Dropout(0.2))
    model.add(Dense(1,activation='sigmoid'))
    model.add(Dropout(0.2))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

comments = pd.read_csv('readmission/notebooks/x_df_clean.csv')
annotations = pd.read_csv('readmission/notebooks/y_df_clean.csv')

comments = comments.reset_index()

x_train, x_test, y_train, y_test = train_test_split(comments, annotations, test_size=.2)
def eval_model():
	model = build_model()
	model.fit(x_train, y_train, epochs=1, callbacks=[tensorBoardCallback])
	scores = model.evaluate(x_test, y_test, verbose=0)
	print("Accuracy: %.2f%%" % (scores[1]*100))
	return scores[1]

estimator = KerasClassifier(build_fn = build_model, validation_split=0.1, batch_size = 100, epochs=2, verbose=0)
history = estimator.fit(x_train, y_train)

# perm = PermutationImportance(estimator, random_state=1).fit(x_train,y_train)
# eli5.show_weights(perm, feature_names =x_train.columns.tolist())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'dev'], loc='upper left')
plt.show()

def f_wrapper(X):
    return estimator.predict(X).flatten()

X_train_summary = shap.kmeans(x_train, 20)
explainer = shap.KernelExplainer(f_wrapper, X_train_summary)

x_train_sample = x_train.sample(50)
shap.initjs()
shap_values = explainer.shap_values(x_train_sample)
shap.summary_plot(shap_values, x_train_sample)
shap.summary_plot(shap_values, x_train_sample, plot_type="bar")
shap.force_plot(explainer.expected_value, shap_values, x_test, link="logit")