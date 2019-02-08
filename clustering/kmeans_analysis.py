import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

X = pd.read_csv('../data/x_with_lacefeatures.csv')


k = [x for x in range(1,11)]
inertias = []
for i in range(1, 11):
	kmeans = KMeans(n_clusters=i, random_state=0).fit(X)
	inertias.append(kmeans.inertia_)

colors = ['b', 'g', 'r']
markers = ['o', 'v', 's']


plt.plot(k, inertias, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()