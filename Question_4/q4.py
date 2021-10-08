import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

labeled_samples_with_pca = pd.read_csv('../processed_data.csv')

with open('../columns.pickle', 'rb') as f:
    label_columns,numerical_columns,categorical_columns,PCA_columns = pickle.load(f)


labels = labeled_samples_with_pca[label_columns]
samples = labeled_samples_with_pca[numerical_columns]

db = DBSCAN(eps=10, min_samples=5)
db_clusters = db.fit_predict(samples[:100])

df = labeled_samples_with_pca[:100] 
df['cluster'] = db_clusters

print(df[df.cluster==1])

plt.figure(figsize=(20,20))
for i in range(1,11):
    plt.subplot(5,2,i)
    neighbours = 2*i
    print(neighbours)
    
    nn = NearestNeighbors(n_neighbors = neighbours)
    nbrs = nn.fit(samples)
    distances, indices = nbrs.kneighbors(samples)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    plt.plot(distances)

plt.show()

db = DBSCAN(eps=4, min_samples=6)
db_clusters = db.fit_predict(samples)

db_clusters = list(db_clusters)
for i in range(len(db_clusters)):
    if db_clusters[i]==-1:
        print(labeled_samples_with_pca['Name'][i])

set(list(db_clusters))
ff = pd.DataFrame(db_clusters)
ff.to_csv('dbscan.csv',index=False)