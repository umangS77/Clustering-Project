import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

labeled_samples_with_pca = pd.read_csv('processed_data.csv')

with open('columns.pickle', 'rb') as f:
    label_columns,numerical_columns,categorical_columns,PCA_columns = pickle.load(f)

labels = labeled_samples_with_pca[label_columns]
samples = labeled_samples_with_pca[numerical_columns]

clusters_kmeans = pd.read_csv('Question_2/k_means.csv')['0']
clusters_agg = pd.read_csv('Question_3/agglomerative_clustering.csv')['0']
clusters_divisive = pd.read_csv('Question_3/divisive_clustering.csv')['0']
clusters_dbscan = pd.read_csv('Question_4/dbscan.csv')['0']

all_models = [clusters_kmeans,clusters_agg,clusters_divisive,clusters_dbscan]
model_titles = ['K-Means','Agglomerative hierarchical Clustering','Divisive Clustering','DBSCAN']

plt.figure(figsize=(15,10))
for i,model in enumerate(all_models):
    ax = plt.subplot(2,2,i+1)
    x = labeled_samples_with_pca[PCA_columns[0]]
    y = labeled_samples_with_pca[PCA_columns[1]]
    ax.scatter(x, y,c=list(model), cmap='rainbow',alpha=0.2)
    ax.set_title(model_titles[i])
    ax.set(xlabel='PCA_1', ylabel='PCA_2')
plt.show()


def cos_similarity(a,b):
#     a = np.array(a)
#     b = np.array(b)
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
def euclidean(a,b):
    return np.sqrt(np.sum((a-b)**2))


np_samples = np.array(samples)
similarity_matrix = cosine_similarity(np_samples)

total = np_samples.shape[0]
inter_class_similarities = []
total = 2000
for model_i, model in enumerate(all_models):
    inter = []
    for i in tqdm(range(total)):
        for j in range(i):
            if model[i]==model[j]:
                inter.append(similarity_matrix[i][j])
    inter_class_similarities.append(np.mean(inter))
inter_class_similarities


fig = plt.figure(figsize = (10, 5)) 
plt.bar(model_titles, inter_class_similarities, color ='maroon',  
        width = 0.4) 
plt.title("Inter Class Similarity (More better)")
plt.show()


intra_class_similarity = []
for model_i, model in enumerate(all_models):
    clusters = set([i for i in model if i!=-1])
    intra = []
    for cluster1 in clusters:
        for cluster2 in clusters:
            if cluster1 != cluster2:
                mean1 = np.mean(samples[(model==cluster1)],axis=0)
                mean2 = np.mean(samples[(model==cluster2)],axis=0)
                intra.append(cos_similarity(mean1,mean2))
    intra_class_similarity.append(np.mean(intra))
print("intra_class_similarity = ", str(intra_class_similarity))

fig = plt.figure(figsize = (10,5)) 
plt.bar(model_titles, intra_class_similarity, color ='maroon',  
        width = 0.4) 
plt.title("Intra Class Similarity (lesser better)")
plt.show()

for model_i, model in enumerate(all_models):
    labels = list(samples.columns)
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(20,6))
    clusters = set([i for i in model if i!=-1])
    for i,cluster1 in enumerate(clusters):
        mean = np.mean(samples[(model == cluster1)])
        ax.bar(x+[-width,0,width][i], list(mean), color = 'rbg'[i],align='center', width = width, label=cluster1)
    clusters = list([str(c) for c in clusters])
    ax.set_ylabel('Value')
    ax.set_title(model_titles[model_i])
    ax.set_xticks(x)
    ax.set_xticklabels(labels,rotation='vertical')
    plt.legend()
    plt.show()

# Variance Distribution within clusters
for model_i, model in enumerate(all_models):
    print("For {}:".format(model_titles[model_i]))
    clusters = set([i for i in model if i!=-1])
    print('Number of clusters: ', len(clusters))
    for cluster1 in clusters:
        print()
        print('For Cluster {}:'.format(cluster1))
        clust = samples[(model == cluster1)]
        variances =  np.var(clust)
        total_var = np.sum(variances)
        variances = variances/total_var
        variances = dict(variances)
        variances = [k for k, v in sorted(variances.items(), key=lambda item: item[1])]
#         print(variances)
        print('Similar Attr: ',variances[:5])
        print('Varied Attr: ',variances[-5:])
    print('--------------------------------------------------------')


for model_i, model in enumerate(all_models):
    print("For {}:".format(model_titles[model_i]))
    clusters = set([i for i in model if i!=-1])
    print('Number of clusters: ', len(clusters))
    for cluster1 in clusters:
        print('Players Included in {}: '.format(cluster1), end='')
        clust = labeled_samples_with_pca[(model == cluster1)]
        print(list(clust.Name[:5]))
    print('----------------------')

        
plt.figure(figsize=(15,10))
for model_i, model in enumerate(all_models):
    ax = plt.subplot(2,2,model_i+1)
    clusters = set([i for i in model if i!=-1])
    length = []
    for cluster1 in clusters:
        length.append(labeled_samples_with_pca[(model == cluster1)].shape[0])
    clusters = list([str(c) for c in clusters])
    ax.bar(clusters, length, color ='maroon',  
        width = 0.4) 
    ax.set_title(model_titles[model_i])
    ax.set(xlabel='Clusters', ylabel='Count')
plt.show()

# outliers in DBSCAN

for i in range(len(clusters_dbscan)):
    if clusters_dbscan[i]==-1:
        print(labeled_samples_with_pca['Name'][i])





