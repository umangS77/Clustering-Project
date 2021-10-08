import numpy as np
import pandas as pd
import pickle
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import seaborn as sns
from sklearn.metrics import silhouette_samples, silhouette_score



labeled_samples_with_pca = pd.read_csv('../processed_data.csv')

with open('../columns.pickle', 'rb') as f:
    label_columns,numerical_columns,categorical_columns,PCA_columns = pickle.load(f)


labels = labeled_samples_with_pca[label_columns]
samples = labeled_samples_with_pca[numerical_columns+categorical_columns]
ksamples = labeled_samples_with_pca[numerical_columns]

kmean = np.array(ksamples)
kmean.shape

def get_clusters(kmean, k):
    prev_err=1e7
    curr_err = 1e6
    flag = 0
    centre = random.sample(range(len(kmean)), k)
    centres = [kmean[i] for i in centre]
    centres = np.array(centres)

    while(prev_err-curr_err>=1e-7):
        if flag:
            centres = []
            for i in range(k):
                centre = np.mean(kmean[np.argwhere(clusters==i)], axis=0)
                centres.append(centre)
            centres = np.array(centres)
            centres = centres.reshape((centres.shape[0], centres.shape[2]))
        else:
            flag = 1
        clusters = []
        prev_err=curr_err
        curr_err=0

        for i in range(k):
            cluster = np.abs(kmean-centres[i])**2
            cluster = np.sum(cluster, axis=1)**(1/2)
            clusters.append(cluster)
        clusters = np.array(clusters)
        clusters = np.argmin(clusters, axis = 0)
        curr_err = abs(kmean-centres[clusters])**2
        curr_err = sum(np.sum(curr_err, axis=1)**(1/2))
    return curr_err, clusters

clusterfile = open("Cluster_data.txt", "w")
k_list = [3, 5, 7]
for k in k_list:
    clusterfile.write("\n")
    clusterfile.write(f"For k = {k}")
    curr_err, clusters = get_clusters(kmean, k)
    for clstr in range(k):
        cls = []
        clusterfile.write("\n")
        clusterfile.write(f"Cluster Number:  {clstr}")
        for i in range(len(kmean)):
            if clusters[i]==clstr:
                cls.append([labeled_samples_with_pca['Name'][i], labeled_samples_with_pca['Overall'][i], labeled_samples_with_pca['Position'][i]])
        
        clusterfile.write(",".join(str(item) for item in cls[:20]))

clusterfile.close()

total_cluster=[]
err_list=[]
for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    curr_err, clusters = get_clusters(kmean, k)
    total_cluster.append(clusters)
    err_list.append(curr_err)

# len(total_cluster)

k=[]
scores=[]
for i in range(9):
    k.append(i+2)
    scores.append(silhouette_score(kmean, total_cluster[i+1]))

plt.plot(k, scores) 
plt.xlabel('k') 
plt.ylabel('Silhoutte score') 
plt.show()


plt.plot(range(1,11), err_list)
plt.xlabel('K')
plt.ylabel('Total Error')
plt.show()

new_cluster = total_cluster[1]
ff_k = pd.DataFrame(new_cluster)
ff_k.to_csv('k_means.csv',index=False)