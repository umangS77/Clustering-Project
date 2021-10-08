import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import pickle
# import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

labeled_samples_with_pca = pd.read_csv('../processed_data.csv')

with open('../columns.pickle', 'rb') as f:
    label_columns,numerical_columns,categorical_columns,PCA_columns = pickle.load(f)

labels = labeled_samples_with_pca[label_columns]
samples = labeled_samples_with_pca[numerical_columns]

linked = linkage(samples[:200], 'complete')

plt.figure(figsize=(20, 40))
dendrogram(linked,
            orientation='left',
            labels=list(labels.Name[:200]),
            distance_sort='descending',
            leaf_font_size = 13)
# plt.tick_params(axis='x', which='major', labelsize=15)
# plt.tick_params(axis='y', which='major', labelsize=10)
plt.show()


linked_complete = linkage(samples, 'complete')
ff = fcluster(linked_complete, t=3, criterion='maxclust')
ff = pd.DataFrame(ff)
ff.to_csv('agglomerative_clustering.csv',index=False)

def get_sse(data):
    data = np.array(data)
    error = np.sum((data-np.mean(data,axis=0))**2)
    return error
    
# def distance(data,r1,r2):
#     first = data[data.cluster==r1].drop('cluster',axis=1)
#     second = data[data.cluster==r2].drop('cluster',axis=1)
    
#     return sum([get_sse(first),get_sse(second)])
    
def divisive(data):
    total = data.shape[0]
    num = total*2-2
    data['cluster'] = num
    link = []
    sse = {}
    sse[num] = get_sse(data.drop('cluster',axis=1))
    to_splits = []
    for jj in tqdm(range(total-1)):
        to_split = max(sse, key=sse.get)
#         grouped = data.groupby('cluster')
#         for name, group in grouped:
#             if(group.shape[0]>1):
#                 sse[name] = get_sse(group.drop(['cluster'],axis=1))
        
        split_index = (data.cluster==to_split)
        to_split_data = data[split_index].drop('cluster',axis=1)
        parent_size = to_split_data.shape[0]
        kmeans = KMeans(n_clusters=2)
        kmeans = kmeans.fit(to_split_data)
        new_clusters = list(kmeans.predict(to_split_data))
        
        indexes_taken = [i for i, n in enumerate(split_index) if n]
        if new_clusters.count(1)==1:
            replace_1 = indexes_taken[new_clusters.index(1)]
        else:
            num -= 1
            replace_1 = num
        if new_clusters.count(0)==1:
            replace_0 = indexes_taken[new_clusters.index(0)]
        else:
            num -= 1
            replace_0 = num
        new_clusters = [replace_1 if j==1 else replace_0 for j in new_clusters]
        data.loc[split_index, 'cluster'] = new_clusters
        to_splits.append(to_split)        
        first = get_sse(data[data.cluster==replace_1].drop('cluster',axis=1))
        second = get_sse(data[data.cluster==replace_0].drop('cluster',axis=1))
        sse[replace_1] = first
        sse[replace_0] = second
        # using as approximation of actual distance (original error - final error, greater the difference further the clusters)
        # average linkage was giving non-decreasing error
        # anyways purpose of distance is only to plot the data
        dist = float(sse[to_split]-first-second)
                       
        link.append([replace_0,replace_1,float(sse[to_split]-dist),parent_size])
        sse.pop(to_split)
    link.reverse()
    to_splits.reverse()
    replacement_id = {}
    for jj in range(total):
        replacement_id[jj]=jj
    for jj,n in enumerate(to_splits):
        replacement_id[n] = jj+total
        to_splits[jj] = jj+total
    new_link = np.array(link)
    for i,line in enumerate(new_link):
        new_link[i][0] = replacement_id[line[0]]
        new_link[i][1] = replacement_id[line[1]]
    new_link[:,2] = np.sqrt(new_link[:,2])
    return to_splits,new_link


n = 200
first = samples[:n].copy()
to_splits,linked_ = divisive(first)
plt.figure(figsize=(20, 40))
dendrogram(linked_,
            orientation='left',
            labels=list(labels.Name[:n]),
            distance_sort='descending',
            leaf_font_size = 13)
plt.show()

linked_complete_d = divisive(samples)

ff_d = fcluster(linked_complete_d[1], t=3, criterion='maxclust')
ff_d = pd.DataFrame(ff_d)
ff_d.to_csv('divisive_clustering.csv',index=False)







