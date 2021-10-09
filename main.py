import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

pd.options.mode.chained_assignment = None

with open('columns.pickle', 'rb') as f:
    lab_cols,num_cols,cat_cols,PCA_cols = pickle.load(f)

def cos_sim(a,b):
    v = (np.linalg.norm(a)*np.linalg.norm(b))
    return np.dot(a, b)/v

PCA_samples = pd.read_csv('processed_data.csv')

# clusters_kmeans = pd.read_csv('Question_2/k_means.csv')['0']
# clusters_agg = pd.read_csv('Question_3/agglomerative_clustering.csv')['0']
# clusters_divisive = pd.read_csv('Question_3/divisive_clustering.csv')['0']
# clusters_dbscan = pd.read_csv('Question_4/dbscan.csv')['0']


labels = PCA_samples[lab_cols]
data_samples = PCA_samples[num_cols]

# model_graphs = [clusters_kmeans,clusters_agg,clusters_divisive,clusters_dbscan]
model_graphs = [pd.read_csv('Question_2/k_means.csv')['0'],pd.read_csv('Question_3/agglomerative_clustering.csv')['0'],pd.read_csv('Question_3/divisive_clustering.csv')['0'],pd.read_csv('Question_4/dbscan.csv')['0']]

graph_headings = ['K-Means','Agglomerative hierarchical Clustering','Divisive Clustering','DBSCAN']

def euclidean(a,b):
    return np.sqrt(np.sum((a-b)**2))


def print_plot1():
    for mi, m in enumerate(model_graphs):
        labels = list(data_samples.columns)

        fig, ax = plt.subplots(figsize=(20,6))
        x = np.arange(len(labels))
        clusters = set([i for i in m if i!=-1])
        width = 0.35
        for i,c1 in enumerate(clusters):
            ax.bar(x+[-width,0,width][i], list(np.mean(data_samples[(m == c1)])), color = 'rbg'[i],align='center', width = width, label=c1)
        
        ax.set_title(graph_headings[mi])
        ax.set_xticks(x)
        ax.set_xticklabels(labels,rotation='vertical')
        ax.set_ylabel('Value')
        plt.legend()
        plt.show()
        clusters = list([str(cl) for cl in clusters])


plt.figure(figsize=(20,15))

for i,m in enumerate(model_graphs):
    j = i+1
    dim = 2
    ax = plt.subplot(dim,dim,j)
    ax.set(xlabel='PCA_1', ylabel='PCA_2')
    ax.scatter(PCA_samples[PCA_cols[0]], PCA_samples[PCA_cols[1]],c=list(m), cmap='rainbow',alpha=0.2)
    ax.set_title(graph_headings[i])
    
plt.show()




def var_dist_within_clust():
    for mi, m in enumerate(model_graphs):
        clusters = set([i for i in m if i!=-1])
        print("For {}:".format(graph_headings[mi]))
        print('Number of clusters: ', len(clusters))
        for c1 in clusters:
            # print('\nFor Cluster {}:'.format(c1))
            clust = data_samples[(m == c1)]
            varns1 =  np.var(clust)
            varns2 = varns1/np.sum(varns1)
            # varns_fin = dict(varns2)
            varns_fin = [k for k, v in sorted(dict(varns2).items(), key=lambda item: item[1])]
            print('\nFor Cluster {}:'.format(c1), '\nSimilar Attribute: ',varns_fin[:5])
            print('Varied Attribute: ',varns_fin[-5:])
        print()
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        print()


sims_inter = []
total = 2000

def print_plot2():
    plt.figure(figsize=(20,15))
    dim = 2
    for mi, m in enumerate(model_graphs):
        ax = plt.subplot(dim,dim,mi+1)
        l = []
        v = []
        c = set([i for i in m if i!=-1])
        
        for c1 in c:
            l.append(PCA_samples[(m == c1)].shape[0])

        # v=c
        ax.set(xlabel='clusters:', ylabel='count:')
        c = list([str(cl) for cl in c])
        l = []
        v = []
        ax.bar(c, l, color ='red',  width = 0.4) 
        ax.set_title(graph_headings[mi])

    plt.show()


np_samples = np.array(data_samples)
total = np_samples.shape[0]

def plot_inter_class_sim():
    fig = plt.figure(figsize = (10, 5))
    plt.title("Superior Inter Class Similarity") 
    plt.bar(graph_headings, sims_inter, color ='blue',  width = 0.4) 
    plt.show()

def plot_intra_class_sim():
    fig = plt.figure(figsize = (10,5)) 
    plt.title("Inferior Intra Class Similarity")
    plt.bar(graph_headings, sims_intra, color ='green', width = 0.4) 
    plt.show()



sim_matrix = cosine_similarity(np_samples)

tempx = []
for mi, m in enumerate(model_graphs):
    for ind in tqdm(range(total)):
        for j in range(ind):
            if m[ind]!=m[j]:
                pass
            else:
                tempx.append(sim_matrix[ind][j])
    sims_inter.append(np.mean(tempx))
    tempx = []


print("inter class similarity = ", str(sims_inter))
sims_intra = []

plot_inter_class_sim()


for mi, m in enumerate(model_graphs):
    temp = []
    clusters = set([i for i in m if i!=-1])
    for c1 in clusters:
        for c2 in clusters:
            if c1 == c2:
                pass
            else:
                temp.append(cos_sim(np.mean(data_samples[(m==c1)],axis=0),np.mean(data_samples[(m==c2)],axis=0)))
    sims_intra.append(np.mean(temp))
print("intra class similarity = ", str(sims_intra))



plot_intra_class_sim()




print_plot1()


var_dist_within_clust()

for mi, m in enumerate(model_graphs):
    c = set([i for i in m if i!=-1])

    print("For {}:".format(graph_headings[mi]))
    print('Number of clusters: ', len(c))
    for cluster1 in c:
        clust = PCA_samples[(m == cluster1)]
        print('Players that are present in {}: '.format(cluster1), end='')
        print(list(clust.Name[:5]))
    print('----------------------')



print_plot2()


clusters_dbscan = pd.read_csv('Question_4/dbscan.csv')['0']

print("outliers in DBSCAN: ")
for i in range(len(clusters_dbscan)):
    if clusters_dbscan[i]==-1:
        print(PCA_samples['Name'][i])





