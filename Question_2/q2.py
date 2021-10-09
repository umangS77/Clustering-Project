import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import random
import math
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score




with open('../columns.pickle', 'rb') as inpfile:
    lab_cols,num_cols,cat_cols,PCA_columns = pickle.load(inpfile)





def plot_graph_1(k, scores):
    plt.plot(k, scores) 
    plt.xlabel('k') 
    plt.ylabel('Silhoutte score') 
    plt.show()

def plot_error_graph(final_error):
    plt.plot(range(1,11), final_error)
    plt.xlabel('K')
    plt.ylabel('Total Error')
    plt.show()

PCA_samples = pd.read_csv('../processed_data.csv')


kmean = np.array(PCA_samples[num_cols])
labels = PCA_samples[lab_cols]
tempcol = num_cols+cat_cols
samples = PCA_samples[tempcol]

# kmean.shape

def extract_clusters(kmean, k):
    error_past , error_present = 1e7 , 1e6
    
    med = random.sample(range(len(kmean)), k)
    multi_meds = [kmean[i] for i in med]
    f = 0
    multi_meds = np.array(multi_meds)

    while(error_past-error_present>=1e-7):
        if f == 0:
            f = 1
        else
            multi_meds = []
            j=0;
            while j < k:
                med = np.mean(kmean[np.argwhere(j==cltrs)], axis=0)
                j = j+1
                multi_meds.append(med)
            kmeds = multi_meds
            multi_meds = np.array(kmeds)
            multi_meds = multi_meds.reshape((multi_meds.shape[0], multi_meds.shape[2]))
        error_past=error_present
        cltrs = []


        for i in range(k):
            error_present=0
            tp = np.abs(kmean-multi_meds[i])
            cl = tp*tp
            cl = np.sum(cl, axis=1)**(0.5)
            cltrs.append(cl)

        # cltrs = 
        cltrs = np.argmin(np.array(cltrs), axis = 0)
        val = abs(kmean-multi_meds[cltrs])
        error_present = val*val
        error_present = sum(np.sum(error_present, axis=1))
        error_present = math.sqrt(error_present)
    return error_present, cltrs



def main():
    clusterfile = open("Cluster_data.txt", "w")
    for k in [3, 5, 7]:
        clusterfile.write("\n")
        clusterfile.write(f"For k = {k}")
        error_present, clusters = extract_clusters(kmean, k)
        for clstr in range(k):
            clsr = []
            clusterfile.write("\n")
            clusterfile.write(f"Cluster Number:  {clstr}")
            for i in range(len(kmean)):
                if clusters[i]!=clstr:
                    pass
                else:
                    tup = [PCA_samples['Name'][i], PCA_samples['Overall'][i], PCA_samples['Position'][i]]
                    clsr.append(tup)
            
            clusterfile.write(",".join(str(item) for item in clsr[:20]))



    clusterfile.close()
    k=[]
    scores=[]
    final_clstr=[]
    final_error=[]
    for kval in range(1,11):
        error_present, clusters = extract_clusters(kmean, kval)
        final_clstr.append(clusters)
        final_error.append(error_present)


    for i in range(9):
        j = i+2
        k.append(j)
        scores.append(silhouette_score(kmean, final_clstr[i+1]))


    plot_graph_1(k, scores)
    plot_error_graph(final_error)


    csvout = pd.DataFrame(final_clstr[1])
    csvout.to_csv('k_means.csv',index=False)