import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors



with open('../columns.pickle', 'rb') as input_file:
    lab_cols,num_cols,cat_cols,pca_cols = pickle.load(input_file)

pd.options.mode.chained_assignment = None

PCA_samples = pd.read_csv('../processed_data.csv')

samples = PCA_samples[num_cols]

labels = PCA_samples[lab_cols]

def plot_graph():
    plt.figure(figsize=(25,25))
    for i in range(1,11):
        l = 5
        w = 2
        plt.subplot(l,w,i)
        nebr = i+i
        # print(nebr)
        nn = NearestNeighbors(n_neighbors = nebr)
        nbrs = nn.fit(samples)
        print('Plotting for k = ', str(nebr))
        dist, j = nbrs.kneighbors(samples)
        temp = np.sort(dist, axis=0)
        # dist = temp[:,1]
        lab = 'for k = ' + str(nebr)
        plt.ylabel(lab)
        plt.plot(temp[:,1])

    plt.show()

def main():

    db = DBSCAN(eps=10, min_samples=5)
    db_cls = db.fit_predict(samples[:100])

    

    plot_graph()
    df = PCA_samples[:100] 
    df['cluster'] = db_cls

    
    print(df[df.cluster==1])

    db = DBSCAN(eps=4, min_samples=6)
    db_cls = db.fit_predict(samples)

    db_cls = list(db_cls)
    val = -1
    print("----------------------------------")
    l = len(db_cls)

    tempfile = open('Outfile.txt', 'w')
    print(l)
    for i in range(len(db_cls)):
        if db_cls[i]!=val:
            val = -1
        else:
            pp = str(i) + "  =  " + PCA_samples['Name'][i] + "\n"
            tempfile.write(pp)
    tempfile.close()
    print("----------------------------------")


    set(list(db_cls))
    print("Exporting DBSCAN data")
    outfile = pd.DataFrame(db_cls)
    outfile.to_csv('dbscan.csv',index=False)

main()