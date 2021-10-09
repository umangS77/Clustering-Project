import numpy as np
import pandas as pd
import pickle
import random
import seaborn as sns
import math
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler


with open('../columns.pickle', 'rb') as inpfile:
    lab_cols,num_cols,cat_cols,pcaCols = pickle.load(inpfile)

def check_list_error(lst):
    print(lst)

PCA_samples = pd.read_csv('../processed_data.csv')

temp_samp = PCA_samples[]


def plot_graph_1():
    fig, ax = plt.subplots(figsize=(12,12))
    print()
    sns.heatmap(temp_samp.corr(), square = True, linewidths=0.5,ax=ax,cmap="coolwarm")
    print("plotting Player Attributes Heatmap")
    zz = fig.suptitle('Player Attributes Heatmap', fontsize=12)
    plt.show()

labels = PCA_samples[lab_cols]
tempcol = num_cols+cat_cols

def plot_graph_2():
    PCA_samples.plot.hexbin(x='GKDiving', y='Position',gridsize=20, sharex=False)
    print("plotting GKDiving vs Position")
    PCA_samples.plot.hexbin(x='SprintSpeed', y='Position',gridsize=20, sharex=False)
    print("plotting SprintSpeed vs Position")
    PCA_samples.plot.hexbin(x='Strength', y='Position',gridsize=20, sharex=False)
    print("plotting Strength vs Position")
    PCA_samples.plot.hexbin(x='Overall', y='Potential',gridsize=20, sharex=False)
    print("plotting Overall vs Position")
    plt.show()

def plot_graph_3(headings):
    pd.plotting.parallel_coordinates(PCA_samples[headings+['Att Work Rate']][:1000], 'Att Work Rate', color=('#FFE888', '#FF9999', '#0000FF'))
    plt.show()

def plot_graph_4(headings):
    pd.plotting.parallel_coordinates(PCA_samples[headings+['Def Work Rate']][:1000], 'Def Work Rate', color=('#FFE888', '#FF9999', '#0000FF'))
    plt.show()

def main():
    print("Sampling:")
    fig, ax = plt.subplots()
    PCA_samples['Nationality'].value_counts().head(10).plot(figsize=(30,10),ax=ax, kind='bar')
    print("1: Nationality")
    rat_mean = {}

    fig, ax = plt.subplots()
    print("2: Position")
    PCA_samples['Position'].value_counts().head(20).plot(figsize=(30,10),ax=ax, kind='bar')

    nations = set(PCA_samples["Nationality"])
    
    fig, ax = plt.subplots()
    print("3: Foot")
    PCA_samples['Foot'].value_counts().head(20).plot(figsize=(30,10),ax=ax, kind='bar')

    print()
    for i in nations:
        if PCA_samples[PCA_samples["Nationality"]==i]["Skill Moves"].count() <= 100:
            pass
        else:
            rat_mean[i]= np.mean(PCA_samples[PCA_samples["Nationality"]==i]["Skill Moves"])

    nations = set(PCA_samples["Nationality"])
    rat_mean = {k: val for k, val in sorted(rat_mean.items(), key=lambda item: item[1], reverse=True)}
    print("Skill Moves")
    df_A = pd.DataFrame({'nations':list(rat_mean.keys())[:10], 'Skill Moves':list(rat_mean.values())[:10]})
    rat_mean = {}
    df_A.plot.bar(x='nations', y='Skill Moves')

    
    for nat in nations:
        if PCA_samples[nat==PCA_samples["Nationality"]]["LongPassing"].count() <= 100:
            pass
        else:
            rat_mean[nat]= np.mean(PCA_samples[PCA_samples["Nationality"]==nat]["LongPassing"])

    rat_mean = {k: val for k, val in sorted(rat_mean.items(), key=lambda item: item[1], reverse=True)}
    print("Long Passing") 
    df_A = pd.DataFrame({'nations':list(rat_mean.keys())[:10], 'LongPassing':list(rat_mean.values())[:10]})
    nations = set(PCA_samples["Nationality"])
    
    df_A.plot.bar(x='nations', y='LongPassing')
    
    rat_mean = {}
    for nat in nations:
        tp = PCA_samples[PCA_samples["Nationality"]==nat]["SlidingTackle"].count()
        if tp <= 100:
            pass
        else:
            rat_mean[nat]= np.mean(PCA_samples[PCA_samples["Nationality"]==nat]["SlidingTackle"])
    
    nations = set(PCA_samples["Nationality"])
    print("Sliding Tackle")
    rat_mean = {k: val for k, val in sorted(rat_mean.items(), key=lambda item: item[1], reverse=True)}
    print()
    print("***************")
    df_A = pd.DataFrame({'nations':list(rat_mean.keys())[:10], 'SlidingTackle':list(rat_mean.values())[:10]})
    print("Scatter plot PCA_1 vs PCA_2")
    print("***************")
    df_A.plot.bar(x='nations', y='SlidingTackle')

    rat_mean = {}
    fig, ax = plt.subplots(figsize=(10,10))
    plt.xlabel('PCA_1')
    plt.ylabel('PCA_2')
    ax.scatter(PCA_samples[pcaCols[0]], PCA_samples[pcaCols[1]], alpha=0.1)
    plt.title('Scatter plot of PCAs')
    
    plt.show()

    plot_graph_1()

    plot_graph_2()

    headings = ['Overall', 'Value', 'Height', 'Dribbling','GKHandling']

    w = 2
    fig = (sns.pairplot(temp_samp[headings][:1000], height=1.8, aspect=1.8, plot_kws=dict(edgecolor="k", linewidth=0.5), diag_kind="kde", diag_kws=dict(shade=True))).fig 
    fig.subplots_adjust(top=0.95, wspace=0.5)

    l = 3
    zz = fig.suptitle('Player Attributes Pairwise Plots', fontsize=15)
    print("plotting Player Attributes")
    plt.show()
    plt.figure(figsize=(24,14))
    j=0
    for i,col in enumerate(headings):
        j=i+1
        plt.subplot(l,w,j)
        PCA_samples[col][:1000].plot.box()

    plt.show()

    plot_graph_3(headings)

    plot_graph_4(headings)

main()