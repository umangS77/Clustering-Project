import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import seaborn as sns
import pickle


labeled_samples_with_pca = pd.read_csv('../processed_data.csv')
with open('../columns.pickle', 'rb') as f:
    label_columns,numerical_columns,categorical_columns,PCA_columns = pickle.load(f)


labels = labeled_samples_with_pca[label_columns]
samples = labeled_samples_with_pca[numerical_columns+categorical_columns]

fig, ax = plt.subplots()
labeled_samples_with_pca['Nationality'].value_counts().head(10).plot(figsize=(30,10),ax=ax, kind='bar')


fig, ax = plt.subplots()
labeled_samples_with_pca['Position'].value_counts().head(20).plot(figsize=(30,10),ax=ax, kind='bar')

fig, ax = plt.subplots()
labeled_samples_with_pca['Foot'].value_counts().head(20).plot(figsize=(30,10),ax=ax, kind='bar')

countries = set(labeled_samples_with_pca["Nationality"])
avg_rat = {}
for i in countries:
    if labeled_samples_with_pca[labeled_samples_with_pca["Nationality"]==i]["Skill Moves"].count() > 100:
        avg_rat[i]= np.mean(labeled_samples_with_pca[labeled_samples_with_pca["Nationality"]==i]["Skill Moves"])
avg_rat = {k: v for k, v in sorted(avg_rat.items(), key=lambda item: item[1], reverse=True)}
df1 = pd.DataFrame({'countries':list(avg_rat.keys())[:10], 'Skill Moves':list(avg_rat.values())[:10]})
df1.plot.bar(x='countries', y='Skill Moves')

countries = set(labeled_samples_with_pca["Nationality"])
avg_rat = {}
for i in countries:
    if labeled_samples_with_pca[labeled_samples_with_pca["Nationality"]==i]["LongPassing"].count() > 100:
        avg_rat[i]= np.mean(labeled_samples_with_pca[labeled_samples_with_pca["Nationality"]==i]["LongPassing"])
avg_rat = {k: v for k, v in sorted(avg_rat.items(), key=lambda item: item[1], reverse=True)}
df1 = pd.DataFrame({'countries':list(avg_rat.keys())[:10], 'LongPassing':list(avg_rat.values())[:10]})
df1.plot.bar(x='countries', y='LongPassing')


countries = set(labeled_samples_with_pca["Nationality"])
avg_rat = {}
for i in countries:
    if labeled_samples_with_pca[labeled_samples_with_pca["Nationality"]==i]["SlidingTackle"].count() > 100:
        avg_rat[i]= np.mean(labeled_samples_with_pca[labeled_samples_with_pca["Nationality"]==i]["SlidingTackle"])
avg_rat = {k: v for k, v in sorted(avg_rat.items(), key=lambda item: item[1], reverse=True)}
df1 = pd.DataFrame({'countries':list(avg_rat.keys())[:10], 'SlidingTackle':list(avg_rat.values())[:10]})
df1.plot.bar(x='countries', y='SlidingTackle')



fig, ax = plt.subplots(figsize=(10,10))
x = labeled_samples_with_pca[PCA_columns[0]]
y = labeled_samples_with_pca[PCA_columns[1]]
ax.scatter(x, y, alpha=0.1)
plt.title('Scatter plot of PCAs')
plt.xlabel('PCA_1')
plt.ylabel('PCA_2')
plt.show()


# Heatmap of the attributes to understand correlation
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(samples.corr(), square = True, linewidths=0.5,ax=ax,cmap="coolwarm")
t = fig.suptitle('Player Attributes Correlation Heatmap', fontsize=14)
plt.show()


labeled_samples_with_pca.plot.hexbin(x='GKDiving', y='Position',gridsize=20, sharex=False)
labeled_samples_with_pca.plot.hexbin(x='SprintSpeed', y='Position',gridsize=20, sharex=False)
labeled_samples_with_pca.plot.hexbin(x='Strength', y='Position',gridsize=20, sharex=False)

labeled_samples_with_pca.plot.hexbin(x='Overall', y='Potential',gridsize=20, sharex=False)

plt.show()

cols = ['Overall', 'Value', 'Height', 'Dribbling','GKHandling']
pp = sns.pairplot(samples[cols][:1000], height=1.8, aspect=1.8,
                  plot_kws=dict(edgecolor="k", linewidth=0.5),
                  diag_kind="kde", diag_kws=dict(shade=True))

fig = pp.fig 
fig.subplots_adjust(top=0.93, wspace=0.3)
t = fig.suptitle('Player Attributes Pairwise Plots', fontsize=14)

plt.show()


plt.figure(figsize=(20,12))
for i,col in enumerate(cols):
    plt.subplot(3,2,i+1)
    labeled_samples_with_pca[col][:1000].plot.box()

plt.show()



parallel = labeled_samples_with_pca[cols+['Att Work Rate']][:1000]
pd.plotting.parallel_coordinates(parallel, 'Att Work Rate', color=('#FFE888', '#FF9999', '#0000FF'))

plt.show()

parallel = labeled_samples_with_pca[cols+['Def Work Rate']][:1000]
pd.plotting.parallel_coordinates(parallel, 'Def Work Rate', color=('#FFE888', '#FF9999', '#0000FF'))


plt.show()















