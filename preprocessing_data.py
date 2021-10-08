import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import seaborn as sns
import pickle

df = pd.read_csv('football_data.csv', encoding='ISO-8859-1')
df = df.drop(['ï»¿', 'ID', 'Photo', 'Flag', 'Club', 'Club Logo', 
       'Special', 'Weak Foot', 'Body Type', 'Real Face',
       'Jersey Number', 'Joined', 'Loaned From', 'Contract Valid Until',
       'LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW',
       'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM',
       'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB', 'Release Clause'], axis=1)

df = df.dropna()
df = df.reset_index()

val = []
val2 = []
wra = []
wrd = []
wt = []
ht=[]

Foot = [1 for i in range(len(df['Name']))]

for i in range(len(df['Name'])):
    if df['Preferred Foot'][i] == 'Right':
        Foot[i]=0

df['Foot'] = Foot

Pos = [-1 for i in range(len(df['Name']))]

for i in range(len(df['Name'])):
    if df['Position'][i] == 'LS' or df['Position'][i] == 'RS' or df['Position'][i] == 'ST' or df['Position'][i] == 'CF':
        Pos[i]=7
for i in range(len(df['Name'])):
    if df['Position'][i] == 'LW' or df['Position'][i] == 'RW' or df['Position'][i] == 'LF' or df['Position'][i] == 'RF' or df['Position'][i] == 'LM' or df['Position'][i] == 'RM':
        Pos[i]=6
for i in range(len(df['Name'])):
    if df['Position'][i] == 'RAM' or df['Position'][i] == 'LAM' or df['Position'][i] == 'CAM':
        Pos[i]=5
for i in range(len(df['Name'])):
    if df['Position'][i] == 'CM' or df['Position'][i] == 'LCM' or df['Position'][i] == 'RCM':
        Pos[i]=4
for i in range(len(df['Name'])):
    if df['Position'][i] == 'RDM' or df['Position'][i] == 'LDM' or df['Position'][i] == 'CDM':            
        Pos[i]=3
for i in range(len(df['Name'])):        
    if df['Position'][i]=='LWB' or df['Position'][i]=='LB' or df['Position'][i]=='RB' or df['Position'][i]=='RWB':
        Pos[i]=2
for i in range(len(df['Name'])):
    if df['Position'][i]=='RCB' or df['Position'][i]=='LCB' or df['Position'][i]=='CB':
        Pos[i]=1
for i in range(len(Pos)):
    if Pos[i]==-1:
        Pos[i]=0


df['Position']=Pos



for v in df['Value']:
    v = v.lstrip('â\x82¬')
    if v[-1] != 'K':
        v=float(v.rstrip('M'))*1000000
    else:
        v=float(v.rstrip('K'))*1000
    val.append(v)

df['Value'] = val
        

for w in df['Wage']:
    w = w.lstrip('â\x82¬').rstrip('K')
    w=float(w)*1000
    val2.append(w)

df['Wage'] = val

for r in df['Work Rate']:
    sp = r.split('/ ')
    tp = 0
    if sp[0]=='High':
        tp = 2
        # wra.append(2)
    elif sp[0]=='Medium':
        tp = 1
        # wra.append(1)
    wra.append(tp)

df['Att Work Rate']=wra

for r in df['Work Rate']:
    sp = r.split('/ ')
    tp = 0
    if sp[1]=='High':
        tp = 2
        # wrd.append(2)
    elif sp[1]=='Medium':
        tp = 1
        # wrd.append(1)
    wrd.append(tp)


df['Def Work Rate']=wrd       

for w in df['Weight']:
    w = float(w.rstrip('lbs'))
    wt.append(w)

for h in df['Height']:
    h = int(h[:1])*12+int(h[2:])
    ht.append(h)

df['Weight']=wt
df['Height']=ht

df = df.drop(['Preferred Foot', 'Work Rate'], axis=1)    
# df.head()
print(df.head())

label_columns = ['index','Name','Nationality']
categorical_columns = ['Position','Foot','Att Work Rate','Def Work Rate']
labels = df[label_columns]
categorical = df[categorical_columns]
numerical = df.drop(label_columns,axis=1)
numerical = numerical.drop(categorical_columns,axis=1)
numerical_columns = list(numerical.columns)


scaler = StandardScaler()
# scaler = MinMaxScaler()
numerical = pd.DataFrame(scaler.fit_transform(numerical))
numerical.columns = numerical_columns
# print(numerical.head())


samples = pd.concat([numerical, categorical], axis=1, join='inner')
labeled_samples = pd.concat([labels, samples], axis=1, join='inner')

pca_decomposer = PCA(n_components=2)
PCs_2d = pd.DataFrame(pca_decomposer.fit_transform(samples))
PCA_columns = ["PC1",'PC2']
PCs_2d.columns = PCA_columns
labeled_samples_with_pca = pd.concat([labeled_samples,PCs_2d], axis=1, join='inner')
labeled_samples_with_pca.head()

# labeled_samples_with_pca.describe()


labeled_samples_with_pca.to_csv('processed_data.csv')
with open('columns.pickle', 'wb') as f:
    pickle.dump((label_columns,numerical_columns,categorical_columns,PCA_columns), f)








