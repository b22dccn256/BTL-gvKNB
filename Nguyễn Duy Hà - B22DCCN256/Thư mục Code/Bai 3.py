#256
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import requests
import sys
import pandas as pd
import time
import csv
import numpy as np
import matplotlib.pyplot as ppl

df = pd.read_csv('results.csv')

#Get_inf_cols
columns = df.columns[5:8]
df[columns] = df[columns].replace('N/A', pd.NA)
data = df[columns].apply(pd.to_numeric, errors='coerce').dropna()

#Mã hóa dữ liệu
datasc = StandardScaler()
data_scaled = datasc.fit_transform(data)

#K-means
n_clusters = 5  
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(data_scaled)

#Tag cho mỗi cầu thủ
df['Cluster'] = kmeans.labels_

#Hiển thị một số thông tin về các cầu thủ trong từng nhóm
for i in range(n_clusters):  
    print(f"\nCluster {i}:")
    print(df[df['Cluster'] == i][['Player', 'Team', 'Cluster']].head())

# Vẽ biểu đồ phân cụm
ppl.figure(figsize=(10, 6))
x_index = 0  # Cột đầu tiên trong data_scaled
y_index = 1  # Cột thứ hai trong data_scaled
ppl.scatter(data_scaled[:, x_index], data_scaled[:, y_index], c=df['Cluster'], cmap='viridis', marker='o')
ppl.title('K-means Clustering of Players')
ppl.xlabel(f'Feature {x_index + 1}')
ppl.ylabel(f'Feature {y_index + 1}')
ppl.colorbar(label='Cluster')
ppl.grid()
ppl.show()