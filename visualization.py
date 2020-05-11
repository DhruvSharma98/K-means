import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

df = pd.read_csv("google_review_ratings.csv")
df = df.head(1000)
df.drop(['Unnamed: 25'], 1, inplace=True)
df.fillna(0, inplace=True)

X = np.array(df.drop(['User'], 1).astype(float))
y = np.array(df['User'])

plt.figure(figsize=(10,8))
wcss = []
for i in range(1,15):
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,15), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
