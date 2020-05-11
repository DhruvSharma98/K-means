import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import pickle
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from scipy.spatial import Voronoi, voronoi_plot_2d


df = pd.read_csv("google_review_ratings.csv")
df.drop(['Unnamed: 25'], 1, inplace=True)
df.fillna(0, inplace=True)

#print(df.head(100))
#print(df.describe())
#print(df.isnull().sum())
#print(df.dtypes)

def convert():
    for i in range(1,len(df['User'])+1):
        yield i

new_data = convert()

df['User'] = df['User'].map(lambda x:new_data.__next__())
print(df.head(100))



#X = np.array(df.drop(['User'], 1).astype(float))
X = np.array(df[['Category 1','Category 2']])
y = np.array(df['User'])

##plt.scatter(X[:,0],X[:,1],c='red',s=5)
##plt.show()


km = KMeans(n_clusters=5, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)

y_km = km.fit_predict(X)

with open('clustering.pickle', 'wb') as f:
    pickle.dump(km, f)

f.close()

##km = open('clustering.pickle','rb')
    
cluster_col = np.array(km.labels_)
df['Cluster'] = cluster_col
print(df.head(100))

    
plt.scatter(
    X[y_km == 0, 0], X[y_km == 0, 1],
    s=50, c='lightgreen',
    marker='s', edgecolor='black',
    label='cluster 1'
)

plt.scatter(
    X[y_km == 1, 0], X[y_km == 1, 1],
    s=50, c='orange',
    marker='o', edgecolor='black',
    label='cluster 2'
)

plt.scatter(
    X[y_km == 2, 0], X[y_km == 2, 1],
    s=50, c='lightblue',
    marker='v', edgecolor='black',
    label='cluster 3'
)

plt.scatter(
    X[y_km == 3, 0], X[y_km == 3, 1],
    s=50, c='yellow',
    marker='^', edgecolor='black',
    label='cluster 4'
)

plt.scatter(
    X[y_km == 4, 0], X[y_km == 4, 1],
    s=50, c='grey',
    marker='d', edgecolor='black',
    label='cluster 5'
)

# plot the centroids
plt.scatter(
    km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
    s=250, marker='*',
    c='red', edgecolor='black',
    label='centroids'
)

vor = Voronoi(km.cluster_centers_)
voronoi_plot_2d(vor)

plt.grid()
plt.show()
