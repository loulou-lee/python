import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')
from sklearn.datasets import load_iris
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram

iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns = iris.feature_names)
print(iris_df.head(3))
print()
dist_vec = pdist(iris_df.loc[0:4, ['sepal length (cm)', 'sepal width (cm)']], metric='euclidean')
print(dist_vec)
row_dist = pd.DataFrame(squareform(dist_vec))
print(row_dist)

from scipy.cluster.hierarchy import linkage
row_clusters = linkage(dist_vec, method='complete')

df = pd.DataFrame(row_clusters, columns=['군집id1', '군집id2', '거리', '멤버수'])
print(df)

# dendrogram으로 row_clusters를 시각화
from scipy.cluster.hierarchy import dendrogram
low_dend = dendrogram(row_clusters)
plt.show()