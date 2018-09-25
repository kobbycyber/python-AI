import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from sklearn.cluster import KMeans  # pip3 install KMeans
from sklearn import metrics         # pip3 install metrics

# genrate 4-clusters
from sklearn.datasets.samples_generator import make_blobs
X, y_true = \
    make_blobs(n_samples = 500, centers = 4, cluster_std = 0.40, random_state = 0)
# Initialized the score.
scores = []
values = np.arange(2, 10)

for num_clusters in values:
    kmeans = KMeans(init = 'k-means++', n_clusters = num_clusters, n_init = 10)
    kmeans.fit(X)

score = metrics.silhouette_score(X, kmeans.labels_, \
    metric = 'euclidean', sample_size = len(X))

# print number of clusters
print("\nNumber of clusters =", num_clusters)
print("Silhouette scores =", score)
scores.append(score)
print ('scores: ', scores)

# Get optimal number of cluster
num_clusters = np.argmax(scores) + values[0]
print('\nnp.argmax(scores) =', np.argmax(scores))
print('\nvalues[0] =', values[0])
print('\nOptimal number of clusters =', num_clusters)