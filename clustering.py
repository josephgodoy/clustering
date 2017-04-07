import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import cluster, datasets
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

np.random.seed(0)

# Sample data generators from SKLEARN. Good for testing algorithm behavior.

n_samples = 1000
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
no_structure = np.random.rand(n_samples, 2), None

# Reading in the CSV file. 

light_curves_df = pd.read_csv(input("Enter the filename for your dataset: "))
#light_curves_df = pd.read_csv('')

# Converting the pandas dataframe into np-arrays, then transposing and combining them to
# form a usable tuple for the clustering algorithms.

max_peak = light_curves_df['maxpeak'].values.tolist()
mean_noise = light_curves_df['meannoise'].values.tolist()
classification = np.array(light_curves_df['classification'].values.tolist())
array = np.array([mean_noise, max_peak])
pre_glued = array.T
glued     = (pre_glued, classification)

# Plot-related stuff.

colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
colors = np.hstack([colors] * 20)
clustering_names = [
    'K-Means', 'Affinity Propagation', 'Meanshift',
    'Spectral', 'Ward / Divisive', 'Agglomerative',
    'DBSCAN', 'BIRCH']

# Most important plot-related stuff.

plt.figure(figsize=(15.5, 15.5))
plt.subplots_adjust(left=.02, right=.98, bottom=.01, top=.94, wspace=.02, hspace=.18)
plt.suptitle(str(len(max_peak)) + " Data Points", fontweight = 'bold')
plot_num = 1

# Choose your dataset here: noisy_circles, noisy_moons, and blobs work well for testing.

datasets = [glued]

# The actual clustering occurs below:

for i_dataset, dataset in enumerate(datasets):
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    bandwidth = cluster.estimate_bandwidth(X, quantile=0.3)
    connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)
    connectivity = 0.5 * (connectivity + connectivity.T)
    ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    two_means = cluster.MiniBatchKMeans(n_clusters=2)
    ward = cluster.AgglomerativeClustering(n_clusters=2, linkage='ward',
                                           connectivity=connectivity)
    spectral = cluster.SpectralClustering(n_clusters=2,
                                          eigen_solver='arpack',
                                          affinity="nearest_neighbors")
    dbscan = cluster.DBSCAN(eps=.2)
    affinity_propagation = cluster.AffinityPropagation(damping=.9,
                                                       preference=-200)
    average_linkage = cluster.AgglomerativeClustering(
        linkage="average", affinity="cityblock", n_clusters=2,
        connectivity=connectivity)
    birch = cluster.Birch(n_clusters=2)
    clustering_algorithms = [
        two_means, affinity_propagation, ms, spectral, ward, average_linkage,
        dbscan, birch]

    for name, algorithm in zip(clustering_names, clustering_algorithms):
        t0 = time.time()
        algorithm.fit(X)
        t1 = time.time()
        if hasattr(algorithm, 'labels_'):
            y_pred = algorithm.labels_.astype(np.int)
        else:
            y_pred = algorithm.predict(X)

        # Plotting happens below:

        plt.subplot(4, len(clustering_algorithms)/4, plot_num)
        if i_dataset == 0:
            plt.title(name, size=18)
        plt.scatter(X[:, 0], X[:, 1], color=colors[y_pred].tolist(), s=10)

        if hasattr(algorithm, 'cluster_centers_'):
            centers = algorithm.cluster_centers_
            center_colors = colors[:len(centers)]
            plt.scatter(centers[:, 0], centers[:, 1], s=100, c=center_colors)
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.xticks(())
        plt.yticks(())
        plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                 transform=plt.gca().transAxes, size=15,
                 horizontalalignment='right')
        plot_num += 1

# Save figure, show plot.

plt.savefig("latest_save.png")
plt.show()
