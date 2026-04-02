import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from .common import fill_by_cluster_mean

def impute_kmeans(img, k=5):
    out = img.copy()
    mask = ~np.isnan(out)
    data = np.column_stack(np.where(mask) + (out[mask],))
    labels = KMeans(n_clusters=k, random_state=42, n_init=5).fit_predict(StandardScaler().fit_transform(data))
    return fill_by_cluster_mean(out, data, labels)