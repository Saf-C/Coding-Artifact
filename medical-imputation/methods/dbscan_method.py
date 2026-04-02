import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from .common import fill_by_cluster_mean

def impute_dbscan(img, eps=0.25, min_samples=5):
    out = img.copy()
    mask = ~np.isnan(out)
    data = np.column_stack(np.where(mask) + (out[mask],))
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(StandardScaler().fit_transform(data))
    return fill_by_cluster_mean(out, data, labels)