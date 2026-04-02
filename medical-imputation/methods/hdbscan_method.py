import numpy as np
from sklearn.preprocessing import StandardScaler
from .common import fill_by_cluster_mean

try:
    import hdbscan
    HAS_HDBSCAN = True
except Exception:
    HAS_HDBSCAN = False

def impute_hdbscan(img, min_cluster_size=10, min_samples=5):
    out = img.copy()
    mask = ~np.isnan(out)
    data = np.column_stack(np.where(mask) + (out[mask],))
    if not HAS_HDBSCAN:
        out[np.isnan(out)] = np.nanmean(out)
        return out
    labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)\
                    .fit_predict(StandardScaler().fit_transform(data))
    return fill_by_cluster_mean(out, data, labels)