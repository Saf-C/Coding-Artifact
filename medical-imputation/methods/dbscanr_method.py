import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from dbscanr import DBSCANR
from .common import fill_by_cluster_mean

def impute_dbscanr(img, k=5, K=np.inf, p=2, expand=True, enforceK=False, max_points=6000):
    out = img.copy()
    mask = ~np.isnan(out)
    data = np.column_stack(np.where(mask) + (out[mask],))

    if data.shape[0] < 20:
        out[np.isnan(out)] = np.nanmean(out)
        return out

    scaled = StandardScaler().fit_transform(data)
    n = scaled.shape[0]

    # DBSCANR is O(N^2) memory; run on subset then propagate labels if too large
    if n > max_points:
        rng = np.random.default_rng(42)
        idx_sub = rng.choice(n, size=max_points, replace=False)
        sub_scaled = scaled[idx_sub]

        sub_labels = DBSCANR(k=k, K=K, p=p, expand=expand, enforceK=enforceK).fit_predict(sub_scaled).astype(int)
        sub_labels[sub_labels == 0] = -1

        valid_sub = sub_labels != -1
        if np.sum(valid_sub) < 10:
            out[np.isnan(out)] = np.nanmean(out)
            return out

        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(data[idx_sub][valid_sub, :2], sub_labels[valid_sub])
        labels = knn.predict(data[:, :2])
    else:
        labels = DBSCANR(k=k, K=K, p=p, expand=expand, enforceK=enforceK).fit_predict(scaled).astype(int)
        labels[labels == 0] = -1

    return fill_by_cluster_mean(out, data, labels)