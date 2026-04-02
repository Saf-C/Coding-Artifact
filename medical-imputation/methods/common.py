import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def fill_by_cluster_mean(img, data, labels):
    valid = labels != -1
    out = img.copy()
    if np.sum(valid) < 10:
        out[np.isnan(out)] = np.nanmean(out)
        return out

    miss = np.isnan(out)
    X_miss = np.column_stack(np.where(miss))
    if len(X_miss) == 0:
        return out

    clf = KNeighborsClassifier(n_neighbors=1).fit(data[valid, :2], labels[valid])
    preds = clf.predict(X_miss)
    means = {l: np.mean(data[labels == l, 2]) for l in np.unique(labels) if l != -1}
    default_mean = np.nanmean(out)
    out[miss] = [means.get(p, default_mean) for p in preds]
    return out