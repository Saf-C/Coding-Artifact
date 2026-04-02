import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from .common import fill_by_cluster_mean

def impute_spectral(img, k=5):
    out = img.copy()
    mask = ~np.isnan(out)
    data = np.column_stack(np.where(mask) + (out[mask],))

    idx = np.random.choice(len(data), 5000, replace=False) if len(data) > 5000 else np.arange(len(data))
    sub = data[idx]
    sub_scaled = StandardScaler().fit_transform(sub)

    spec = SpectralClustering(n_clusters=k, affinity="nearest_neighbors", n_neighbors=20, random_state=42)
    sub_labels = spec.fit_predict(sub_scaled)

    prop = KNeighborsClassifier(n_neighbors=1).fit(sub, sub_labels)
    labels = prop.predict(data)
    return fill_by_cluster_mean(out, data, labels)