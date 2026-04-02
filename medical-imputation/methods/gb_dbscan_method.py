import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from .common import fill_by_cluster_mean

def impute_gb_dbscan(img, ratio=0.92):
    out = img.copy()
    mask = ~np.isnan(out)
    data = np.column_stack(np.where(mask) + (out[mask],)).astype(float)
    n = data.shape[0]

    if n < 20:
        out[np.isnan(out)] = np.nanmean(out)
        return out

    scaled = StandardScaler().fit_transform(data)
    K = max(3, int(np.ceil(np.sqrt(n) * 0.3)))

    nn = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(scaled)
    nbrs = nn.kneighbors(scaled, return_distance=False)

    used = -1 * np.ones(n, dtype=int)
    gb_list = []
    for i in range(n):
        if used[i] == -1:
            gb = nbrs[i]
            gb_list.append(gb)
            used[gb] = 1

    radii, centers = [], []
    for gb in gb_list:
        g = scaled[gb]
        c = g.mean(axis=0)
        r = np.max(np.sqrt(np.sum((g - c) ** 2, axis=1)))
        radii.append(r)
        centers.append(c)

    radii = np.array(radii)
    centers = np.array(centers)
    gb_num = len(gb_list)

    th = np.sort(radii)[max(0, min(gb_num - 1, int(gb_num * ratio) - 1))]
    CBS = np.where(radii <= th)[0]
    NCBS = np.where(radii > th)[0]

    if len(CBS) == 0:
        out[np.isnan(out)] = np.nanmean(out)
        return out

    cbs_centers = centers[CBS]
    cbs_r = radii[CBS]

    cbs_n = len(CBS)
    unvisited = list(range(cbs_n))
    cluster = -1 * np.ones(cbs_n, dtype=int)
    cid = -1

    while unvisited:
        p = unvisited.pop(0)
        neighbors = []
        for i in range(cbs_n):
            if i != p:
                if np.linalg.norm(cbs_centers[i] - cbs_centers[p]) <= (cbs_r[i] + cbs_r[p]):
                    neighbors.append(i)
        cid += 1
        cluster[p] = cid
        j = 0
        while j < len(neighbors):
            pi = neighbors[j]; j += 1
            if pi in unvisited:
                unvisited.remove(pi)
                for t in range(cbs_n):
                    if t != pi and np.linalg.norm(cbs_centers[t] - cbs_centers[pi]) <= (cbs_r[t] + cbs_r[pi]):
                        if t not in neighbors:
                            neighbors.append(t)
            if cluster[pi] == -1:
                cluster[pi] = cid

    point_cluster = -1 * np.ones(n, dtype=int)
    for i_cb, gb_idx in enumerate(CBS):
        for p in gb_list[gb_idx]:
            point_cluster[p] = cluster[i_cb]

    for gb_idx in NCBS:
        pts = gb_list[gb_idx]
        labeled = pts[point_cluster[pts] != -1]
        unlabeled = pts[point_cluster[pts] == -1]
        if len(unlabeled) == 0:
            continue
        if len(labeled) > 0:
            d = cdist(scaled[unlabeled], scaled[labeled])
            near = np.argmin(d, axis=1)
            point_cluster[unlabeled] = point_cluster[labeled[near]]
        else:
            center_ncb = scaled[pts].mean(axis=0, keepdims=True)
            near_cb = int(np.argmin(cdist(center_ncb, cbs_centers)))
            point_cluster[pts] = cluster[near_cb]

    return fill_by_cluster_mean(out, data, point_cluster)