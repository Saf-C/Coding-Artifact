import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

from ar_dbscan.model.model import DrlDbscan
from ar_dbscan.utils.utils import generate_parameter_space


def _safe_mean_fill(img):
    out = img.copy()
    m = np.isnan(out)
    if np.all(m):
        out[m] = 0.0
    else:
        out[m] = np.nanmean(out)
    return out


def impute_ar_dbscan(
    img,
    train_size=0.2,
    episode_num=2,   # fast mode for now
    layer_num=1,     # fast mode for now
    eps_size=5,
    min_size=4,
    batch_size=8,
    step_num=6,
    device="cpu",
):
    """
    AR-DBSCAN core integration (without structural-entropy pre_partition).
    Uses DrlDbscan RL parameter search, then fills missing pixels by cluster means.
    """
    out = img.copy()
    miss_mask = np.isnan(out)
    obs_mask = ~miss_mask

    # if too sparse, fallback
    if obs_mask.sum() < 30:
        return _safe_mean_fill(out)

    # Features for observed pixels: row, col, intensity
    rr, cc = np.where(obs_mask)
    vv = out[obs_mask]
    data = np.column_stack([rr, cc, vv]).astype(np.float64)

    # Normalize for DBSCAN/AR stability
    features = MinMaxScaler().fit_transform(data)
    MAX_AR_POINTS = 12000
    if features.shape[0] > MAX_AR_POINTS:
        rng_ds = np.random.default_rng(42)
        keep = rng_ds.choice(features.shape[0], size=MAX_AR_POINTS, replace=False)
        features = features[keep]
        rr = rr[keep]
        cc = cc[keep]
        vv = vv[keep]
    n = features.shape[0]

    # Pseudo-labels just to satisfy AR reward interface
    bins = np.quantile(features[:, 2], [0.25, 0.5, 0.75])
    labels = np.digitize(features[:, 2], bins).astype(int)

    rng = np.random.default_rng(42)
    reward_mask = rng.choice(n, size=max(2, int(train_size * n)), replace=False)

    # Generate AR parameter space
    param_size, param_step, param_center, param_bound = generate_parameter_space(
        features, layer_num, eps_size, min_size, "Stream-Image"
    )

    agents = []
    for l in range(layer_num):
        agent = DrlDbscan(
            param_size=param_size,
            param_step=param_step[l],
            param_center=param_center,
            param_bound=param_bound,
            device=device,
            batch_size=batch_size,
            step_num=step_num,
            dim=features.shape[1],
        )
        agents.append(agent)

    label_dic = {}
    max_max_reward = [0, list(param_center), 0]
    cur_labels = None

    for l in range(layer_num):
        agent = agents[l]
        agent.reset(max_max_reward)
        max_nmi = -1
        best_cluster_log = []

        for ep in range(1, episode_num + 1):
            agent.reset0()
            cur_labels, _, _, _, max_reward, max_nmi, best_cluster_log = agent.train(
                episode_i=ep,
                extract_masks=reward_mask,
                extract_features=features,
                extract_labels=labels,
                label_dic=label_dic,
                reward_factor=0.2,
                max_nmi=max_nmi,
                best_cluster_log=best_cluster_log,
                log_flag=False,
            )
            if max_reward[0] > max_max_reward[0]:
                max_max_reward = list(max_reward)

        # optional detect pass
        agent.reset0()
        cur_labels, _, _ = agent.detect(features, label_dic)

    # Use best discovered parameter labels if available
    key = f"{max_max_reward[1][0]}+{max_max_reward[1][1]}"
    if key in label_dic:
        final_labels = np.asarray(label_dic[key])
    elif cur_labels is not None:
        final_labels = np.asarray(cur_labels)
    else:
        return _safe_mean_fill(out)

    valid = final_labels != -1
    if valid.sum() < 10:
        return _safe_mean_fill(out)

    # Map missing coords to nearest observed cluster
    miss_r, miss_c = np.where(miss_mask)
    miss_xy = np.column_stack([miss_r, miss_c])
    obs_xy = np.column_stack([rr, cc])

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(obs_xy[valid], final_labels[valid])
    pred_clusters = knn.predict(miss_xy)

    cluster_means = {
        c: np.mean(vv[final_labels == c])
        for c in np.unique(final_labels)
        if c != -1 and np.any(final_labels == c)
    }
    global_mean = np.nanmean(out) if not np.isnan(out).all() else 0.0

    filled_vals = np.array([cluster_means.get(c, global_mean) for c in pred_clusters], dtype=float)
    out[miss_mask] = filled_vals
    return out