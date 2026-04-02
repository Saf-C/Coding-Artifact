import numpy as np

class DBSCANR:
    def __init__(self, k, K=np.inf, p=2, expand=True, enforceK=False):
        self.k = int(k)
        self.K = K
        self.p = p
        self.expand = bool(expand)
        self.enforceK = bool(enforceK)

    def fit_predict(self, data):
        data = np.asarray(data, dtype=float)
        M = data.shape[0]
        U = np.zeros(M, dtype=int)
        C = 1

        Q, Q_flag, p_dist = self._rnn(data, self.k, self.p)

        if not Q_flag:
            core_pts_mat, _, _ = self._get_core_pts_mat(Q, M, self.k, order=(True, -2))
            ranked = core_pts_mat[:, 0].astype(int) if core_pts_mat.size else np.array([], dtype=int)

            if ranked.size >= 2:
                while ranked.size > 0:
                    q_idx = int(ranked[0])
                    ranked = self._setdiff_stable(ranked, np.array([q_idx], dtype=int))
                    neighbours, _ = self._search_neighbourhood(q_idx, Q, core_pts_mat[:, 0].astype(int))
                    U, S = self._expand_cluster(q_idx, neighbours, C, U, Q, core_pts_mat[:, 0].astype(int))
                    if S >= self.k:
                        C += 1
                    if C > self.K and self.enforceK:
                        ranked = np.array([], dtype=int)
                    ranked = self._setdiff_stable(ranked, np.where(U > 0)[0].astype(int))

            if self.expand:
                U = self._get_clustering(
                    p_dist, U, M,
                    core_pts_mat[:, 0].astype(int) if core_pts_mat.size else np.array([], dtype=int)
                )
        return U

    @staticmethod
    def _setdiff_stable(a, b):
        bset = set(np.asarray(b, dtype=int).tolist())
        return np.array([x for x in np.asarray(a, dtype=int).tolist() if x not in bset], dtype=int)

    def _mink_dist(self, data, k, p, M):
        dist = np.zeros((M, M), dtype=float)
        kdist = np.full(M, np.nan, dtype=float)
        for i in range(M):
            diff = np.abs(data[i, :] - data) ** p
            dist[i, :] = np.sum(diff, axis=1)
            tmp = np.sort(dist[i, :])
            kdist[i] = tmp[k]
        return dist, kdist

    def _rnn(self, data, k, p):
        M = data.shape[0]
        if k >= M:
            R = np.zeros((M, M), dtype=bool)
            p_dist = np.zeros((M, M), dtype=float)
            return R, True, p_dist

        p_dist, kdist = self._mink_dist(data, k, p, M)
        kdist_mat = np.repeat(kdist.reshape(M, 1), M, axis=1)
        R = p_dist <= kdist_mat
        np.fill_diagonal(R, False)
        return R, False, p_dist

    def _get_core_pts_mat(self, Q, M, k, order=(True, -2)):
        pts_idx = np.arange(M, dtype=int)
        rnn_mat = np.sum(Q, axis=0)
        core_pts_idx = pts_idx[rnn_mat >= k]
        nRNN = rnn_mat[core_pts_idx]
        if core_pts_idx.size == 0:
            return np.empty((0, 2), dtype=float), core_pts_idx, nRNN
        mat = np.column_stack([core_pts_idx, nRNN])
        if order[0] and order[1] == -2:
            idx = np.argsort(-mat[:, 1], kind="mergesort")
            mat = mat[idx, :]
        return mat, core_pts_idx, nRNN

    def _search_neighbourhood(self, q_idx, Q, core_pts):
        neighbours = np.where(Q[:, q_idx])[0].astype(int)
        neighbours = np.array([c for c in core_pts if c in set(neighbours.tolist())], dtype=int)
        nearest_neighbours = np.where(Q[q_idx, :])[0].astype(int)
        return neighbours, nearest_neighbours

    def _expand_cluster(self, q_idx, neighbours, C, U, Q, core_pts):
        oldU = U.copy()
        U[q_idx] = C
        neighbours = np.array(neighbours, dtype=int).copy()

        while neighbours.size > 0:
            j = int(neighbours[0])
            neighbours = np.delete(neighbours, 0)

            if U[j] == 0:
                new_neighs, _ = self._search_neighbourhood(j, Q, core_pts)
                to_add = self._setdiff_stable(new_neighs, neighbours)
                if to_add.size > 0:
                    neighbours = np.concatenate([neighbours, to_add])

                if neighbours.size > 0:
                    assigned = neighbours[U[neighbours] > 0]
                    neighbours = self._setdiff_stable(neighbours, assigned)

                U[j] = C

        current_cluster_idx = np.where(U - oldU != 0)[0]
        for i in current_cluster_idx:
            _, nearest = self._search_neighbourhood(int(i), Q, core_pts)
            nearest = self._setdiff_stable(nearest, core_pts)
            if nearest.size > 0:
                U[nearest] = C

        S = int(np.sum(oldU != U))
        return U, S

    def _get_clustering(self, p_dist, U, M, core_pts):
        idxs = np.arange(M, dtype=int)
        labeled = idxs[U != 0]
        core_idxs = np.intersect1d(labeled, core_pts, assume_unique=False)
        unlabeled = idxs[U == 0]
        L = unlabeled.size

        if unlabeled.size > 0 and L <= M * 0.5 and core_idxs.size > 0:
            for q_idx in unlabeled:
                d = p_dist[core_idxs, q_idx]
                best = core_idxs[np.argmin(d)]
                U[q_idx] = U[best]
        return U