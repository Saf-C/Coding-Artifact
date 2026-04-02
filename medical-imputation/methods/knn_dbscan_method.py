# C++ MPI/OpenMP implementation is external. Keep explicit placeholder for now.
from .dbscan_method import impute_dbscan

def impute_knn_dbscan(img, eps=0.25, min_samples=5, knn_k=10):
    return impute_dbscan(img, eps=eps, min_samples=min_samples)