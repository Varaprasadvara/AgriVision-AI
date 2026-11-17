from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
import numpy as np
try:
    from src.config import RANDOM_STATE
except ImportError:
    from config import RANDOM_STATE

def run_pca(data, n_components=2):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(data)

def run_kmeans(data, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=RANDOM_STATE)
    return kmeans.fit_predict(data)

def run_dbscan(data, eps=0.5, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    return dbscan.fit_predict(data)

def run_anomaly_detection(data, contamination=0.1):
    iso = IsolationForest(contamination=contamination, random_state=RANDOM_STATE)
    return iso.fit_predict(data)
