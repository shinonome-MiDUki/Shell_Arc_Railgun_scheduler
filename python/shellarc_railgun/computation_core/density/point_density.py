import numpy as np
from sklearn.neighbors import NearestNeighbors

from shellarc_railgun.data_model import data_struct 

def get_dense_areas(M: np.ndarray,
                    top_n: int = 5,
                    k: int = 25
                    ) -> np.ndarray:
    M = M[data_struct.NpData.POS]
    nn = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(M)
    distances, _ = nn.kneighbors(M)
    k_distances = distances[:, -1]
    
    sorted_indices = np.argsort(k_distances)
    dense_centers = []
    
    M_normalized = (M - M.mean(axis=0)) / (M.std(axis=0) + 1e-9)
    min_distance_threshold = np.std(M_normalized) * 0.1
    
    for idx in sorted_indices:
        candidate = M[idx]
        if all(np.linalg.norm(candidate - center) > min_distance_threshold for center in dense_centers):
            dense_centers.append(candidate)
        if len(dense_centers) == top_n:
            break
            
    return np.array(dense_centers)