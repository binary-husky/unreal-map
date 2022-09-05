import numpy as np

def pca(samples, target_dim):
    assert len(samples.shape) == 2
    data = samples - np.mean(samples,axis=0)  # mean at batch dim
    covMat = np.cov(data,rowvar=0)
    fValue,fVector = np.linalg.eig(covMat)
    fValueSort = np.argsort(-fValue)
    fValueTopN = fValueSort[:target_dim]
    fvectormat = fVector[:,fValueTopN]
    down_dim_data = np.dot(data, fvectormat)
    return down_dim_data