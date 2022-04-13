import numpy as np
import time
import torch
import pdb
from pykeops.torch import LazyTensor

def KMeans(x, K=10, Niter=10, verbose=True):
    """Implements Lloyd's algorithm for the Euclidean metric."""
    
    N, D = x.shape  # Number of samples, dimension of the ambient space
    if verbose:
        print(
            f"K-means for the Euclidean metric with {N:,} points in dimension {D:,}, K = {K:,}:"
        )

    start = time.time()

    # c = x[:K, :].clone()  # Simplistic initialization for the centroids
    c = plus_plus(x, K) # k-means++ initialization

    x_i = LazyTensor(x.view(N, 1, D))  # (N, 1, D) samples
    c_j = LazyTensor(c.view(1, K, D))  # (1, K, D) centroids

    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    for i in range(Niter):

        # E step: assign points to the closest cluster -------------------------
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        c.zero_()
        c.scatter_add_(0, cl[:, None].repeat(1, D), x)

        # Divide by the number of points per cluster:
        Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
        c /= Ncl  # in-place division to compute the average

    if verbose:  # Fancy display -----------------------------------------------
        torch.cuda.synchronize()
        end = time.time()
        print(
            "Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n".format(
                Niter, end - start, Niter, (end - start) / Niter
            )
        )

    return cl, c

# Source: https://www.kdnuggets.com/2020/06/centroid-initialization-k-means-clustering.html
def plus_plus(x, k):
    """
    Create cluster centroids using the k-means++ algorithm.
    Parameters
    ----------
    ds : numpy array
        The dataset to be used for centroid initialization.
    k : int
        The desired number of clusters for which centroids are required.
    Returns
    -------
    centroids : numpy array
        Collection of k centroids as a numpy array.
    Inspiration from here: https://stackoverflow.com/questions/5466323/how-could-one-implement-the-k-means-algorithm
    """
    N, D = x.shape
    x_i = LazyTensor(x.view(N, 1, D))  # (N, 1, D) samples

    centroids = [x[0]]
    c = torch.stack(centroids, dim=0)
    c_j = LazyTensor(c.view(1, c.shape[0], D))  # (1, K, D) centroids

    for _ in range(1, k):
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
        dist_sq = D_ij.min(dim=1).view(-1)
        probs = dist_sq/dist_sq.sum()
        cumulative_probs = torch.cumsum(probs, dim=0)
        r = np.random.rand()
        
        for j, p in enumerate(cumulative_probs):
            if r < p:
                i = j
                break
        
        centroids.append(x[i])
        c = torch.stack(centroids, dim=0)
        c_j = LazyTensor(c.view(1, c.shape[0], D))  # (1, K, D) centroids

    return c