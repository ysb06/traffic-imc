"""Utility functions for DCRNN model.

Contains graph Laplacian and random walk matrix calculations.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg


def calculate_normalized_laplacian(adj):
    """Calculate normalized Laplacian matrix.
    
    L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    where D = diag(A @ 1)
    
    Args:
        adj: Adjacency matrix (numpy array or sparse matrix)
        
    Returns:
        Normalized Laplacian as sparse COO matrix
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = (
        sp.eye(adj.shape[0])
        - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    )
    return normalized_laplacian


def calculate_random_walk_matrix(adj_mx):
    """Calculate random walk matrix D^-1 * A.
    
    Args:
        adj_mx: Adjacency matrix
        
    Returns:
        Random walk matrix as sparse COO matrix
    """
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.0
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx


def calculate_reverse_random_walk_matrix(adj_mx):
    """Calculate reverse random walk matrix.
    
    Args:
        adj_mx: Adjacency matrix
        
    Returns:
        Reverse random walk matrix
    """
    return calculate_random_walk_matrix(np.transpose(adj_mx))


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    """Calculate scaled Laplacian matrix for Chebyshev polynomials.
    
    L_scaled = 2/lambda_max * L - I
    
    Args:
        adj_mx: Adjacency matrix
        lambda_max: Maximum eigenvalue. If None, computed from data.
        undirected: Whether to make the graph undirected
        
    Returns:
        Scaled Laplacian as sparse matrix (float32)
    """
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which="LM")
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format="csr", dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32)
