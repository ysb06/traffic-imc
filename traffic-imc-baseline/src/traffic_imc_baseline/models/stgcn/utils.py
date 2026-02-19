"""
Graph Shift Operator (GSO) utilities for STGCN model.

This module provides functions to compute various types of GSO from adjacency matrices.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import norm
import torch
from typing import Literal

GSO_TYPE = Literal[
    "sym_norm_adj",
    "sym_renorm_adj",
    "rw_norm_adj",
    "rw_renorm_adj",
    "sym_norm_lap",
    "sym_renorm_lap",
    "rw_norm_lap",
    "rw_renorm_lap",
]


def calc_gso(
    dir_adj: np.ndarray, 
    gso_type: GSO_TYPE,
    force_symmetric: bool = True
) -> sp.csc_matrix:
    """
    Calculate Graph Shift Operator (GSO) from adjacency matrix.

    Args:
        dir_adj: Adjacency matrix of shape (n_vertex, n_vertex).
                 Can be directed or undirected.
        gso_type: Type of GSO to compute
            - 'sym_norm_adj': Symmetric normalized adjacency
            - 'sym_renorm_adj': Symmetric renormalized (with self-loops)
            - 'rw_norm_adj': Random walk normalized adjacency
            - 'rw_renorm_adj': Random walk renormalized
            - 'sym_norm_lap': Symmetric normalized Laplacian (recommended)
            - 'sym_renorm_lap': Symmetric renormalized Laplacian
            - 'rw_norm_lap': Random walk normalized Laplacian
            - 'rw_renorm_lap': Random walk renormalized Laplacian
        force_symmetric: If True, symmetrize the adjacency matrix by taking max(A[i,j], A[j,i]).
                        If False, preserve directionality (use with caution for 'sym_*' types).
                        Default: True (original STGCN behavior)

    Returns:
        GSO matrix in CSC sparse format
        
    Note:
        - Symmetrization (force_symmetric=True) will lose directional information
        - For directed graphs, consider using 'rw_norm_*' types with force_symmetric=False
        - Traffic networks: directionality loss may be acceptable if focusing on spatial correlation
    """
    n_vertex = dir_adj.shape[0]

    if sp.issparse(dir_adj) == False:
        dir_adj = sp.csc_matrix(dir_adj)
    elif dir_adj.format != 'csc':
        dir_adj = dir_adj.tocsc()

    id = sp.identity(n_vertex, format='csc')

    # Symmetrizing an adjacency matrix (optional)
    if force_symmetric:
        # Take max(A[i,j], A[j,i]) for each edge
        # This loses directional information but ensures mathematical properties
        adj = dir_adj + dir_adj.T.multiply(dir_adj.T > dir_adj) - dir_adj.multiply(dir_adj.T > dir_adj)
        #adj = 0.5 * (dir_adj + dir_adj.transpose())  # Alternative: average
    else:
        # Preserve directionality (use with caution for symmetric GSO types)
        adj = dir_adj
    
    if gso_type == 'sym_renorm_adj' or gso_type == 'rw_renorm_adj' \
        or gso_type == 'sym_renorm_lap' or gso_type == 'rw_renorm_lap':
        adj = adj + id
    
    if gso_type == 'sym_norm_adj' or gso_type == 'sym_renorm_adj' \
        or gso_type == 'sym_norm_lap' or gso_type == 'sym_renorm_lap':
        row_sum = adj.sum(axis=1).A1
        row_sum_inv_sqrt = np.power(row_sum, -0.5)
        row_sum_inv_sqrt[np.isinf(row_sum_inv_sqrt)] = 0.
        deg_inv_sqrt = sp.diags(row_sum_inv_sqrt, format='csc')
        # A_{sym} = D^{-0.5} * A * D^{-0.5}
        sym_norm_adj = deg_inv_sqrt.dot(adj).dot(deg_inv_sqrt)

        if gso_type == 'sym_norm_lap' or gso_type == 'sym_renorm_lap':
            sym_norm_lap = id - sym_norm_adj
            gso = sym_norm_lap
        else:
            gso = sym_norm_adj

    elif gso_type == 'rw_norm_adj' or gso_type == 'rw_renorm_adj' \
        or gso_type == 'rw_norm_lap' or gso_type == 'rw_renorm_lap':
        row_sum = np.sum(adj, axis=1).A1
        row_sum_inv = np.power(row_sum, -1)
        row_sum_inv[np.isinf(row_sum_inv)] = 0.
        deg_inv = sp.diags(row_sum_inv, format='csc')  # Fixed: Use sparse matrix
        # A_{rw} = D^{-1} * A
        rw_norm_adj = deg_inv.dot(adj)

        if gso_type == 'rw_norm_lap' or gso_type == 'rw_renorm_lap':
            rw_norm_lap = id - rw_norm_adj
            gso = rw_norm_lap
        else:
            gso = rw_norm_adj

    else:
        raise ValueError(f'{gso_type} is not defined.')

    return gso

def calc_chebynet_gso(gso: sp.csc_matrix) -> sp.csc_matrix:
    """
    Scale GSO for Chebyshev graph convolution.

    This function rescales the eigenvalues of the GSO to the range [-1, 1],
    which is required for Chebyshev polynomial approximation.

    Args:
        gso: Graph Shift Operator in CSC sparse format

    Returns:
        Scaled GSO for Chebyshev graph convolution

    Note:
        Requires scipy >= 1.10.1 for the norm() function.
    """
    if sp.issparse(gso) == False:
        gso = sp.csc_matrix(gso)
    elif gso.format != 'csc':
        gso = gso.tocsc()

    id = sp.identity(gso.shape[0], format='csc')
    # If you encounter a NotImplementedError, please update your scipy version to 1.10.1 or later.
    eigval_max = norm(gso, 2)

    # If the gso is symmetric or random walk normalized Laplacian,
    # then the maximum eigenvalue is smaller than or equals to 2.
    if eigval_max >= 2:
        gso = gso - id
    else:
        gso = 2 * gso / eigval_max - id

    return gso

def cnv_sparse_mat_to_coo_tensor(
    sp_mat: sp.csc_matrix, device: torch.device
) -> torch.Tensor:
    """
    Convert scipy sparse matrix to PyTorch sparse COO tensor.

    Args:
        sp_mat: Scipy sparse matrix (CSR or CSC format)
        device: Target device for the tensor ('cpu' or 'cuda')

    Returns:
        PyTorch sparse COO tensor
    """
    # convert a compressed sparse row (csr) or compressed sparse column (csc) matrix to a hybrid sparse coo tensor
    sp_coo_mat = sp_mat.tocoo()
    i = torch.from_numpy(np.vstack((sp_coo_mat.row, sp_coo_mat.col)))
    v = torch.from_numpy(sp_coo_mat.data)
    s = torch.Size(sp_coo_mat.shape)

    if sp_mat.dtype == np.float32 or sp_mat.dtype == np.float64:
        return torch.sparse_coo_tensor(indices=i, values=v, size=s, dtype=torch.float32, device=device, requires_grad=False)
    else:
        raise TypeError(f'ERROR: The dtype of {sp_mat} is {sp_mat.dtype}, not been applied in implemented models.')


def prepare_gso_for_model(
    adj_mx: np.ndarray,
    gso_type: GSO_TYPE = "sym_norm_lap",
    graph_conv_type: Literal["cheb_graph_conv", "graph_conv"] = "graph_conv",
    device: torch.device = torch.device("cpu"),
    force_symmetric: bool = True,
) -> torch.Tensor:
    """
    Prepare GSO tensor for STGCN model (convenience function).

    This combines GSO calculation and conversion to PyTorch tensor,
    with optional Chebyshev scaling.

    Args:
        adj_mx: Adjacency matrix of shape (n_vertex, n_vertex)
        gso_type: Type of GSO to compute (default: 'sym_norm_lap')
        graph_conv_type: Type of graph convolution
            - 'cheb_graph_conv': Apply Chebyshev scaling
            - 'graph_conv': Use GSO directly
        device: Target device for the tensor
        force_symmetric: Whether to symmetrize the adjacency matrix (default: True)

    Returns:
        GSO as PyTorch dense tensor

    Example:
        >>> from metr.components.adj_mx import AdjacencyMatrix
        >>> adj_mx_obj = AdjacencyMatrix.import_from_pickle('adj_mx.pkl')
        >>> 
        >>> # Standard usage (symmetric, loses directionality)
        >>> gso_tensor = prepare_gso_for_model(
        ...     adj_mx_obj.adj_mx,
        ...     gso_type='sym_norm_lap',
        ...     device=torch.device('cuda')
        ... )
        >>> 
        >>> # Preserve directionality (experimental)
        >>> gso_tensor = prepare_gso_for_model(
        ...     adj_mx_obj.adj_mx,
        ...     gso_type='rw_norm_lap',
        ...     force_symmetric=False,
        ...     device=torch.device('cuda')
        ... )
    """
    # Calculate GSO
    gso = calc_gso(adj_mx, gso_type=gso_type, force_symmetric=force_symmetric)

    # Apply Chebyshev scaling if needed
    if graph_conv_type == "cheb_graph_conv":
        gso = calc_chebynet_gso(gso)

    # Convert to dense tensor (STGCN models expect dense tensors)
    if sp.issparse(gso):
        gso = gso.toarray()

    gso_tensor = torch.FloatTensor(gso).to(device)

    return gso_tensor


def evaluate_model(model, loss, data_iter):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            y_pred = model(x).view(len(x), -1)
            l = loss(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        mse = l_sum / n
        
        return mse

def evaluate_metric(model, data_iter, scaler):
    model.eval()
    with torch.no_grad():
        mae, sum_y, mape, mse = [], [], [], []
        for x, y in data_iter:
            y = scaler.inverse_transform(y.cpu().numpy()).reshape(-1)
            y_pred = scaler.inverse_transform(model(x).view(len(x), -1).cpu().numpy()).reshape(-1)
            d = np.abs(y - y_pred)
            mae += d.tolist()
            sum_y += y.tolist()
            mape += (d / y).tolist()
            mse += (d ** 2).tolist()
        MAE = np.array(mae).mean()
        #MAPE = np.array(mape).mean()
        RMSE = np.sqrt(np.array(mse).mean())
        WMAPE = np.sum(np.array(mae)) / np.sum(np.array(sum_y))

        #return MAE, MAPE, RMSE
        return MAE, RMSE, WMAPE