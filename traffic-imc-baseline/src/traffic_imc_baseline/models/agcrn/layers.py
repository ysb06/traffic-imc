"""
AGCRN Layer implementations.

Contains:
- AVWGCN: Adaptive View-Weighted Graph Convolution
- AGCRNCell: AGCRN GRU Cell with graph convolution
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AVWGCN(nn.Module):
    """Adaptive View-Weighted Graph Convolution Network.
    
    Learns spatial dependencies through node embeddings without
    requiring a predefined adjacency matrix.
    
    Args:
        dim_in: Input feature dimension
        dim_out: Output feature dimension
        cheb_k: Order of Chebyshev polynomials
        embed_dim: Node embedding dimension
    """
    
    def __init__(self, dim_in: int, dim_out: int, cheb_k: int, embed_dim: int):
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(
            torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out)
        )
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        
        # Initialize parameters
        nn.init.xavier_uniform_(self.weights_pool)
        nn.init.zeros_(self.bias_pool)
    
    def forward(self, x: torch.Tensor, node_embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, N, C)
            node_embeddings: Node embeddings of shape (N, embed_dim)
            
        Returns:
            Output tensor of shape (B, N, dim_out)
        """
        node_num = node_embeddings.shape[0]
        
        # Compute adaptive adjacency matrix from node embeddings
        # supports: (N, N) - learned spatial relationships
        supports = F.softmax(
            F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), 
            dim=1
        )
        
        # Build Chebyshev polynomial basis
        support_set = [torch.eye(node_num).to(supports.device), supports]
        for k in range(2, self.cheb_k):
            support_set.append(
                torch.matmul(2 * supports, support_set[-1]) - support_set[-2]
            )
        supports = torch.stack(support_set, dim=0)  # (cheb_k, N, N)
        
        # Node-specific weights and bias
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)
        bias = torch.matmul(node_embeddings, self.bias_pool)  # (N, dim_out)
        
        # Graph convolution
        x_g = torch.einsum("knm,bmc->bknc", supports, x)  # (B, cheb_k, N, dim_in)
        x_g = x_g.permute(0, 2, 1, 3)  # (B, N, cheb_k, dim_in)
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias  # (B, N, dim_out)
        
        return x_gconv


class AGCRNCell(nn.Module):
    """AGCRN GRU Cell with graph convolution.
    
    A GRU-style recurrent cell that uses AVWGCN for spatial aggregation.
    
    Args:
        node_num: Number of nodes
        dim_in: Input feature dimension
        dim_out: Hidden state dimension
        cheb_k: Order of Chebyshev polynomials
        embed_dim: Node embedding dimension
    """
    
    def __init__(
        self, 
        node_num: int, 
        dim_in: int, 
        dim_out: int, 
        cheb_k: int, 
        embed_dim: int
    ):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        
        # GRU gates using graph convolution
        self.gate = AVWGCN(dim_in + self.hidden_dim, 2 * dim_out, cheb_k, embed_dim)
        self.update = AVWGCN(dim_in + self.hidden_dim, dim_out, cheb_k, embed_dim)
    
    def forward(
        self, 
        x: torch.Tensor, 
        state: torch.Tensor, 
        node_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, N, input_dim)
            state: Hidden state tensor of shape (B, N, hidden_dim)
            node_embeddings: Node embeddings of shape (N, embed_dim)
            
        Returns:
            New hidden state of shape (B, N, hidden_dim)
        """
        state = state.to(x.device)
        
        # Concatenate input and state
        input_and_state = torch.cat((x, state), dim=-1)
        
        # Compute reset and update gates
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        
        # Compute candidate hidden state (r = reset gate)
        candidate = torch.cat((x, r * state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings))
        
        # Compute new hidden state (z = update gate)
        h = z * state + (1 - z) * hc
        
        return h
    
    def init_hidden_state(self, batch_size: int) -> torch.Tensor:
        """Initialize hidden state to zeros.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Zero tensor of shape (batch_size, node_num, hidden_dim)
        """
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)
