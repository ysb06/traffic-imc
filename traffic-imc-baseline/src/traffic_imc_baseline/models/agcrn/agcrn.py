import torch
import torch.nn as nn
from typing import Tuple, List

from .layers import AGCRNCell


class AVWDCRNN(nn.Module):
    def __init__(
        self, 
        node_num: int, 
        dim_in: int, 
        dim_out: int, 
        cheb_k: int, 
        embed_dim: int, 
        num_layers: int = 1
    ):
        super(AVWDCRNN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        
        # Stack of AGCRN cells
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))
    
    def forward(
        self, 
        x: torch.Tensor, 
        init_state: torch.Tensor, 
        node_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            
            for t in range(seq_length):
                state = self.dcrnn_cells[i](
                    current_inputs[:, t, :, :], 
                    state, 
                    node_embeddings
                )
                inner_states.append(state)
            
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        
        # current_inputs: outputs of last layer (B, T, N, hidden_dim)
        # output_hidden: final state for each layer (num_layers, B, N, hidden_dim)
        return current_inputs, output_hidden
    
    def init_hidden(self, batch_size: int) -> torch.Tensor:
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)


class AGCRN(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        input_dim: int = 1,
        output_dim: int = 1,
        horizon: int = 24,
        rnn_units: int = 64,
        num_layers: int = 2,
        embed_dim: int = 10,
        cheb_k: int = 2,
    ):
        super(AGCRN, self).__init__()
        
        self.num_node = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = rnn_units
        self.output_dim = output_dim
        self.horizon = horizon
        self.num_layers = num_layers
        
        # Learnable node embeddings for adaptive graph construction
        self.node_embeddings = nn.Parameter(
            torch.randn(num_nodes, embed_dim), 
            requires_grad=True
        )
        
        # Encoder
        self.encoder = AVWDCRNN(
            node_num=num_nodes, 
            dim_in=input_dim, 
            dim_out=rnn_units, 
            cheb_k=cheb_k,
            embed_dim=embed_dim, 
            num_layers=num_layers
        )
        
        # CNN-based predictor
        self.end_conv = nn.Conv2d(
            in_channels=1, 
            out_channels=horizon * output_dim, 
            kernel_size=(1, rnn_units), 
            bias=True
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.uniform_(param)
    
    def forward(
        self, 
        source: torch.Tensor, 
        targets: torch.Tensor = None, 
        teacher_forcing_ratio: float = 0.0
    ) -> torch.Tensor:
        batch_size = source.shape[0]
        
        # Initialize hidden states
        init_state = self.encoder.init_hidden(batch_size)
        init_state = init_state.to(source.device)
        
        # Encode input sequence
        output, _ = self.encoder(source, init_state, self.node_embeddings)
        # output: (B, T, N, hidden_dim)
        
        # Use only the last hidden state for prediction
        output = output[:, -1:, :, :]  # (B, 1, N, hidden_dim)
        
        # CNN predictor
        output = self.end_conv(output)  # (B, horizon*output_dim, N, 1)
        output = output.squeeze(-1)  # (B, horizon*output_dim, N)
        output = output.reshape(batch_size, self.horizon, self.output_dim, self.num_node)
        output = output.permute(0, 1, 3, 2)  # (B, horizon, N, output_dim)
        
        return output
