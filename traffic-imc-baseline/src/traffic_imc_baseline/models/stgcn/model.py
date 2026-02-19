from argparse import Namespace
from dataclasses import dataclass

import torch
import torch.nn as nn

from . import layers

# Code Reference:
# https://github.com/hazdzz/stgcn

@dataclass
class STGCNConfig:
    n_his: int
    Kt: int
    Ks: int
    act_func: str
    graph_conv_type: str
    gso: torch.Tensor
    enable_bias: bool
    droprate: float


class STGCNChebGraphConv(nn.Module):
    # STGCNChebGraphConv contains 'TGTND TGTND TNFF' structure
    # ChebGraphConv is the graph convolution from ChebyNet.
    # Using the Chebyshev polynomials of the first kind as a graph filter.

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (ChebGraphConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (ChebGraphConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normalization
    # F: Fully-Connected Layer
    # F: Fully-Connected Layer

    def __init__(self, args: STGCNConfig, blocks, n_vertex):
        super(STGCNChebGraphConv, self).__init__()
        modules = []
        for l in range(len(blocks) - 3):
            modules.append(
                layers.STConvBlock(
                    args.Kt,
                    args.Ks,
                    n_vertex,
                    blocks[l][-1],
                    blocks[l + 1],
                    args.act_func,
                    args.graph_conv_type,
                    args.gso,
                    args.enable_bias,
                    args.droprate,
                )
            )
        self.st_blocks = nn.Sequential(*modules)
        Ko = args.n_his - (len(blocks) - 3) * 2 * (args.Kt - 1)
        self.Ko = Ko
        if self.Ko > 1:
            self.output = layers.OutputBlock(
                Ko,
                blocks[-3][-1],
                blocks[-2],
                blocks[-1][0],
                n_vertex,
                args.act_func,
                args.enable_bias,
                args.droprate,
            )
        elif self.Ko == 0:
            self.fc1 = nn.Linear(
                in_features=blocks[-3][-1],
                out_features=blocks[-2][0],
                bias=args.enable_bias,
            )
            self.fc2 = nn.Linear(
                in_features=blocks[-2][0],
                out_features=blocks[-1][0],
                bias=args.enable_bias,
            )
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=args.droprate)

    def forward(self, x):
        x = self.st_blocks(x)
        if self.Ko > 1:
            x = self.output(x)
        elif self.Ko == 0:
            x = self.fc1(x.permute(0, 2, 3, 1))
            x = self.relu(x)
            x = self.fc2(x).permute(0, 3, 1, 2)

        return x


class STGCNGraphConv(nn.Module):
    # STGCNGraphConv contains 'TGTND TGTND TNFF' structure
    # GraphConv is the graph convolution from GCN.
    # GraphConv is not the first-order ChebConv, because the renormalization trick is adopted.
    # Be careful about over-smoothing.

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (GraphConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (GraphConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normalization
    # F: Fully-Connected Layer
    # F: Fully-Connected Layer

    def __init__(self, args: STGCNConfig, blocks, n_vertex):
        super(STGCNGraphConv, self).__init__()
        modules = []
        for l in range(len(blocks) - 3):
            modules.append(
                layers.STConvBlock(
                    args.Kt,
                    args.Ks,
                    n_vertex,
                    blocks[l][-1],
                    blocks[l + 1],
                    args.act_func,
                    args.graph_conv_type,
                    args.gso,
                    args.enable_bias,
                    args.droprate,
                )
            )
        self.st_blocks = nn.Sequential(*modules)
        Ko = args.n_his - (len(blocks) - 3) * 2 * (args.Kt - 1)
        self.Ko = Ko
        if self.Ko > 1:
            self.output = layers.OutputBlock(
                Ko,
                blocks[-3][-1],
                blocks[-2],
                blocks[-1][0],
                n_vertex,
                args.act_func,
                args.enable_bias,
                args.droprate,
            )
        elif self.Ko == 0:
            self.fc1 = nn.Linear(
                in_features=blocks[-3][-1],
                out_features=blocks[-2][0],
                bias=args.enable_bias,
            )
            self.fc2 = nn.Linear(
                in_features=blocks[-2][0],
                out_features=blocks[-1][0],
                bias=args.enable_bias,
            )
            self.relu = nn.ReLU()
            self.do = nn.Dropout(p=args.droprate)

    def forward(self, x):
        x = self.st_blocks(x)
        if self.Ko > 1:
            x = self.output(x)
        elif self.Ko == 0:
            x = self.fc1(x.permute(0, 2, 3, 1))
            x = self.relu(x)
            x = self.fc2(x).permute(0, 3, 1, 2)

        return x


class BaseSTGCN(STGCNChebGraphConv):
    def __init__(
        self,
        n_vertex: int,
        gso: torch.Tensor,
        dropout_rate: float = 0.5,
        n_his: int = 24,
        Kt: int = 3,
        stblock_num: int = 2,
        Ks: int = 3,
        act_func: str = "glu",
        graph_conv_type: str = "graph_conv",
        enable_bias: bool = True,
    ):
        # Calculate Ko (output temporal dimension)
        Ko = n_his - (Kt - 1) * 2 * stblock_num

        # Build blocks configuration
        # blocks: settings of channel size in st_conv_blocks and output layer,
        # using the bottleneck design in st_conv_blocks
        blocks = []
        blocks.append([1])
        for l in range(stblock_num):
            blocks.append([64, 16, 64])
        if Ko == 0:
            blocks.append([128])
        elif Ko > 0:
            blocks.append([128, 128])
        blocks.append([1])

        args = STGCNConfig(
            n_his=n_his,
            Kt=Kt,
            Ks=Ks,
            act_func=act_func,
            graph_conv_type=graph_conv_type,
            gso=gso,
            enable_bias=enable_bias,
            droprate=dropout_rate,
        )
        super().__init__(args, blocks, n_vertex)
