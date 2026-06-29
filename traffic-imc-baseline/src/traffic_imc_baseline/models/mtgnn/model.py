"""MTGNN model."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layer import LayerNorm, dilated_inception, graph_constructor, mixprop


class MTGNN(nn.Module):
    """Multivariate Time Series Graph Neural Network.

    Input shape:
        ``(batch, input_dim, num_nodes, seq_len)``.

    Output shape:
        ``(batch, horizon, num_nodes, 1)``.
    """

    def __init__(
        self,
        gcn_true: bool,
        buildA_true: bool,
        gcn_depth: int,
        num_nodes: int,
        predefined_A: Optional[torch.Tensor] = None,
        static_feat: Optional[torch.Tensor] = None,
        dropout: float = 0.3,
        subgraph_size: int = 20,
        node_dim: int = 40,
        dilation_exponential: int = 1,
        conv_channels: int = 32,
        residual_channels: int = 32,
        skip_channels: int = 64,
        end_channels: int = 128,
        seq_length: int = 12,
        in_dim: int = 2,
        out_dim: int = 12,
        layers: int = 3,
        propalpha: float = 0.05,
        tanhalpha: float = 3,
        layer_norm_affline: bool = True,
    ) -> None:
        super().__init__()
        if num_nodes < 1:
            raise ValueError("num_nodes must be positive.")
        if subgraph_size < 1:
            raise ValueError("subgraph_size must be positive.")
        if conv_channels % 4 != 0:
            raise ValueError("conv_channels must be divisible by 4.")

        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.seq_length = seq_length
        self.layers = layers

        self.register_buffer("idx", torch.arange(num_nodes, dtype=torch.long))
        if predefined_A is None:
            self.register_buffer("predefined_A", None)
        else:
            self.register_buffer("predefined_A", predefined_A.float())

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.start_conv = nn.Conv2d(
            in_channels=in_dim,
            out_channels=residual_channels,
            kernel_size=(1, 1),
        )
        self.gc = graph_constructor(
            num_nodes,
            min(subgraph_size, num_nodes),
            node_dim,
            alpha=tanhalpha,
            static_feat=static_feat,
        )

        kernel_size = 7
        if dilation_exponential > 1:
            self.receptive_field = int(
                1
                + (kernel_size - 1)
                * (dilation_exponential**layers - 1)
                / (dilation_exponential - 1)
            )
        else:
            self.receptive_field = layers * (kernel_size - 1) + 1

        for i in range(1):
            if dilation_exponential > 1:
                rf_size_i = int(
                    1
                    + i
                    * (kernel_size - 1)
                    * (dilation_exponential**layers - 1)
                    / (dilation_exponential - 1)
                )
            else:
                rf_size_i = i * layers * (kernel_size - 1) + 1
            new_dilation = 1
            for j in range(1, layers + 1):
                if dilation_exponential > 1:
                    rf_size_j = int(
                        rf_size_i
                        + (kernel_size - 1)
                        * (dilation_exponential**j - 1)
                        / (dilation_exponential - 1)
                    )
                else:
                    rf_size_j = rf_size_i + j * (kernel_size - 1)

                self.filter_convs.append(
                    dilated_inception(
                        residual_channels,
                        conv_channels,
                        dilation_factor=new_dilation,
                    )
                )
                self.gate_convs.append(
                    dilated_inception(
                        residual_channels,
                        conv_channels,
                        dilation_factor=new_dilation,
                    )
                )
                self.residual_convs.append(
                    nn.Conv2d(
                        in_channels=conv_channels,
                        out_channels=residual_channels,
                        kernel_size=(1, 1),
                    )
                )
                if self.seq_length > self.receptive_field:
                    skip_kernel = self.seq_length - rf_size_j + 1
                else:
                    skip_kernel = self.receptive_field - rf_size_j + 1
                self.skip_convs.append(
                    nn.Conv2d(
                        in_channels=conv_channels,
                        out_channels=skip_channels,
                        kernel_size=(1, skip_kernel),
                    )
                )

                if self.gcn_true:
                    self.gconv1.append(
                        mixprop(
                            conv_channels,
                            residual_channels,
                            gcn_depth,
                            dropout,
                            propalpha,
                        )
                    )
                    self.gconv2.append(
                        mixprop(
                            conv_channels,
                            residual_channels,
                            gcn_depth,
                            dropout,
                            propalpha,
                        )
                    )

                if self.seq_length > self.receptive_field:
                    norm_shape = (
                        residual_channels,
                        num_nodes,
                        self.seq_length - rf_size_j + 1,
                    )
                else:
                    norm_shape = (
                        residual_channels,
                        num_nodes,
                        self.receptive_field - rf_size_j + 1,
                    )
                self.norm.append(
                    LayerNorm(norm_shape, elementwise_affine=layer_norm_affline)
                )

                new_dilation *= dilation_exponential

        self.end_conv_1 = nn.Conv2d(
            in_channels=skip_channels,
            out_channels=end_channels,
            kernel_size=(1, 1),
            bias=True,
        )
        self.end_conv_2 = nn.Conv2d(
            in_channels=end_channels,
            out_channels=out_dim,
            kernel_size=(1, 1),
            bias=True,
        )
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(
                in_channels=in_dim,
                out_channels=skip_channels,
                kernel_size=(1, self.seq_length),
                bias=True,
            )
            self.skipE = nn.Conv2d(
                in_channels=residual_channels,
                out_channels=skip_channels,
                kernel_size=(1, self.seq_length - self.receptive_field + 1),
                bias=True,
            )
        else:
            self.skip0 = nn.Conv2d(
                in_channels=in_dim,
                out_channels=skip_channels,
                kernel_size=(1, self.receptive_field),
                bias=True,
            )
            self.skipE = nn.Conv2d(
                in_channels=residual_channels,
                out_channels=skip_channels,
                kernel_size=(1, 1),
                bias=True,
            )

    def forward(
        self,
        input: torch.Tensor,
        idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        seq_len = input.size(3)
        if seq_len != self.seq_length:
            raise ValueError("input sequence length not equal to preset sequence length")

        if self.seq_length < self.receptive_field:
            input = F.pad(input, (self.receptive_field - self.seq_length, 0, 0, 0))

        if idx is not None:
            idx = idx.to(input.device)

        adp = None
        if self.gcn_true:
            if self.buildA_true:
                adp = self.gc(self.idx if idx is None else idx)
            else:
                if self.predefined_A is None:
                    raise ValueError("predefined_A is required when buildA_true=False.")
                adp = self.predefined_A
                if idx is not None:
                    adp = adp.index_select(0, idx).index_select(1, idx)

        x = self.start_conv(input)
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))
        for i in range(self.layers):
            residual = x
            filter_out = torch.tanh(self.filter_convs[i](x))
            gate = torch.sigmoid(self.gate_convs[i](x))
            x = filter_out * gate
            x = F.dropout(x, self.dropout, training=self.training)

            s = self.skip_convs[i](x)
            skip = s + skip
            if self.gcn_true:
                if adp is None:
                    raise RuntimeError("Graph adjacency is not initialized.")
                x = self.gconv1[i](x, adp) + self.gconv2[i](x, adp.transpose(1, 0))
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3) :]
            x = self.norm[i](x, self.idx if idx is None else idx)

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        return self.end_conv_2(x)


gtnet = MTGNN
