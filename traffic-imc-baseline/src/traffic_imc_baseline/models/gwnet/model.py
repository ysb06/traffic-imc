"""Graph WaveNet model."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class NConv(nn.Module):
    """Neighborhood convolution over graph nodes."""

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = torch.einsum("ncvl,vw->ncwl", (x, adj))
        return x.contiguous()


class Linear(nn.Module):
    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.mlp = nn.Conv2d(
            c_in,
            c_out,
            kernel_size=(1, 1),
            padding=(0, 0),
            stride=(1, 1),
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class GCN(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        dropout: float,
        support_len: int = 3,
        order: int = 2,
    ) -> None:
        super().__init__()
        self.nconv = NConv()
        self.mlp = Linear((order * support_len + 1) * c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(
        self,
        x: torch.Tensor,
        supports: list[torch.Tensor],
    ) -> torch.Tensor:
        out = [x]
        for adj in supports:
            x1 = self.nconv(x, adj)
            out.append(x1)
            for _ in range(2, self.order + 1):
                x2 = self.nconv(x1, adj)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        return F.dropout(h, self.dropout, training=self.training)


class GraphWaveNet(nn.Module):
    """Graph WaveNet for spatio-temporal forecasting.

    Args:
        supports: Static adjacency support matrices. They are registered as
            buffers so Lightning can move them with the module.

    Input shape:
        ``(batch, input_dim, num_nodes, seq_len)``.

    Output shape:
        ``(batch, out_dim, num_nodes, 1)`` where ``out_dim`` is the horizon.
    """

    def __init__(
        self,
        num_nodes: int,
        dropout: float = 0.3,
        supports: Optional[list[torch.Tensor]] = None,
        gcn_bool: bool = True,
        addaptadj: bool = True,
        aptinit: Optional[torch.Tensor] = None,
        in_dim: int = 2,
        out_dim: int = 12,
        residual_channels: int = 32,
        dilation_channels: int = 32,
        skip_channels: int = 256,
        end_channels: int = 512,
        kernel_size: int = 2,
        blocks: int = 4,
        layers: int = 2,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj
        self._support_names: list[str] = []

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(
            in_channels=in_dim,
            out_channels=residual_channels,
            kernel_size=(1, 1),
        )

        support_len = 0
        if supports is not None:
            for i, support in enumerate(supports):
                name = f"support_{i}"
                self.register_buffer(name, support.float())
                self._support_names.append(name)
            support_len += len(supports)

        if gcn_bool and addaptadj:
            if aptinit is None:
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10))
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes))
            else:
                m, p, vh = torch.linalg.svd(aptinit.float())
                rank = min(10, p.size(0))
                initemb1 = torch.mm(m[:, :rank], torch.diag(p[:rank] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:rank] ** 0.5), vh[:rank, :])
                self.nodevec1 = nn.Parameter(initemb1)
                self.nodevec2 = nn.Parameter(initemb2)
            support_len += 1

        receptive_field = 1
        for _ in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for _ in range(layers):
                self.filter_convs.append(
                    nn.Conv2d(
                        in_channels=residual_channels,
                        out_channels=dilation_channels,
                        kernel_size=(1, kernel_size),
                        dilation=new_dilation,
                    )
                )
                self.gate_convs.append(
                    nn.Conv2d(
                        in_channels=residual_channels,
                        out_channels=dilation_channels,
                        kernel_size=(1, kernel_size),
                        dilation=new_dilation,
                    )
                )
                self.residual_convs.append(
                    nn.Conv2d(
                        in_channels=dilation_channels,
                        out_channels=residual_channels,
                        kernel_size=(1, 1),
                    )
                )
                self.skip_convs.append(
                    nn.Conv2d(
                        in_channels=dilation_channels,
                        out_channels=skip_channels,
                        kernel_size=(1, 1),
                    )
                )
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(
                        GCN(
                            dilation_channels,
                            residual_channels,
                            dropout,
                            support_len=support_len,
                        )
                    )

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
        self.receptive_field = receptive_field

    @property
    def supports(self) -> list[torch.Tensor]:
        return [getattr(self, name) for name in self._support_names]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        in_len = input.size(3)
        if in_len < self.receptive_field:
            x = F.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input
        x = self.start_conv(x)

        skip: torch.Tensor | int = 0
        supports = self.supports
        if self.gcn_bool and self.addaptadj:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            supports = supports + [adp]

        for i in range(self.blocks * self.layers):
            residual = x
            filter_out = torch.tanh(self.filter_convs[i](residual))
            gate = torch.sigmoid(self.gate_convs[i](residual))
            x = filter_out * gate

            s = self.skip_convs[i](x)
            if isinstance(skip, torch.Tensor):
                skip = skip[:, :, :, -s.size(3) :]
            else:
                skip = 0
            skip = s + skip

            if self.gcn_bool and supports:
                x = self.gconv[i](x, supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3) :]
            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x[:, :, :, -1:]


gwnet = GraphWaveNet
