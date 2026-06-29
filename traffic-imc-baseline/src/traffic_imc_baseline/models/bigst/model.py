"""BigST model components."""

import math

import numpy as np
import torch
import torch.nn as nn


class BigST(nn.Module):
    """Linear-complexity spatio-temporal graph neural network."""

    def __init__(
        self,
        num_nodes: int,
        input_length: int = 24,
        output_length: int = 24,
        input_dim: int = 3,
        hid_dim: int = 32,
        num_layers: int = 3,
        tau: float = 0.25,
        random_feature_dim: int = 64,
        node_dim: int = 32,
        time_dim: int = 32,
        time_num: int = 24,
        week_num: int = 7,
        dropout: float = 0.3,
        use_residual: bool = True,
        use_bn: bool = True,
        use_spatial: bool = False,
        supports: list[torch.Tensor] | None = None,
        edge_indices: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if input_dim != 3:
            raise ValueError("BigST expects input_dim=3: traffic, tod, dow.")
        if use_spatial and (not supports or edge_indices is None):
            raise ValueError(
                "supports and edge_indices are required when use_spatial=True."
            )

        self.num_nodes = num_nodes
        self.input_length = input_length
        self.output_length = output_length
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.tau = tau
        self.num_layers = num_layers
        self.random_feature_dim = random_feature_dim
        self.use_residual = use_residual
        self.use_bn = use_bn
        self.use_spatial = use_spatial
        self.dropout = dropout
        self.time_num = time_num
        self.week_num = week_num
        self.hidden_dim = hid_dim + node_dim + time_dim * 2

        self.activation = nn.ReLU()
        self.node_emb_layer = nn.Parameter(torch.empty(num_nodes, node_dim))
        nn.init.xavier_uniform_(self.node_emb_layer)

        self.time_emb_layer = nn.Parameter(torch.empty(time_num, time_dim))
        nn.init.xavier_uniform_(self.time_emb_layer)
        self.week_emb_layer = nn.Parameter(torch.empty(week_num, time_dim))
        nn.init.xavier_uniform_(self.week_emb_layer)

        self.input_emb_layer = nn.Conv2d(
            input_length * input_dim,
            hid_dim,
            kernel_size=(1, 1),
            bias=True,
        )
        self.W_1 = nn.Conv2d(
            node_dim + time_dim * 2,
            hid_dim,
            kernel_size=(1, 1),
            bias=True,
        )
        self.W_2 = nn.Conv2d(
            node_dim + time_dim * 2,
            hid_dim,
            kernel_size=(1, 1),
            bias=True,
        )

        self.linear_conv = nn.ModuleList()
        self.bn = nn.ModuleList()
        for _ in range(num_layers):
            self.linear_conv.append(
                LinearizedConv(
                    self.hidden_dim,
                    self.hidden_dim,
                    dropout,
                    tau,
                    random_feature_dim,
                )
            )
            self.bn.append(nn.LayerNorm(self.hidden_dim))

        self.regression_layer = nn.Conv2d(
            self.hidden_dim * 2,
            output_length,
            kernel_size=(1, 1),
            bias=True,
        )

        self._support_names: list[str] = []
        for i, support in enumerate(supports or []):
            name = f"support_{i}"
            self.register_buffer(name, support.detach().clone().float())
            self._support_names.append(name)

        if edge_indices is None:
            self.edge_indices = None
        else:
            self.register_buffer(
                "edge_indices",
                edge_indices.detach().clone().long(),
            )

    @property
    def supports(self) -> list[torch.Tensor]:
        return [getattr(self, name) for name in self._support_names]

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # input: (B, N, T, D)
        batch_size, num_nodes, _, _ = x.size()

        time_idx = (x[:, :, -1, 1] * self.time_num).long()
        time_idx = time_idx.clamp(0, self.time_num - 1)
        week_idx = x[:, :, -1, 2].long().clamp(0, self.week_num - 1)

        time_emb = self.time_emb_layer[time_idx]
        week_emb = self.week_emb_layer[week_idx]

        x = x.contiguous().view(batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        input_emb = self.input_emb_layer(x)

        node_emb = (
            self.node_emb_layer.unsqueeze(0)
            .expand(batch_size, -1, -1)
            .transpose(1, 2)
            .unsqueeze(-1)
        )
        time_emb = time_emb.transpose(1, 2).unsqueeze(-1)
        week_emb = week_emb.transpose(1, 2).unsqueeze(-1)

        x_g = torch.cat([node_emb, time_emb, week_emb], dim=1)
        x = torch.cat([input_emb, node_emb, time_emb, week_emb], dim=1)

        x_pool = [x]
        node_vec1 = self.W_1(x_g).permute(0, 2, 3, 1)
        node_vec2 = self.W_2(x_g).permute(0, 2, 3, 1)
        node_vec1_prime = node_vec1
        node_vec2_prime = node_vec2

        for layer_idx in range(self.num_layers):
            residual = x
            x, node_vec1_prime, node_vec2_prime = self.linear_conv[layer_idx](
                x,
                node_vec1,
                node_vec2,
            )

            if self.use_residual:
                x = x + residual

            if self.use_bn:
                x = x.permute(0, 2, 3, 1)
                x = self.bn[layer_idx](x)
                x = x.permute(0, 3, 1, 2)

        x_pool.append(x)
        x = torch.cat(x_pool, dim=1)
        x = self.activation(x)

        x = self.regression_layer(x)
        x = x.squeeze(-1).permute(0, 2, 1)

        if self.use_spatial:
            if self.edge_indices is None:
                raise RuntimeError("edge_indices is not initialized.")
            s_loss = spatial_loss(
                node_vec1_prime,
                node_vec2_prime,
                self.supports,
                self.edge_indices,
            )
            return x, s_loss

        return x, x.new_zeros(())


def create_products_of_givens_rotations(dim: int, seed: int) -> torch.Tensor:
    nb_givens_rotations = dim * int(math.ceil(math.log(float(dim))))
    q = np.eye(dim, dim)
    rng = np.random.default_rng(seed)
    for _ in range(nb_givens_rotations):
        random_angle = math.pi * rng.uniform()
        random_indices = rng.choice(dim, 2, replace=False)
        index_i = min(random_indices[0], random_indices[1])
        index_j = max(random_indices[0], random_indices[1])
        slice_i = q[index_i]
        slice_j = q[index_j]
        new_slice_i = math.cos(random_angle) * slice_i + math.cos(random_angle) * slice_j
        new_slice_j = -math.sin(random_angle) * slice_i + math.cos(random_angle) * slice_j
        q[index_i] = new_slice_i
        q[index_j] = new_slice_j
    return torch.tensor(q, dtype=torch.float32)


def _orthogonal_matrix(dim: int, seed: int, struct_mode: bool) -> torch.Tensor:
    if struct_mode:
        return create_products_of_givens_rotations(dim, seed)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    unstructured_block = torch.randn((dim, dim), generator=generator)
    q, _ = torch.linalg.qr(unstructured_block, mode="reduced")
    return torch.t(q)


def create_random_matrix(
    m: int,
    d: int,
    seed: int = 0,
    scaling: int = 0,
    struct_mode: bool = False,
) -> torch.Tensor:
    seed = int(seed)
    nb_full_blocks = int(m / d)
    block_list = []
    current_seed = seed
    for _ in range(nb_full_blocks):
        block_list.append(_orthogonal_matrix(d, current_seed, struct_mode))
        current_seed += 1

    remaining_rows = m - nb_full_blocks * d
    if remaining_rows > 0:
        q = _orthogonal_matrix(d, current_seed, struct_mode)
        block_list.append(q[0:remaining_rows])

    final_matrix = torch.vstack(block_list)

    current_seed += 1
    generator = torch.Generator(device="cpu")
    generator.manual_seed(current_seed)
    if scaling == 0:
        multiplier = torch.norm(torch.randn((m, d), generator=generator), dim=1)
    elif scaling == 1:
        multiplier = torch.sqrt(torch.tensor(float(d))) * torch.ones(m)
    else:
        raise ValueError(f"Scaling must be one of {{0, 1}}. Was {scaling}")

    return torch.matmul(torch.diag(multiplier), final_matrix)


def random_feature_map(
    data: torch.Tensor,
    is_query: bool,
    projection_matrix: torch.Tensor,
    numerical_stabilizer: float = 0.000001,
) -> torch.Tensor:
    data_normalizer = 1.0 / torch.sqrt(
        torch.sqrt(torch.tensor(data.shape[-1], dtype=torch.float32, device=data.device))
    )
    data = data_normalizer * data
    ratio = 1.0 / torch.sqrt(
        torch.tensor(projection_matrix.shape[0], dtype=torch.float32, device=data.device)
    )
    data_dash = torch.einsum("bnhd,md->bnhm", data, projection_matrix)
    diag_data = torch.square(data)
    diag_data = torch.sum(diag_data, dim=len(data.shape) - 1)
    diag_data = diag_data / 2.0
    diag_data = torch.unsqueeze(diag_data, dim=len(data.shape) - 1)
    last_dims_t = len(data_dash.shape) - 1
    attention_dims_t = len(data_dash.shape) - 3
    if is_query:
        data_dash = ratio * (
            torch.exp(
                data_dash
                - diag_data
                - torch.max(data_dash, dim=last_dims_t, keepdim=True)[0]
            )
            + numerical_stabilizer
        )
    else:
        data_dash = ratio * (
            torch.exp(
                data_dash
                - diag_data
                - torch.max(
                    torch.max(data_dash, dim=last_dims_t, keepdim=True)[0],
                    dim=attention_dims_t,
                    keepdim=True,
                )[0]
            )
            + numerical_stabilizer
        )
    return data_dash


def linear_kernel(
    x: torch.Tensor,
    node_vec1: torch.Tensor,
    node_vec2: torch.Tensor,
) -> torch.Tensor:
    node_vec1 = node_vec1.permute(1, 0, 2, 3)
    node_vec2 = node_vec2.permute(1, 0, 2, 3)
    x = x.permute(1, 0, 2, 3)

    v2x = torch.einsum("nbhm,nbhd->bhmd", node_vec2, x)
    out1 = torch.einsum("nbhm,bhmd->nbhd", node_vec1, v2x)

    one_matrix = torch.ones([node_vec2.shape[0]], device=node_vec1.device)
    node_vec2_sum = torch.einsum("nbhm,n->bhm", node_vec2, one_matrix)
    out2 = torch.einsum("nbhm,bhm->nbh", node_vec1, node_vec2_sum)

    out1 = out1.permute(1, 0, 2, 3)
    out2 = out2.permute(1, 0, 2)
    out2 = torch.unsqueeze(out2, len(out2.shape))
    return out1 / out2


def spatial_loss(
    node_vec1: torch.Tensor,
    node_vec2: torch.Tensor,
    supports: list[torch.Tensor],
    edge_indices: torch.Tensor,
) -> torch.Tensor:
    batch_size = node_vec1.size(0)
    node_vec1 = node_vec1.permute(1, 0, 2, 3)
    node_vec2 = node_vec2.permute(1, 0, 2, 3)

    node_vec1_end = node_vec1[edge_indices[:, 0]]
    node_vec2_start = node_vec2[edge_indices[:, 1]]
    attn1 = torch.einsum("ebhm,ebhm->ebh", node_vec1_end, node_vec2_start)
    attn1 = attn1.permute(1, 0, 2)

    one_matrix = torch.ones([node_vec2.shape[0]], device=node_vec1.device)
    node_vec2_sum = torch.einsum("nbhm,n->bhm", node_vec2, one_matrix)
    attn_norm = torch.einsum("nbhm,bhm->nbh", node_vec1, node_vec2_sum)

    attn2 = attn_norm[edge_indices[:, 0]]
    attn2 = attn2.permute(1, 0, 2)
    attn_score = (attn1 / attn2).clamp_min(1e-12)

    d_norm = supports[0][edge_indices[:, 0], edge_indices[:, 1]]
    d_norm = d_norm.reshape(1, -1, 1).repeat(batch_size, 1, attn_score.shape[-1])
    return torch.mean(attn_score.log() * d_norm)


class ConvApproximation(nn.Module):
    def __init__(self, dropout: float, tau: float, random_feature_dim: int) -> None:
        super().__init__()
        self.tau = tau
        self.random_feature_dim = random_feature_dim
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        node_vec1: torch.Tensor,
        node_vec2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dim = node_vec1.shape[-1]
        random_seed = int(
            torch.ceil(torch.abs(torch.sum(node_vec1.detach())) * 1e8)
            .cpu()
            .item()
        )
        random_seed = random_seed % (2**31 - 1)
        random_matrix = create_random_matrix(
            self.random_feature_dim,
            dim,
            seed=random_seed,
        ).to(node_vec1.device)

        node_vec1 = node_vec1 / math.sqrt(self.tau)
        node_vec2 = node_vec2 / math.sqrt(self.tau)
        node_vec1_prime = random_feature_map(node_vec1, True, random_matrix)
        node_vec2_prime = random_feature_map(node_vec2, False, random_matrix)

        x = linear_kernel(x, node_vec1_prime, node_vec2_prime)
        return x, node_vec1_prime, node_vec2_prime


class LinearizedConv(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hid_dim: int,
        dropout: float,
        tau: float = 1.0,
        random_feature_dim: int = 64,
    ) -> None:
        super().__init__()
        self.input_fc = nn.Conv2d(
            in_channels=in_dim,
            out_channels=hid_dim,
            kernel_size=(1, 1),
            bias=True,
        )
        self.output_fc = nn.Conv2d(
            in_channels=in_dim,
            out_channels=hid_dim,
            kernel_size=(1, 1),
            bias=True,
        )
        self.activation = nn.Sigmoid()
        self.dropout_layer = nn.Dropout(p=dropout)
        self.conv_app_layer = ConvApproximation(dropout, tau, random_feature_dim)

    def forward(
        self,
        input_data: torch.Tensor,
        node_vec1: torch.Tensor,
        node_vec2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.input_fc(input_data)
        x = self.activation(x) * self.output_fc(input_data)
        x = self.dropout_layer(x)

        x = x.permute(0, 2, 3, 1)
        x, node_vec1_prime, node_vec2_prime = self.conv_app_layer(
            x,
            node_vec1,
            node_vec2,
        )
        x = x.permute(0, 3, 1, 2)

        return x, node_vec1_prime, node_vec2_prime
