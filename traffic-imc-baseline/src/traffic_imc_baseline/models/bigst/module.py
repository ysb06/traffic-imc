"""PyTorch Lightning module for BigST."""

from typing import Tuple

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ...utils import compute_test_metrics_by_horizon
from .datamodule import build_bigst_supports
from .model import BigST


class BigSTLightningModule(L.LightningModule):
    """PyTorch Lightning module for training BigST."""

    def __init__(
        self,
        adj_mx: np.ndarray,
        num_nodes: int,
        input_length: int = 24,
        output_length: int = 24,
        input_dim: int = 3,
        output_dim: int = 1,
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
        spatial_loss_weight: float = 0.3,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        adam_epsilon: float = 1.0e-8,
        milestones: tuple[int, ...] = (80, 100),
        gamma: float = 0.1,
        scaler=None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["adj_mx", "scaler"])

        if output_dim != 1:
            raise ValueError("BigSTLightningModule currently supports output_dim=1 only.")
        if input_dim != 3:
            raise ValueError("BigSTLightningModule expects input_dim=3.")

        adj_array = np.asarray(adj_mx, dtype=np.float32)
        if adj_array.shape != (num_nodes, num_nodes):
            raise ValueError(
                "adj_mx shape must match num_nodes: "
                f"{adj_array.shape} != ({num_nodes}, {num_nodes})"
            )

        self.scaler = scaler
        self.criterion = nn.L1Loss()
        self.use_spatial = use_spatial
        self.spatial_loss_weight = spatial_loss_weight
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.adam_epsilon = adam_epsilon
        self.milestones = milestones
        self.gamma = gamma

        if use_spatial:
            supports_np = build_bigst_supports(adj_array)
            supports = [torch.tensor(i, dtype=torch.float32) for i in supports_np]
            edge_indices = torch.nonzero(supports[0] > 0, as_tuple=False)
        else:
            supports = None
            edge_indices = None

        self.model = BigST(
            num_nodes=num_nodes,
            input_length=input_length,
            output_length=output_length,
            input_dim=input_dim,
            hid_dim=hid_dim,
            num_layers=num_layers,
            tau=tau,
            random_feature_dim=random_feature_dim,
            node_dim=node_dim,
            time_dim=time_dim,
            time_num=time_num,
            week_num=week_num,
            dropout=dropout,
            use_residual=use_residual,
            use_bn=use_bn,
            use_spatial=use_spatial,
            supports=supports,
            edge_indices=edge_indices,
        )

        self.validation_outputs: list[dict[str, np.ndarray | float]] = []
        self.test_outputs: list[dict[str, np.ndarray | float]] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_hat, _ = self._predict_with_aux_loss(x)
        return y_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            eps=self.adam_epsilon,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(self.milestones),
            gamma=self.gamma,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        x, y = batch
        y_hat, spatial_loss = self._predict_with_aux_loss(x)
        y_hat, y = self._normalize_outputs(y_hat, y)

        mae_loss = self.criterion(y_hat, y)
        loss = mae_loss
        if self.use_spatial:
            loss = mae_loss - self.spatial_loss_weight * spatial_loss
            self.log(
                "train_spatial_loss",
                spatial_loss,
                on_step=True,
                on_epoch=True,
                batch_size=x.size(0),
            )

        self.log(
            "train_mae_loss",
            mae_loss,
            on_step=True,
            on_epoch=True,
            batch_size=x.size(0),
        )
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=x.size(0),
        )
        return loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        x, y = batch
        y_hat, _ = self._predict_with_aux_loss(x)
        y_hat, y = self._normalize_outputs(y_hat, y)
        loss = self.criterion(y_hat, y)

        self.validation_outputs.append(
            {
                "y_true": y.detach().cpu().numpy(),
                "y_pred": y_hat.detach().cpu().numpy(),
                "loss": loss.item(),
            }
        )

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=x.size(0),
        )
        return loss

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        x, y, y_is_missing = batch
        y_hat, _ = self._predict_with_aux_loss(x)
        y_hat, y = self._normalize_outputs(y_hat, y)
        loss = self.criterion(y_hat, y)

        self.test_outputs.append(
            {
                "y_true": y.detach().cpu().numpy(),
                "y_pred": y_hat.detach().cpu().numpy(),
                "y_is_missing": y_is_missing.detach().cpu().numpy(),
                "loss": loss.item(),
            }
        )

        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=x.size(0),
        )
        return loss

    def on_validation_epoch_end(self) -> None:
        if not self.validation_outputs:
            return

        y_true = np.concatenate([x["y_true"] for x in self.validation_outputs], axis=0)
        y_pred = np.concatenate([x["y_pred"] for x in self.validation_outputs], axis=0)

        y_true_eval = self._inverse_transform(y_true)
        y_pred_eval = self._inverse_transform(y_pred)
        mae = mean_absolute_error(y_true_eval.flatten(), y_pred_eval.flatten())
        rmse = np.sqrt(mean_squared_error(y_true_eval.flatten(), y_pred_eval.flatten()))
        self.log("val_mae", mae, prog_bar=True)
        self.log("val_rmse", rmse)
        self.validation_outputs.clear()

    def on_test_epoch_end(self) -> None:
        if not self.test_outputs:
            return

        y_true = np.concatenate([x["y_true"] for x in self.test_outputs], axis=0)
        y_pred = np.concatenate([x["y_pred"] for x in self.test_outputs], axis=0)
        y_is_missing = np.concatenate(
            [x["y_is_missing"] for x in self.test_outputs],
            axis=0,
        )

        y_true_eval = self._inverse_transform(y_true)
        y_pred_eval = self._inverse_transform(y_pred)
        metrics = compute_test_metrics_by_horizon(
            y_true_eval,
            y_pred_eval,
            missing_mask=y_is_missing,
        )

        for name, value in metrics.items():
            self.log(name, value)

        if "test_mae" not in metrics:
            print("\nWarning: No non-missing test points available for metric calculation.")
            self.test_outputs.clear()
            return

        valid_points = int(metrics["test_valid_points"])
        missing_points = int(metrics["test_missing_points"])
        total_points = valid_points + missing_points
        print(
            "\nTest Data Statistics:"
            f"\n  Non-missing points: {valid_points}/{total_points} "
            f"({valid_points / total_points * 100:.2f}%)"
        )

        print("\nTest Results (Original Scale - Non-Missing Data Only):")
        print(f"  MAE:   {metrics['test_mae']:.4f}")
        print(f"  RMSE:  {metrics['test_rmse']:.4f}")
        print("\nKey Horizon Test Results:")
        for horizon in (1, 3, 6, 12, 24):
            suffix = f"h{horizon:02d}"
            mae_key = f"test_mae_{suffix}"
            if mae_key not in metrics:
                continue
            print(
                f"  {suffix}: MAE={metrics[mae_key]:.4f}, "
                f"RMSE={metrics[f'test_rmse_{suffix}']:.4f}"
            )

        self.test_outputs.clear()

    def _predict_with_aux_loss(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model_input = x.permute(0, 2, 1, 3).contiguous()
        y_hat, spatial_loss = self.model(model_input)
        y_hat = y_hat.permute(0, 2, 1).contiguous()
        return y_hat, spatial_loss

    def _normalize_outputs(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if y_hat.dim() == 4 and y_hat.size(-1) == 1:
            y_hat = y_hat.squeeze(-1)
        if y.dim() == 4 and y.size(-1) == 1:
            y = y.squeeze(-1)
        return y_hat, y

    def _inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            return data

        original_shape = data.shape
        flat_data = data.reshape(-1, 1)
        unscaled = self.scaler.inverse_transform(flat_data)
        return unscaled.reshape(original_shape)
