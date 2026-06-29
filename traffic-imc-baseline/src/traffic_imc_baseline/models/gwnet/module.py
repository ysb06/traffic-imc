"""PyTorch Lightning module for Graph WaveNet."""

from typing import Tuple

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ...utils import compute_test_metrics_by_horizon
from .datamodule import build_adj_supports
from .model import GraphWaveNet


class GWNetLightningModule(L.LightningModule):
    """PyTorch Lightning module for training Graph WaveNet."""

    def __init__(
        self,
        adj_mx: np.ndarray,
        num_nodes: int,
        seq_len: int = 24,
        horizon: int = 24,
        input_dim: int = 2,
        output_dim: int = 1,
        dropout: float = 0.3,
        gcn_bool: bool = True,
        addaptadj: bool = True,
        randomadj: bool = True,
        apt_only: bool = False,
        adjtype: str = "doubletransition",
        residual_channels: int = 32,
        dilation_channels: int = 32,
        skip_channels: int = 256,
        end_channels: int = 512,
        kernel_size: int = 2,
        blocks: int = 4,
        layers: int = 2,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        scaler=None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["adj_mx", "scaler"])

        if output_dim != 1:
            raise ValueError("GWNetLightningModule currently supports output_dim=1 only.")

        self.scaler = scaler

        supports_np = build_adj_supports(adj_mx, adjtype)
        supports = [torch.tensor(i, dtype=torch.float32) for i in supports_np]
        aptinit = None if randomadj else supports[0]
        if apt_only:
            supports = None

        self.model = GraphWaveNet(
            num_nodes=num_nodes,
            dropout=dropout,
            supports=supports,
            gcn_bool=gcn_bool,
            addaptadj=addaptadj,
            aptinit=aptinit,
            in_dim=input_dim,
            out_dim=horizon,
            residual_channels=residual_channels,
            dilation_channels=dilation_channels,
            skip_channels=skip_channels,
            end_channels=end_channels,
            kernel_size=kernel_size,
            blocks=blocks,
            layers=layers,
        )

        self.criterion = nn.L1Loss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.validation_outputs: list[dict[str, np.ndarray | float]] = []
        self.test_outputs: list[dict[str, np.ndarray | float]] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            return data

        original_shape = data.shape
        flat_data = data.reshape(-1, 1)
        unscaled = self.scaler.inverse_transform(flat_data)
        return unscaled.reshape(original_shape)

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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return optimizer

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        x, y = batch

        y_hat = self(x)
        y_hat, y = self._normalize_outputs(y_hat, y)

        loss = self.criterion(y_hat, y)
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

        y_hat = self(x)
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

        y_hat = self(x)
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
