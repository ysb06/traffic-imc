"""PyTorch Lightning module for AGCRN."""

from typing import Tuple

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ...utils import compute_test_metrics_by_horizon
from .agcrn import AGCRN


class AGCRNLightningModule(L.LightningModule):
    """PyTorch Lightning module for training AGCRN model."""

    def __init__(
        self,
        num_nodes: int,
        in_steps: int = 24,
        out_steps: int = 24,
        input_dim: int = 1,
        output_dim: int = 1,
        rnn_units: int = 64,
        num_layers: int = 2,
        embed_dim: int = 10,
        cheb_k: int = 2,
        learning_rate: float = 0.003,
        weight_decay: float = 0.0,
        adam_epsilon: float = 1.0e-8,
        scaler=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["scaler"])

        self.scaler = scaler

        self.model = AGCRN(
            num_nodes=num_nodes,
            input_dim=input_dim,
            output_dim=output_dim,
            horizon=out_steps,
            rnn_units=rnn_units,
            num_layers=num_layers,
            embed_dim=embed_dim,
            cheb_k=cheb_k,
        )

        self.criterion = nn.L1Loss()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.adam_epsilon = adam_epsilon

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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            eps=self.adam_epsilon,
        )
        return optimizer

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        x, y = batch

        y_hat = self(x)
        if y_hat.dim() == 4 and y_hat.size(-1) == 1:
            y_hat = y_hat.squeeze(-1)
        if y.dim() == 4 and y.size(-1) == 1:
            y = y.squeeze(-1)

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
        if y_hat.dim() == 4 and y_hat.size(-1) == 1:
            y_hat = y_hat.squeeze(-1)
        if y.dim() == 4 and y.size(-1) == 1:
            y = y.squeeze(-1)

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
        if y_hat.dim() == 4 and y_hat.size(-1) == 1:
            y_hat = y_hat.squeeze(-1)
        if y.dim() == 4 and y.size(-1) == 1:
            y = y.squeeze(-1)

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

    def on_validation_epoch_end(self):
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

    def on_test_epoch_end(self):
        if not self.test_outputs:
            return

        y_true = np.concatenate([x["y_true"] for x in self.test_outputs], axis=0)
        y_pred = np.concatenate([x["y_pred"] for x in self.test_outputs], axis=0)
        y_is_missing = np.concatenate([x["y_is_missing"] for x in self.test_outputs], axis=0)

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
