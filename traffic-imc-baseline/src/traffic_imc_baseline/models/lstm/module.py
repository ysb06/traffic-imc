from typing import Tuple

import lightning as L
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .model import LSTMBaseModel
from ...utils import compute_test_metrics_by_horizon

SimpleBatchType = Tuple[torch.Tensor, torch.Tensor]
SimpleWithMissingBatchType = (
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
)


class LSTMLightningModule(L.LightningModule):
    def __init__(
        self,
        scaler: StandardScaler,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 24,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        dropout_rate: float = 0.2,
        scheduler_factor: float = 0.5,
        scheduler_patience: int = 10,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["scaler"])

        self.scaler = scaler

        self.model = LSTMBaseModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout_rate=dropout_rate,
        )
        self.criterion = nn.L1Loss()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience

        self.validation_outputs = []
        self.test_outputs = []

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
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: SimpleBatchType, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat: torch.Tensor = self(x)

        if y.dim() > 2:
            y = y.squeeze(-1)

        loss: torch.Tensor = self.criterion(y_hat, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch: SimpleBatchType, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat: torch.Tensor = self(x)

        if y.dim() > 2:
            y = y.squeeze(-1)

        loss: torch.Tensor = self.criterion(y_hat, y)

        self.validation_outputs.append(
            {
                "y_true": y.detach().cpu().numpy(),
                "y_pred": y_hat.detach().cpu().numpy(),
                "loss": loss.item(),
            }
        )

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss

    def test_step(
        self, batch: SimpleWithMissingBatchType, batch_idx: int
    ) -> torch.Tensor:
        if len(batch) == 4:
            x, y, y_is_missing_list, sensor_idx = batch
        else:
            x, y, y_is_missing_list = batch
            sensor_idx = None
        y_hat: torch.Tensor = self(x)

        if y.dim() > 2:
            y = y.squeeze(-1)

        loss: torch.Tensor = self.criterion(y_hat, y)

        output = {
            "y_true": y.detach().cpu().numpy(),
            "y_pred": y_hat.detach().cpu().numpy(),
            "loss": loss.item(),
            "is_missing": y_is_missing_list.detach().cpu().numpy(),
        }
        if sensor_idx is not None:
            output["sensor_idx"] = sensor_idx.detach().cpu().numpy()
        self.test_outputs.append(output)

        self.log("test_loss", loss, on_step=False, on_epoch=True)
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
        self.log("val_mae", mae)
        self.log("val_rmse", rmse)
        self.validation_outputs.clear()

    def on_test_epoch_end(self):
        if not self.test_outputs:
            return

        y_true = np.concatenate([x["y_true"] for x in self.test_outputs], axis=0)
        y_pred = np.concatenate([x["y_pred"] for x in self.test_outputs], axis=0)
        if "is_missing" in self.test_outputs[0]:
            is_missing = np.concatenate(
                [x["is_missing"] for x in self.test_outputs],
                axis=0,
            ).astype(bool)
        else:
            is_missing = None

        y_true_eval = self._inverse_transform(y_true)
        y_pred_eval = self._inverse_transform(y_pred)
        metrics = compute_test_metrics_by_horizon(
            y_true_eval,
            y_pred_eval,
            missing_mask=is_missing,
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
            f"\nUsing {valid_points}/{total_points} original samples "
            f"(excluding {missing_points} interpolated values)"
        )

        print("\nTest Results (Original Scale - Non-Missing Data Only):")
        print(f"MAE: {metrics['test_mae']:.4f}")
        print(f"RMSE: {metrics['test_rmse']:.4f}")
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
