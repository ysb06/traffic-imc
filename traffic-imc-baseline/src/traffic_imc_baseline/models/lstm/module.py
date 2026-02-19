from typing import List, Tuple

import lightning as L
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .model import LSTMBaseModel
from ...utils import smape

SimpleBatchType = Tuple[torch.Tensor, torch.Tensor]
SimpleWithMissingBatchType = Tuple[torch.Tensor, torch.Tensor, List[bool]]


class LSTMLightningModule(L.LightningModule):
    def __init__(
        self,
        scaler: MinMaxScaler,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        learning_rate: float = 0.001,
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
        self.criterion = nn.MSELoss()

        self.learning_rate = learning_rate
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
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
                "y_true": y.cpu().numpy(),
                "y_pred": y_hat.cpu().numpy(),
                "loss": loss.item(),
            }
        )

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss

    def test_step(
        self, batch: SimpleWithMissingBatchType, batch_idx: int
    ) -> torch.Tensor:
        x, y, y_is_missing_list = batch
        y_hat: torch.Tensor = self(x)

        if y.dim() > 2:
            y = y.squeeze(-1)

        loss: torch.Tensor = self.criterion(y_hat, y)

        self.test_outputs.append(
            {
                "y_true": y.cpu().numpy(),
                "y_pred": y_hat.cpu().numpy(),
                "loss": loss.item(),
                "is_missing": y_is_missing_list,
            }
        )

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
        smape_value = smape(y_true_eval.flatten(), y_pred_eval.flatten())

        self.log("val_mae", mae)
        self.log("val_rmse", rmse)
        self.log("val_smape", float(smape_value))

        self.validation_outputs.clear()

    def on_test_epoch_end(self):
        has_missing_info = "is_missing" in self.test_outputs[0]
        if has_missing_info:
            original_outputs = []
            for output in self.test_outputs:
                is_missing_list = output["is_missing"]
                y_true = output["y_true"]
                y_pred = output["y_pred"]

                original_mask = ~np.array(is_missing_list, dtype=bool)
                if original_mask.any():
                    original_outputs.append(
                        {
                            "y_true": y_true[original_mask],
                            "y_pred": y_pred[original_mask],
                        }
                    )

            if len(original_outputs) == 0:
                print("\nWarning: No original (non-interpolated) test data found!")
                self.test_outputs.clear()
                return

            y_true = np.concatenate([x["y_true"] for x in original_outputs], axis=0)
            y_pred = np.concatenate([x["y_pred"] for x in original_outputs], axis=0)

            total_samples = sum(
                len(output["is_missing"]) for output in self.test_outputs
            )
            original_samples = len(y_true)
            print(
                f"\nUsing {original_samples}/{total_samples} original samples "
                f"(excluding {total_samples - original_samples} interpolated values)"
            )
        else:
            y_true = np.concatenate([x["y_true"] for x in self.test_outputs], axis=0)
            y_pred = np.concatenate([x["y_pred"] for x in self.test_outputs], axis=0)

        y_true_eval = self._inverse_transform(y_true)
        y_pred_eval = self._inverse_transform(y_pred)

        mae = mean_absolute_error(y_true_eval.flatten(), y_pred_eval.flatten())
        rmse = np.sqrt(mean_squared_error(y_true_eval.flatten(), y_pred_eval.flatten()))
        smape_value = smape(y_true_eval.flatten(), y_pred_eval.flatten())

        self.log("test_mae", mae)
        self.log("test_rmse", rmse)
        self.log("test_smape", float(smape_value))

        print("\nTest Results (Original Scale - Non-Missing Data Only):")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"sMAPE: {smape_value:.2f}%")

        self.test_outputs.clear()
