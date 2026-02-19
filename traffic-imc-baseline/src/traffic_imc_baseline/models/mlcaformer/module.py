from typing import Tuple

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ...utils import smape
from .mlcaformer import MLCAFormer


class MLCAFormerLightningModule(L.LightningModule):
    """PyTorch Lightning module for training MLCAFormer model."""

    def __init__(
        self,
        num_nodes: int,
        in_steps: int = 24,
        out_steps: int = 1,
        steps_per_day: int = 24,
        input_dim: int = 3,
        output_dim: int = 1,
        input_embedding_dim: int = 24,
        tod_embedding_dim: int = 24,
        dow_embedding_dim: int = 24,
        nid_embedding_dim: int = 24,
        col_embedding_dim: int = 80,
        feed_forward_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        scheduler_factor: float = 0.5,
        scheduler_patience: int = 10,
        scaler=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["scaler"])

        self.scaler = scaler

        self.model = MLCAFormer(
            num_nodes=num_nodes,
            in_steps=in_steps,
            out_steps=out_steps,
            steps_per_day=steps_per_day,
            input_dim=input_dim,
            output_dim=output_dim,
            input_embedding_dim=input_embedding_dim,
            tod_embedding_dim=tod_embedding_dim,
            dow_embedding_dim=dow_embedding_dim,
            nid_embedding_dim=nid_embedding_dim,
            col_embedding_dim=col_embedding_dim,
            feed_forward_dim=feed_forward_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.criterion = nn.MSELoss()
        self.learning_rate = learning_rate
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience

        self.validation_outputs: list[dict[str, np.ndarray | float]] = []
        self.test_outputs: list[dict[str, np.ndarray | float]] = []

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

    def _inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            return data

        original_shape = data.shape
        flat_data = data.reshape(-1, 1)
        unscaled = self.scaler.inverse_transform(flat_data)
        return unscaled.reshape(original_shape)

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
                "y_true": y.cpu().numpy(),
                "y_pred": y_hat.cpu().numpy(),
                "loss": loss.item(),
            }
        )

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
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
                "y_true": y.cpu().numpy(),
                "y_pred": y_hat.cpu().numpy(),
                "y_is_missing": y_is_missing.cpu().numpy(),
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
        smape_value = smape(y_true_eval.flatten(), y_pred_eval.flatten())

        self.log("val_mae", mae)
        self.log("val_rmse", rmse)
        self.log("val_smape", float(smape_value))

        self.validation_outputs.clear()

    def on_test_epoch_end(self):
        if not self.test_outputs:
            return

        y_true = np.concatenate([x["y_true"] for x in self.test_outputs], axis=0)
        y_pred = np.concatenate([x["y_pred"] for x in self.test_outputs], axis=0)
        y_is_missing = np.concatenate([x["y_is_missing"] for x in self.test_outputs], axis=0)

        y_is_missing_flat = y_is_missing.flatten()
        non_missing_mask = ~y_is_missing_flat

        if non_missing_mask.any():
            print(f"Non-missing ratio: {non_missing_mask.sum() / len(y_is_missing_flat) * 100:.2f}%")
        else:
            print("\nWarning: No non-missing test points available for metric calculation.")
            self.test_outputs.clear()
            return

        y_true_eval = self._inverse_transform(y_true)
        y_pred_eval = self._inverse_transform(y_pred)
        y_true_eval_flat = y_true_eval.flatten()
        y_pred_eval_flat = y_pred_eval.flatten()
        y_true_valid = y_true_eval_flat[non_missing_mask]
        y_pred_valid = y_pred_eval_flat[non_missing_mask]

        mae = mean_absolute_error(y_true_valid, y_pred_valid)
        rmse = np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))
        smape_value = smape(y_true_valid, y_pred_valid)

        self.log("test_mae", mae)
        self.log("test_rmse", rmse)
        self.log("test_smape", float(smape_value))

        print("\nTest Results (Original Scale - Non-Missing Data Only):")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"sMAPE: {smape_value:.4f}%")

        self.test_outputs.clear()
