"""PyTorch Lightning module for DCRNN model training."""

import logging
from typing import Optional, Tuple

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ...utils import smape
from .dcrnn_model import DCRNNModel


class DCRNNLightningModule(L.LightningModule):
    """PyTorch Lightning module for training DCRNN model for traffic prediction."""

    def __init__(
        self,
        adj_mx: np.ndarray,
        num_nodes: int,
        input_dim: int = 2,
        output_dim: int = 1,
        seq_len: int = 24,
        horizon: int = 1,
        rnn_units: int = 64,
        num_rnn_layers: int = 2,
        max_diffusion_step: int = 2,
        filter_type: str = "dual_random_walk",
        use_curriculum_learning: bool = True,
        cl_decay_steps: int = 2000,
        learning_rate: float = 0.01,
        weight_decay: float = 0,
        scheduler_milestones: Optional[list[int]] = None,
        scheduler_gamma: float = 0.1,
        scaler=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["adj_mx", "scaler"])

        self.scaler = scaler
        self._logger = logging.getLogger(__name__)

        model_kwargs = {
            "num_nodes": num_nodes,
            "input_dim": input_dim,
            "output_dim": output_dim,
            "seq_len": seq_len,
            "horizon": horizon,
            "rnn_units": rnn_units,
            "num_rnn_layers": num_rnn_layers,
            "max_diffusion_step": max_diffusion_step,
            "filter_type": filter_type,
            "use_curriculum_learning": use_curriculum_learning,
            "cl_decay_steps": cl_decay_steps,
        }

        self.model = DCRNNModel(adj_mx, self._logger, **model_kwargs)

        self.criterion = nn.MSELoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_milestones = scheduler_milestones or [20, 30, 40, 50]
        self.scheduler_gamma = scheduler_gamma

        self.batches_seen = 0

        self.validation_outputs: list[dict[str, np.ndarray | float]] = []
        self.test_outputs: list[dict[str, np.ndarray | float]] = []

        # Initialize dynamically registered DCGRU parameters before optimizer setup.
        self._initialize_dynamic_parameters(
            num_nodes=num_nodes,
            input_dim=input_dim,
            output_dim=output_dim,
            seq_len=seq_len,
            horizon=horizon,
        )

    def _initialize_dynamic_parameters(
        self,
        num_nodes: int,
        input_dim: int,
        output_dim: int,
        seq_len: int,
        horizon: int,
    ) -> None:
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            dummy_inputs = torch.zeros(seq_len, 1, num_nodes * input_dim)
            _ = self.model(dummy_inputs, labels=None, batches_seen=1)
        if was_training:
            self.model.train()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            eps=1e-8,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.scheduler_milestones,
            gamma=self.scheduler_gamma,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def forward(
        self,
        inputs: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        batches_seen: Optional[int] = None,
    ) -> torch.Tensor:
        return self.model(inputs, labels, batches_seen)

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
        y_hat = self(x, y, self.batches_seen)

        self.batches_seen += 1

        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)

        loss = self.criterion(y_hat, y)

        self.validation_outputs.append(
            {
                "y_true": y.permute(1, 0, 2).cpu().numpy(),
                "y_pred": y_hat.permute(1, 0, 2).cpu().numpy(),
                "loss": loss.item(),
            }
        )

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        x, y, y_is_missing = batch
        y_hat = self(x)

        loss = self.criterion(y_hat, y)

        self.test_outputs.append(
            {
                "y_true": y.permute(1, 0, 2).cpu().numpy(),
                "y_pred": y_hat.permute(1, 0, 2).cpu().numpy(),
                "y_is_missing": y_is_missing.permute(1, 0, 2).cpu().numpy(),
                "loss": loss.item(),
            }
        )

        self.log("test_loss", loss, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        if not self.validation_outputs:
            return

        y_true = np.concatenate(
            [x["y_true"] for x in self.validation_outputs], axis=0
        )
        y_pred = np.concatenate(
            [x["y_pred"] for x in self.validation_outputs], axis=0
        )

        y_true_eval = self._inverse_transform(y_true)
        y_pred_eval = self._inverse_transform(y_pred)

        mae = mean_absolute_error(y_true_eval.flatten(), y_pred_eval.flatten())
        rmse = np.sqrt(mean_squared_error(y_true_eval.flatten(), y_pred_eval.flatten()))
        smape_value = smape(y_true_eval.flatten(), y_pred_eval.flatten())

        self.log("val_mae", mae)
        self.log("val_rmse", rmse)
        self.log("val_smape", float(smape_value))

        for horizon_idx, horizon_name in [(2, "3"), (5, "6"), (11, "12")]:
            if y_true_eval.shape[1] > horizon_idx:
                y_true_h = y_true_eval[:, horizon_idx, :]
                y_pred_h = y_pred_eval[:, horizon_idx, :]
                mae_h = mean_absolute_error(y_true_h.flatten(), y_pred_h.flatten())
                self.log(f"val_mae_horizon{horizon_name}", mae_h)

        self.validation_outputs.clear()

    def on_test_epoch_end(self):
        if not self.test_outputs:
            return

        y_true = np.concatenate([x["y_true"] for x in self.test_outputs], axis=0)
        y_pred = np.concatenate([x["y_pred"] for x in self.test_outputs], axis=0)
        y_is_missing = np.concatenate(
            [x["y_is_missing"] for x in self.test_outputs],
            axis=0,
        )

        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        y_is_missing_flat = y_is_missing.flatten()
        non_missing_mask = ~y_is_missing_flat

        total_points = len(y_true_flat)
        non_missing_points = int(non_missing_mask.sum())
        missing_points = total_points - non_missing_points

        print("\nTest Data Statistics:")
        print(f"  Total points: {total_points}")
        print(
            "  Non-missing (original) points: "
            f"{non_missing_points} ({non_missing_points/total_points*100:.1f}%)"
        )
        print(
            "  Missing (interpolated) points: "
            f"{missing_points} ({missing_points/total_points*100:.1f}%)"
        )

        if non_missing_points == 0:
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

        print("\nTest Results (Original Scale - Non-Missing Only):")
        print(f"  MAE:   {mae:.4f}")
        print(f"  RMSE:  {rmse:.4f}")
        print(f"  sMAPE: {smape_value:.2f}%")

        self.test_outputs.clear()
