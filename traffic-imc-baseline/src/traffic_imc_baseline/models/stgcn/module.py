import torch
import torch.nn as nn
import lightning as L
from typing import Literal, Tuple
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from .model import BaseSTGCN
from ...utils import smape


class STGCNLightningModule(L.LightningModule):
    """
    PyTorch Lightning module for training STGCN model for traffic prediction
    """

    def __init__(
        self,
        gso: torch.Tensor,
        learning_rate: float = 0.001,
        scheduler_factor: float = 0.95,
        scheduler_patience: int = 10,
        dropout_rate: float = 0.5,
        n_his: int = 24,
        Kt: int = 3,
        stblock_num: int = 2,
        Ks: int = 3,
        act_func: Literal["glu", "gtu", "relu", "silu"] = "glu",
        graph_conv_type: Literal["cheb_graph_conv", "graph_conv"] = "graph_conv",
        enable_bias: bool = True,
        scaler=None,
    ):
        """
        Args:
            n_vertex: Number of nodes in the graph
            gso: Graph shift operator (adjacency matrix or Laplacian)
            input_size: Input feature size (default: 1)
            hidden_size: Hidden layer size (default: 64)
            num_layers: Number of layers (default: 2)
            output_size: Output feature size (default: 1)
            learning_rate: Learning rate (default: 0.001)
            dropout_rate: Dropout rate (default: 0.5)
            scheduler_factor: LR scheduler factor (default: 0.95)
            scheduler_patience: LR scheduler patience (default: 10)
            n_his: Historical time steps (default: 24)
            Kt: Kernel size of temporal convolution (default: 3)
            stblock_num: Number of ST-Conv blocks (default: 2)
            Ks: Kernel size of spatial convolution (default: 3)
            act_func: Activation function ('glu' or 'gtu', default: 'glu')
            graph_conv_type: Graph convolution type ('cheb_graph_conv' or 'graph_conv', default: 'graph_conv')
            enable_bias: Enable bias in layers (default: True)
            scaler: MinMaxScaler instance for inverse transform (optional, default: None)
        """
        super().__init__()
        self.save_hyperparameters(ignore=["gso", "scaler"])
        
        # Store scaler for inverse transform (not saved in checkpoints)
        self.scaler = scaler

        # Initialize the STGCN model
        self.model = BaseSTGCN(
            n_vertex=gso.size(0),
            gso=gso,
            dropout_rate=dropout_rate,
            n_his=n_his,
            Kt=Kt,
            stblock_num=stblock_num,
            Ks=Ks,
            act_func=act_func,
            graph_conv_type=graph_conv_type,
            enable_bias=enable_bias,
        )

        # Loss function declaration
        self.criterion = nn.MSELoss()

        # Learning rate and scheduler parameters
        self.learning_rate = learning_rate
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience

        # Metrics storage
        self.validation_outputs = []
        self.test_outputs = []

    def configure_optimizers(self):
        """Configure optimizer with StepLR scheduler"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.scheduler_patience,
            gamma=self.scheduler_factor,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def _inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform scaled data back to original scale.
        
        Args:
            data: Scaled data of shape (batch, n_vertex)
            
        Returns:
            Unscaled data with the same shape
        """
        if self.scaler is None:
            return data
        
        original_shape = data.shape
        flat_data = data.reshape(-1, 1)
        unscaled = self.scaler.inverse_transform(flat_data)
        return unscaled.reshape(original_shape)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step"""
        x, y = batch  # Simple collate function
        y_hat: torch.Tensor = self(x)

        # Reshape model output from (batch, 1, 1, n_vertex) to (batch, n_vertex)
        y_hat = y_hat.squeeze(1).squeeze(1)

        loss: torch.Tensor = self.criterion(y_hat, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True)

        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Validation step"""
        x, y = batch
        y_hat: torch.Tensor = self(x)

        # Reshape model output from (batch, 1, 1, n_vertex) to (batch, n_vertex)
        y_hat = y_hat.squeeze(1).squeeze(1)

        loss: torch.Tensor = self.criterion(y_hat, y)

        # Store outputs for epoch-end metrics
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
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Test step"""
        x, y, missing = batch  # Simple collate function
        y_hat: torch.Tensor = self(x)

        # Reshape model output from (batch, 1, 1, n_vertex) to (batch, n_vertex)
        y_hat = y_hat.squeeze(1).squeeze(1)

        loss: torch.Tensor = self.criterion(y_hat, y)

        # Store outputs for epoch-end metrics (including missing mask)
        self.test_outputs.append(
            {
                "y_true": y.cpu().numpy(),
                "y_pred": y_hat.cpu().numpy(),
                "loss": loss.item(),
                "is_missing": missing.cpu().numpy(),
            }
        )

        self.log("test_loss", loss, on_step=False, on_epoch=True)

        return loss

    def on_validation_epoch_end(self):
        """Calculate validation metrics at the end of epoch"""
        if len(self.validation_outputs) == 0:
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

        # Clear outputs for next epoch
        self.validation_outputs.clear()

    def on_test_epoch_end(self):
        """Calculate test metrics at the end of epoch (excluding interpolated data)"""
        if len(self.test_outputs) == 0:
            return

        # Concatenate all predictions, targets, and missing masks
        y_true = np.concatenate([x["y_true"] for x in self.test_outputs], axis=0)
        y_pred = np.concatenate([x["y_pred"] for x in self.test_outputs], axis=0)
        is_missing = np.concatenate([x["is_missing"] for x in self.test_outputs], axis=0)

        # Flatten all arrays
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        is_missing_flat = is_missing.flatten()

        # Create mask for valid (non-interpolated) data points
        valid_mask = ~is_missing_flat  # True = original data (not interpolated)
        
        # Count statistics
        total_points = len(y_true_flat)
        valid_points = valid_mask.sum()
        interpolated_points = total_points - valid_points
        
        print(f"\nTest Data Statistics:")
        print(f"  Total points: {total_points}")
        print(f"  Valid (original) points: {valid_points} ({valid_points/total_points*100:.1f}%)")
        print(f"  Interpolated points (excluded): {interpolated_points} ({interpolated_points/total_points*100:.1f}%)")

        if valid_points == 0:
            print("\nWarning: No non-missing test points available for metric calculation.")
            self.test_outputs.clear()
            return

        y_true_eval = self._inverse_transform(y_true)
        y_pred_eval = self._inverse_transform(y_pred)
        y_true_eval_flat = y_true_eval.flatten()
        y_pred_eval_flat = y_pred_eval.flatten()
        y_true_eval_valid = y_true_eval_flat[valid_mask]
        y_pred_eval_valid = y_pred_eval_flat[valid_mask]

        mae = mean_absolute_error(y_true_eval_valid, y_pred_eval_valid)
        rmse = np.sqrt(mean_squared_error(y_true_eval_valid, y_pred_eval_valid))
        smape_value = smape(y_true_eval_valid, y_pred_eval_valid)

        self.log("test_mae", mae)
        self.log("test_rmse", rmse)
        self.log("test_smape", float(smape_value))

        print(f"\nTest Results (Original Scale, Excluding Interpolated):")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  sMAPE: {smape_value:.4f}%")

        # Clear outputs
        self.test_outputs.clear()
