import torch
import torch.nn as nn
import lightning as L
from typing import Literal, Tuple
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from .model import BaseSTGCN
from ...utils import compute_test_metrics_by_horizon


class STGCNLightningModule(L.LightningModule):
    """
    PyTorch Lightning module for training STGCN model for traffic prediction
    """

    def __init__(
        self,
        gso: torch.Tensor,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        rmsprop_alpha: float = 0.9,
        rmsprop_epsilon: float = 1.0e-10,
        lr_decay_step: int = 5,
        lr_decay_rate: float = 0.7,
        dropout_rate: float = 0.5,
        n_his: int = 24,
        n_pred: int = 24,
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
            weight_decay: RMSprop weight decay (default: 0.0)
            rmsprop_alpha: RMSprop alpha (default: 0.9)
            rmsprop_epsilon: RMSprop epsilon (default: 1e-10)
            lr_decay_step: StepLR step size in epochs (default: 5)
            lr_decay_rate: StepLR decay rate (default: 0.7)
            n_his: Historical time steps (default: 24)
            n_pred: Prediction time steps (default: 24)
            Kt: Kernel size of temporal convolution (default: 3)
            stblock_num: Number of ST-Conv blocks (default: 2)
            Ks: Kernel size of spatial convolution (default: 3)
            act_func: Activation function ('glu' or 'gtu', default: 'glu')
            graph_conv_type: Graph convolution type ('cheb_graph_conv' or 'graph_conv', default: 'graph_conv')
            enable_bias: Enable bias in layers (default: True)
            scaler: StandardScaler instance for inverse transform (optional, default: None)
        """
        super().__init__()
        self.save_hyperparameters(ignore=["gso", "scaler"])
        
        # Store scaler for inverse transform (not saved in checkpoints)
        self.scaler = scaler
        self.n_pred = n_pred

        # Initialize the STGCN model
        self.model = BaseSTGCN(
            n_vertex=gso.size(0),
            gso=gso,
            dropout_rate=dropout_rate,
            n_his=n_his,
            n_pred=n_pred,
            Kt=Kt,
            stblock_num=stblock_num,
            Ks=Ks,
            act_func=act_func,
            graph_conv_type=graph_conv_type,
            enable_bias=enable_bias,
        )

        # Loss function declaration
        self.criterion = nn.L1Loss()

        # Learning rate and scheduler parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.rmsprop_alpha = rmsprop_alpha
        self.rmsprop_epsilon = rmsprop_epsilon
        self.lr_decay_step = lr_decay_step
        self.lr_decay_rate = lr_decay_rate

        # Metrics storage
        self.validation_outputs = []
        self.test_outputs = []

    def configure_optimizers(self):
        """Configure optimizer and epoch scheduler."""
        optimizer = torch.optim.RMSprop(
            self.parameters(),
            lr=self.learning_rate,
            alpha=self.rmsprop_alpha,
            eps=self.rmsprop_epsilon,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.lr_decay_step,
            gamma=self.lr_decay_rate,
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

    def _format_prediction(self, y_hat: torch.Tensor) -> torch.Tensor:
        """Convert STGCN output to (batch, n_pred, n_vertex)."""
        if y_hat.dim() == 4 and y_hat.size(2) == 1:
            y_hat = y_hat.squeeze(2)

        if y_hat.dim() != 3:
            raise ValueError(
                "Expected STGCN output shape (batch, n_pred, n_vertex), "
                f"got {tuple(y_hat.shape)}."
            )

        if y_hat.size(1) != self.n_pred:
            raise ValueError(
                f"Expected {self.n_pred} prediction steps, got {y_hat.size(1)}."
            )

        return y_hat
    
    def _inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform scaled data back to original scale.
        
        Args:
            data: Scaled data of shape (batch, n_pred, n_vertex)
            
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
        y_hat = self._format_prediction(y_hat)

        loss: torch.Tensor = self.criterion(y_hat, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True)

        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Validation step"""
        x, y = batch
        y_hat: torch.Tensor = self(x)
        y_hat = self._format_prediction(y_hat)

        loss: torch.Tensor = self.criterion(y_hat, y)

        # Store outputs for epoch-end metrics
        self.validation_outputs.append(
            {
                "y_true": y.detach().cpu().numpy(),
                "y_pred": y_hat.detach().cpu().numpy(),
                "loss": loss.item(),
            }
        )

        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Test step"""
        x, y, missing = batch  # Simple collate function
        y_hat: torch.Tensor = self(x)
        y_hat = self._format_prediction(y_hat)

        loss: torch.Tensor = self.criterion(y_hat, y)

        # Store outputs for epoch-end metrics (including missing mask)
        self.test_outputs.append(
            {
                "y_true": y.detach().cpu().numpy(),
                "y_pred": y_hat.detach().cpu().numpy(),
                "loss": loss.item(),
                "is_missing": missing.detach().cpu().numpy(),
            }
        )

        self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

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
        self.log("val_mae", mae, sync_dist=True)
        self.log("val_rmse", rmse, sync_dist=True)
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

        y_true_eval = self._inverse_transform(y_true)
        y_pred_eval = self._inverse_transform(y_pred)
        metrics = compute_test_metrics_by_horizon(
            y_true_eval,
            y_pred_eval,
            missing_mask=is_missing,
        )

        for name, value in metrics.items():
            self.log(name, value, sync_dist=True)

        if "test_mae" not in metrics:
            print("\nWarning: No non-missing test points available for metric calculation.")
            self.test_outputs.clear()
            return

        valid_points = int(metrics["test_valid_points"])
        interpolated_points = int(metrics["test_missing_points"])
        total_points = valid_points + interpolated_points
        print("\nTest Data Statistics:")
        print(f"  Total points: {total_points}")
        print(
            f"  Valid (original) points: {valid_points} "
            f"({valid_points / total_points * 100:.1f}%)"
        )
        print(
            f"  Interpolated points (excluded): {interpolated_points} "
            f"({interpolated_points / total_points * 100:.1f}%)"
        )

        print(f"\nTest Results (Original Scale, Excluding Interpolated):")
        print(f"  MAE:  {metrics['test_mae']:.4f}")
        print(f"  RMSE: {metrics['test_rmse']:.4f}")
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

        # Clear outputs
        self.test_outputs.clear()
