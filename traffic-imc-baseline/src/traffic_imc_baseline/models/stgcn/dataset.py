from typing import Optional, Protocol, runtime_checkable
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

class STGCNDatasetWithMissing(Dataset):
    """PyTorch Dataset for STGCN model with missing mask support.
    
    Args:
        data: Traffic data array of shape (time_steps, n_vertex)
        n_his: Number of historical time steps
        n_pred: Number of prediction time steps ahead
        missing_mask: Optional boolean array of shape (time_steps, n_vertex)
                     True indicates the value was originally missing (interpolated)
        
    Returns:
        x: Input tensor of shape (in_channels, n_his, n_vertex)
        y: Target tensor of shape (n_vertex,)
        y_is_missing: Boolean tensor of shape (n_vertex,) - True if originally missing
    """
    
    def __init__(
        self,
        data: np.ndarray,
        n_his: int,
        n_pred: int,
        missing_mask: Optional[np.ndarray] = None,
    ):
        self.n_his = n_his
        self.n_pred = n_pred
        self.missing_mask = missing_mask
        
        self.x, self.y, self.y_missing = self._data_transform(data, n_his, n_pred, missing_mask)
        self._is_scaled = False
        
        assert len(self.x) == len(self.y), "x and y must have the same length"
        if len(self.x) == 0:
            raise ValueError("All samples contained NaNs and were filtered out. Check your data.")
    
    def apply_scaler(self, scaler: MinMaxScaler) -> None:
        """Apply scaling to the dataset using a fitted scaler."""
        if self._is_scaled:
            return
        
        if len(self.x) == 0:
            self._is_scaled = True
            return
        
        # Scale x: shape (num_samples, in_channels, n_his, n_vertex)
        x_shape = self.x.shape
        x_flat = self.x.numpy().reshape(-1, 1)
        x_scaled = scaler.transform(x_flat)
        self.x = torch.tensor(x_scaled.reshape(x_shape), dtype=torch.float32)
        
        # Scale y: shape (num_samples, n_vertex)
        y_shape = self.y.shape
        y_flat = self.y.numpy().reshape(-1, 1)
        y_scaled = scaler.transform(y_flat)
        self.y = torch.tensor(y_scaled.reshape(y_shape), dtype=torch.float32)
        
        self._is_scaled = True
    
    def __len__(self) -> int:
        return len(self.x)
    
    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx], self.y_missing[idx]
    
    def _data_transform(
        self,
        data: np.ndarray,
        n_his: int,
        n_pred: int,
        missing_mask: Optional[np.ndarray],
    ):
        n_vertex = data.shape[1]
        l = len(data)
        num = l - n_his - n_pred + 1

        x_list, y_list, y_missing_list = [], [], []
        filtered_count = 0

        for i in range(num):
            head = i
            tail = i + n_his
            
            x_window = data[head:tail, :]  # (n_his, n_vertex)
            y_window = data[tail + n_pred - 1]  # (n_vertex,)

            # 결측치 존재 여부 확인
            if np.isnan(x_window).any() or np.isnan(y_window).any():
                filtered_count += 1
                continue

            x_list.append(torch.tensor(x_window).unsqueeze(0))  # (1, n_his, n_vertex)
            y_list.append(torch.tensor(y_window))
            
            # Missing mask 처리
            if missing_mask is not None:
                y_missing = missing_mask[tail + n_pred - 1]  # (n_vertex,)
                y_missing_list.append(torch.tensor(y_missing, dtype=torch.bool))
            else:
                # missing_mask가 없으면 모두 False (missing 없음)
                y_missing_list.append(torch.zeros(n_vertex, dtype=torch.bool))

        x = torch.stack(x_list) if x_list else torch.empty(0)
        y = torch.stack(y_list) if y_list else torch.empty(0)
        y_missing = torch.stack(y_missing_list) if y_missing_list else torch.empty(0, dtype=torch.bool)
        
        if num > 0:
            retention_rate = len(x_list) / num * 100
            print(f"Dataset filtering: {len(x_list)}/{num} samples retained ({retention_rate:.1f}%), {filtered_count} filtered due to NaNs")

        return x, y, y_missing