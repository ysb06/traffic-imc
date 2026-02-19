import torch
import torch.nn as nn


class LSTMBaseModel(nn.Module):
    """
    Basic PyTorch LSTM model for traffic prediction.
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        dropout_rate: float = 0.2,
    ):
        super(LSTMBaseModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LSTM and fully connected layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_size)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size)
        """
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]  # Get the last time step output
        output = self.fc(last_output)  # Fully connected layer

        return output
