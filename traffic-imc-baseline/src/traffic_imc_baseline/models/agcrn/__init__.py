"""
AGCRN model components for traffic prediction.

AGCRN (Adaptive Graph Convolutional Recurrent Network) learns:
- Node embeddings that capture spatial relationships
- Adaptive adjacency matrices through node embedding similarity
- Temporal patterns through GRU-based recurrent units

Reference:
    Bai, L., et al. "Adaptive Graph Convolutional Recurrent Network 
    for Traffic Forecasting." NeurIPS 2020.
"""
from .agcrn import AGCRN, AVWDCRNN
from .layers import AVWGCN, AGCRNCell
from .module import AGCRNLightningModule

__all__ = [
    "AGCRN",
    "AVWDCRNN",
    "AVWGCN",
    "AGCRNCell",
    "AGCRNLightningModule",
]
