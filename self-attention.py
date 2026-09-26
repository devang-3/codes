import torch
import torch.nn as nn
from torchtype import TensorType


class SingleHeadAttention(nn.Module):
    """Single-head causal self-attention module.

    Implements the scaled dot-product attention mechanism with causal masking
    to prevent attending to future positions (autoregressive property).

    Args:
        embedding_dim: Dimensionality of the input embeddings.
        attention_dim: Dimensionality of the key, query, and value projections.
    """

    def __init__(self, embedding_dim: int, attention_dim: int):
        """Initialize the attention module with linear projections for K, Q, V.

        Args:
            embedding_dim: Dimensionality of the input embeddings.
            attention_dim: Dimensionality of the key, query, and value projections.
        """
        super().__init__()
        torch.manual_seed(42)

        # Initialize weights for key, query, and value projections
        self.wk = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.wq = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.wv = nn.Linear(embedding_dim, attention_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> tuple[TensorType[float], TensorType[float]]:
        """Compute causal self-attention over the input sequence.

        Args:
            embedded: Input tensor of shape (batch_size, seq_len, embedding_dim).

        Returns:
            A tuple containing:
                - weighted_values: Output tensor of shape (batch_size, seq_len, attention_dim).
                - attn_weights: Attention weights of shape (batch_size, seq_len, seq_len).
        """
        # Compute the key, query, and value representations
        q = self.wq(embedded)
        k = self.wk(embedded)
        v = self.wv(embedded)

        # Compute the attention scores (scaled dot-product)
        attn_score = q @ k.transpose(-2, -1) / (q.shape[-1] ** 0.5)

        # Apply a causal mask to prevent attending to future positions
        upper_triangular_mask = torch.triu(
            torch.ones(attn_score.shape[-2], attn_score.shape[-1]),
            diagonal=1
        ).bool()
        attn_score = attn_score.masked_fill(upper_triangular_mask, float('-inf'))

        # Apply softmax to get the attention weights
        attn_weights = torch.softmax(attn_score, dim=-1)

        weighted_values = attn_weights @ v
        return weighted_values, attn_weights