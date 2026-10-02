import math
import torch
import torch.nn as nn
from torch import Tensor

class PrepareForMultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, heads: int, d_k: int, bias: bool):
        super().__init__()

        # Linear layer for Linear transform
        self.linear = nn.Linear(d_model, heads * d_k, bias=bias)
        # No. of heads
        self.heads = heads
        # Dimensionality of each head
        self.d_k = d_k

    def forward(self, x: torch.Tensor):

        head_shape = x.shape[:-1]
        # apply linear transformation
        x = self.linear(x)
        # split last dimension into heads
        x = x.view(*head_shape, self.heads, self.d_k)
        # output
        return x

class MultiHeadAttention(nn.Module):
    """Compute multi-head scaled dot-product attention over a sequence.

    This module projects the input query, key, and value tensors into multiple
    attention heads, computes pairwise attention scores within each head, and then
    combines the attended values back into the model dimension. It is intended for
    sequence-to-sequence or self-attention use cases.

    Args:
        heads: Number of attention heads to split the representation into.
        d_model: Size of the input embedding dimension for each token.
        dropout_prob: Dropout probability applied after the softmax attention weights.
        bias: Whether to include bias terms in the linear projections and output layer.
    """

    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1, bias: bool = True):
        super().__init__()
        self.d_k = d_model // heads
        self.heads = heads

        self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias)
        self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias)
        self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias)

        self.softmax = nn.Softmax(dim=1)
        self.output = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout_prob)
        self.scale = 1 / math.sqrt(self.d_k)
        self.attn = None

    def get_scores(self, query: torch.Tensor, key: torch.Tensor):

        return torch.einsum('ibhd,jbhd->ijbh', query, key)

    def prepare_mask(self, mask:torch.Tensor, query_shape: List[int], key_shape: List[int]):
        assert mask.shape[0] == 1 or mask.shape[0] == query_shape[0]
        assert mask.shape[1] == key_shape[0]
        assert mask.shape[2] == 1 or mask.shape[2] == query_shape[1]

        mask = mask.unsqueeze(-1)

        return mask

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None):

        seq_len, batch_size, _ = query.shape

        if mask is not None:
            mask = self.prepare_mask(mask, query.shape, key.shape)

        # prepare query, key, value for multi-head attention
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)
        scores = self.get_scores(query, key)
        scores = scores * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = self.softmax(scores)
        attn = self.dropout(attn)

        # Apply attention to values
        x = torch.einsum('ijbh,jbhd->ibhd', attn, value)
        x = x.contiguous()
        x = x.view(seq_len, batch_size, -1)
        x = self.output(x)

        return x
