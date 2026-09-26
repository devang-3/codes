import torch
import torch.nn as nn
from torchtype import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(42)

        #Initialize weights 
        self.wk = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.wq = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.wv = nn.Linear(embedding_dim, attention_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # Compute the key, query, and value representations
        q = self.wq(embedded)
        k = self.wk(embedded)
        v = self.wv(embedded)

        # Compute the attention scores
        attn_score = q @ k.transpose(-2, -1) / (q.shape[-1] ** 0.5)

        # Apply a mask to the attention scores to prevent attending to future positions
        upper_triangular_mask = torch.triu(torch.ones(attn_score.shape[-2], attn_score.shape[-1]), diagonal=1).bool()
        attn_score = attn_score.masked_fill(upper_triangular_mask, float('-inf'))

        # Apply softmax to get the attention weights
        attn_weights = torch.softmax(attn_score, dim=-1)

        weighted_values = attn_weights @ v
        return weighted_values, attn_weights