import torch
import torch.nn as nn
import math
from torch.nn import MultiheadAttention, Linear, LayerNorm
import torch.nn.functional as F

class MultiAttn(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, emb_dim, num_heads, dropout_rate):
        super().__init__()
        self.W_q = Linear(query_dim, emb_dim)
        self.W_k = Linear(key_dim, emb_dim)
        self.W_v = Linear(value_dim, emb_dim)
        self.multihead_attn = MultiheadAttention(emb_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.norm = LayerNorm(emb_dim)
        
    def forward(self, query, key, value, mask):
        q = self.W_q(query)
        k = self.W_k(key)
        v = self.W_v(value)
        attn_output, _ = self.multihead_attn(q, k, v, key_padding_mask=mask.bool(), need_weights=False)
        return self.norm(v + attn_output)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create a matrix of shape (max_len, d_model) to hold the positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Compute the positional encodings for each dimension
        div_term = torch.exp(torch.arange(0, d_model // 2 * 2, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0:-1:2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # If d_model is odd, the last dimension will only have a sine term
        if d_model % 2 == 1:
            pe[:, -1] = torch.sin(position * div_term[-1]).squeeze()
        
        # Add an extra dimension so it can be added to the embeddings
        pe = pe.unsqueeze(0)
        
        # Register the positional encoding as a buffer (not a parameter, but still part of the module)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to the input embeddings
        x = x + self.pe[:, :x.size(1)]
        return x