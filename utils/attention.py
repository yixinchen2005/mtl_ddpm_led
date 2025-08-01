import torch
import torch.nn as nn
import torch.nn.functional as F

class RelativePositionBias(nn.Module):
    def __init__(self, num_heads, max_len=512):
        super().__init__()
        self.num_heads = num_heads
        self.max_len = max_len
        self.rel_bias = nn.Embedding(2 * max_len - 1, num_heads)

    def forward(self, seq_len):
        """
        Returns: [num_heads, seq_len, seq_len]
        """
        assert seq_len <= self.max_len, f"Sequence length {seq_len} exceeds max_len {self.max_len}"
        pos = torch.arange(seq_len, device=self.rel_bias.weight.device)
        rel_pos = pos[None, :] - pos[:, None]  # [seq_len, seq_len]
        rel_pos = rel_pos.clamp(min=-self.max_len + 1, max=self.max_len - 1) + self.max_len - 1
        bias = self.rel_bias(rel_pos)  # [seq_len, seq_len, num_heads]
        return bias.permute(2, 0, 1)  # [num_heads, seq_len, seq_len]

class UnifiedAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, emb_dim, num_heads, dropout_rate, use_relative=False, max_len=512):
        super().__init__()
        self.use_relative = use_relative
        self.num_heads = num_heads
        self.emb_dim = emb_dim
        self.head_dim = emb_dim // num_heads
        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"

        self.W_q = nn.Linear(query_dim, emb_dim)
        self.W_k = nn.Linear(key_dim, emb_dim)
        self.W_v = nn.Linear(value_dim, emb_dim)
        self.out_proj = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(emb_dim)

        if use_relative:
            self.rel_bias = RelativePositionBias(num_heads, max_len)

    def forward(self, query, key, value, mask=None):
        """
        Inputs:
            query: [batch, seq_len, query_dim]
            key:   [batch, seq_len, key_dim]
            value: [batch, seq_len, value_dim]
            mask:  [batch, seq_len] (optional key padding mask)
        Returns: [batch, seq_len, emb_dim]
        """
        B, Lq, _ = query.size()
        Lk = key.size(1)

        q = self.W_q(query).view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, Lq, d]
        k = self.W_k(key).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)    # [B, H, Lk, d]
        v = self.W_v(value).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, Lk, d]

        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, H, Lq, Lk]

        if self.use_relative and Lq == Lk:
            # Self-attention case with relative position bias
            bias = self.rel_bias(Lq)  # [H, Lq, Lq]
            attn_scores += bias.unsqueeze(0)  # [1, H, Lq, Lq]

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask[:, None, None, :] == 0, float('-inf'))

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        out = attn_probs @ v  # [B, H, Lq, d]
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.emb_dim)

        return self.norm(query + self.out_proj(out))