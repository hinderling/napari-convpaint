import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import einsum


class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.norm_q = nn.RMSNorm(query_dim)
        self.norm_k = nn.RMSNorm(key_dim)
        self.norm_v = nn.RMSNorm(value_dim)
        # Keep nn.MultiheadAttention as the parameter store so that
        # existing checkpoints load without any state-dict changes.
        self.attention = nn.MultiheadAttention(
            embed_dim=query_dim,
            vdim=value_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )

    def forward(self, query, key, value):
        query = self.norm_q(query)
        key = self.norm_k(key)

        attn_scores = self._compute_attn_scores(query, key)
        attn_output = einsum("b i j, b j d -> b i d", attn_scores, value)

        return attn_output, attn_scores

    def _compute_attn_scores(self, query, key):
        """Compute head-averaged attention scores using the Q/K projection
        weights stored in self.attention (nn.MultiheadAttention).

        This replaces the call to nn.MultiheadAttention.forward() which
        crashes on MPS due to a PyTorch Metal backend bug.  Only the Q and K
        projections are needed — V projection and out_proj are skipped because
        the caller discards the MHA output and recomputes it via einsum.
        """
        mha = self.attention
        embed_dim = mha.embed_dim

        # --- biases (stored as a single [3*embed_dim] tensor) ---
        if mha.in_proj_bias is not None:
            q_bias = mha.in_proj_bias[:embed_dim]
            k_bias = mha.in_proj_bias[embed_dim:2 * embed_dim]
        else:
            q_bias = k_bias = None

        # --- project Q and K ---
        q = F.linear(query, mha.q_proj_weight, q_bias)   # (B, N_q, embed_dim)
        k = F.linear(key,   mha.k_proj_weight, k_bias)   # (B, N_k, embed_dim)

        # --- reshape to (B, num_heads, N, head_dim) ---
        B, N_q, _ = q.shape
        N_k = k.shape[1]
        q = q.view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)

        # --- scaled dot-product attention scores ---
        scale = self.head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, N_q, N_k)
        attn_weights = F.softmax(scores, dim=-1)

        # --- average over heads ---
        return attn_weights.mean(dim=1)  # (B, N_q, N_k)


class CrossAttentionBlock(nn.Module):

    def __init__(self, query_dim, key_dim, value_dim, num_heads, **kwargs):
        super().__init__()

        self.cross_attn = CrossAttention(
            query_dim,
            key_dim,
            value_dim,
            num_heads,
        )
        self.conv2d = nn.Conv2d(query_dim, query_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, q, k, v, **kwargs):
        q = self.conv2d(q)
        q = rearrange(q, "b c h w -> b (h w) c")
        k = rearrange(k, "b c h w -> b (h w) c")
        v = rearrange(v, "b c h w -> b (h w) c")
        features, _ = self.cross_attn(q, k, v)

        return features
