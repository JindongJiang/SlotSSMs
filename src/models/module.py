import torch
from torch import nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_model, num_heads, dropout=0., inverted=False, bias=True, 
                 norm_over_input=True, epsilon=1e-5, d_model_hidden=None):
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.inverted = inverted
        self.norm_over_input = norm_over_input

        self.epsilon = epsilon
        
        self.attn_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

        if d_model_hidden is None:
            d_model_hidden = d_model
        
        self.proj_q = nn.Linear(d_model, d_model_hidden, bias=bias)
        self.proj_k = nn.Linear(d_model, d_model_hidden, bias=bias)
        self.proj_v = nn.Linear(d_model, d_model_hidden, bias=bias)
        self.proj_o = nn.Linear(d_model_hidden, d_model, bias=bias)
    
    def forward(self, q, k, v, attn_mask=None, attn_bias=None, output_attentions=False):
        """
        q: batch_size x target_len x d_model
        k: batch_size x source_len x d_model
        v: batch_size x source_len x d_model·
        attn_mask: target_len x source_len
        return: batch_size x target_len x d_model
        """
        B, T, _ = q.shape
        _, S, _ = k.shape
        
        q = self.proj_q(q).view(B, T, self.num_heads, -1).transpose(1, 2)
        k = self.proj_k(k).view(B, S, self.num_heads, -1).transpose(1, 2)
        v = self.proj_v(v).view(B, S, self.num_heads, -1).transpose(1, 2)
        
        q = q * (q.shape[-1] ** (-0.5))
        attn = torch.matmul(q, k.transpose(-1, -2))

        if attn_bias is not None:
            attn = attn + attn_bias

        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, float('-inf'))

        if self.inverted:
            attn = F.softmax(attn.flatten(start_dim=1, end_dim=2), dim=1).reshape(B, self.num_heads, T, S)
            attn_vis = attn.detach()

            if attn_mask is not None:
                attn = attn.masked_fill(attn_mask, float('0.'))

            # attn /= attn.sum(dim=-1, keepdim=True) + self.epsilon
            if self.norm_over_input:
                attn = attn / (attn.sum(dim=-1, keepdim=True) + self.epsilon)
        else:
            attn = F.softmax(attn, dim=-1)
            attn_vis = attn.detach()

        attn = self.attn_dropout(attn)
        
        output = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, -1)
        output = self.proj_o(output)
        output = self.output_dropout(output)

        outputs = (output, attn_vis) if output_attentions else (output,)
        return outputs