
import torch.nn as nn

from .trm import *


class _MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, n_heads, dropout):
        super(_MultiHeadAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.n_heads = n_heads

        self.w_q = Linear(d_model, d_k * n_heads)
        self.w_k = Linear(d_model, d_k * n_heads)
        self.w_v = Linear(d_model, d_v * n_heads)

    def forward(self, q, k, v):
        # q: [b_size x len_q x d_model]
        # k: [b_size x len_k x d_model]
        # v: [b_size x len_k x d_model]
        b_size = q.size(0)

        # q_s: [b_size x n_heads x len_q x d_k]
        # k_s: [b_size x n_heads x len_k x d_k]
        # v_s: [b_size x n_heads x len_k x d_v]
        q_s = self.w_q(q).view(b_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.w_k(k).view(b_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.w_v(v).view(b_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        return q_s, k_s, v_s

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PoswiseFeedForwardNet, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNormalization(d_model)

    def forward(self, inputs):
        # inputs: [b_size x len_q x d_model]
        residual = inputs
        output = self.relu(self.conv1(inputs.transpose(1, 2)))

        # outputs: [b_size x len_q x d_model]
        output = self.conv2(output).transpose(1, 2)
        output = self.dropout(output)

        return self.layer_norm(residual + output)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_v, n_heads, dropout, d_model, visual_len, sen_len, fea_v, fea_s, pos):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.multihead_attn_v = _MultiHeadAttention(d_k, d_v, d_model, n_heads, dropout)
        self.multihead_attn_s = _MultiHeadAttention(d_k, d_v, d_model, n_heads, dropout)
        self.pos_emb_v = PosEncoding(visual_len * 10, d_model)
        self.pos_emb_s = PosEncoding(sen_len * 10, d_model)
        self.linear_v = nn.Linear(in_features=fea_v, out_features=d_model)
        self.linear_s = nn.Linear(in_features=fea_s, out_features=d_model)
        self.proj_v = Linear(n_heads * d_v, d_model)
        self.proj_s = Linear(n_heads * d_v, d_model)
        self.d_v = d_v
        self.dropout = nn.Dropout(dropout)
        self.layer_norm_v = LayerNormalization(d_model)
        self.layer_norm_s = LayerNormalization(d_model)
        self.attention = ScaledDotProductAttention(d_k, dropout)
        self.pos = pos

    def forward(self, v, s, v_len, s_len):
        b_size = v.size(0)
        # q: [b_size x len_q x d_model]
        # k: [b_size x len_k x d_model]
        # v: [b_size x len_v x d_model] note (len_k == len_v)
        v, s = self.linear_v(v), self.linear_s(s)
        if self.pos:
            pos_v, pos_s = self.pos_emb_v(v_len), self.pos_emb_s(s_len)
            residual_v, residual_s = v + pos_v, s + pos_s
        else:
            residual_v, residual_s = v, s
        # context: a tensor of shape [b_size x len_q x n_heads * d_v]
        q_v, k_v, v_v = self.multihead_attn_v(v, v, v)
        q_s, k_s, v_s = self.multihead_attn_s(s, s, s)
        context_v, attn_v = self.attention(q_v, k_s, v_s)
        context_s, attn_s = self.attention(q_s, k_v, v_v)
        context_v = context_v.transpose(1, 2).contiguous().view(b_size, -1, self.n_heads * self.d_v)
        context_s = context_s.transpose(1, 2).contiguous().view(b_size, -1, self.n_heads * self.d_v)
        # project back to the residual size, outputs: [b_size x len_q x d_model]
        output_v = self.dropout(self.proj_v(context_v))
        output_s = self.dropout(self.proj_s(context_s))
        return self.layer_norm_v(residual_v + output_v), self.layer_norm_s(residual_s + output_s)

class co_attention(nn.Module):
    def __init__(self, d_k, d_v, n_heads, dropout, d_model, visual_len, sen_len, fea_v, fea_s, pos):
        super(co_attention, self).__init__()
        # self.layer_num = layer_num
        # self.multi_head = MultiHeadAttention(d_k=d_k, d_v=d_v, n_heads=n_heads, dropout=dropout, d_model=d_model,
        #                                      visual_len=visual_len, sen_len=sen_len, fea_v=fea_v, fea_s=fea_s, pos=False)
        # self.PoswiseFeedForwardNet_v = nn.ModuleList([PoswiseFeedForwardNet(d_model=d_model, d_ff=256)])
        # self.PoswiseFeedForwardNet_s = nn.ModuleList([PoswiseFeedForwardNet(d_model=d_model, d_ff=256)])
        # self.multi_head = nn.ModuleList([MultiHeadAttention(d_k=d_k, d_v=d_v, n_heads=n_heads, dropout=dropout, d_model=d_model,
        #                                      visual_len=visual_len, sen_len=sen_len, fea_v=fea_v, fea_s=fea_s, pos=False)])
        # for i in range(1, layer_num):
        #     self.PoswiseFeedForwardNet_v.append(PoswiseFeedForwardNet(d_model=d_model, d_ff=256))
        #     self.PoswiseFeedForwardNet_s.append(PoswiseFeedForwardNet(d_model=d_model, d_ff=256))
        #     self.multi_head.append(MultiHeadAttention(d_k=d_k, d_v=d_v, n_heads=n_heads, dropout=dropout, d_model=d_model,
        #                                      visual_len=visual_len, sen_len=sen_len, fea_v=d_model, fea_s=d_model, pos=True))
        self.multi_head = MultiHeadAttention(d_k=d_k, d_v=d_v, n_heads=n_heads, dropout=dropout, d_model=d_model,
                                             visual_len=visual_len, sen_len=sen_len, fea_v=fea_v, fea_s=fea_s, pos=pos)
        self.PoswiseFeedForwardNet_v = PoswiseFeedForwardNet(d_model=d_model, d_ff=128, dropout=dropout)
        self.PoswiseFeedForwardNet_s = PoswiseFeedForwardNet(d_model=d_model, d_ff=128,dropout=dropout)
    def forward(self, v, s, v_len, s_len):
        # for i in range(self.layer_num):
        #     v, s = self.multi_head[i](v, s, v_len, s_len)
        #     v = self.PoswiseFeedForwardNet_v[i](v)
        #     s = self.PoswiseFeedForwardNet_s[i](s)
        v, s = self.multi_head(v, s, v_len, s_len)
        v = self.PoswiseFeedForwardNet_v(v)
        s = self.PoswiseFeedForwardNet_s(s)
        return v, s
