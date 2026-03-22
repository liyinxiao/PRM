import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_model // n_head

        self.W_Q = nn.Linear(d_model, d_model, bias=True)
        self.W_K = nn.Linear(d_model, d_model, bias=True)
        self.W_V = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, key_mask=None):
        """
        queries: (B, T_q, d_model)
        keys:    (B, T_k, d_model)
        key_mask: (B, T_k) — 1 for valid, 0 for padding
        """
        B, T_q, _ = queries.size()
        T_k = keys.size(1)

        Q = self.W_Q(queries)  # (B, T_q, d_model)
        K = self.W_K(keys)     # (B, T_k, d_model)
        V = self.W_V(keys)     # (B, T_k, d_model)

        # Split into heads: (B*h, T, d_k)
        Q = Q.view(B, T_q, self.n_head, self.d_k).permute(0, 2, 1, 3).reshape(B * self.n_head, T_q, self.d_k)
        K = K.view(B, T_k, self.n_head, self.d_k).permute(0, 2, 1, 3).reshape(B * self.n_head, T_k, self.d_k)
        V = V.view(B, T_k, self.n_head, self.d_k).permute(0, 2, 1, 3).reshape(B * self.n_head, T_k, self.d_k)

        # Scaled dot-product attention
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.d_k)  # (B*h, T_q, T_k)

        # Key masking (mask padding positions in keys)
        if key_mask is not None:
            # key_mask: (B, T_k) -> (B*h, 1, T_k)
            key_mask_expanded = key_mask.unsqueeze(1).repeat(self.n_head, 1, 1)  # (B*h, 1, T_k)
            scores = scores.masked_fill(key_mask_expanded == 0, -1e9)

        attn = F.softmax(scores, dim=-1)  # (B*h, T_q, T_k)

        # Query masking: zero out attention for padding query positions
        if key_mask is not None:
            # Use the same mask for queries (self-attention, so T_q == T_k)
            query_mask = key_mask.unsqueeze(2).repeat(self.n_head, 1, 1)  # (B*h, T_q, 1)
            attn = attn * query_mask

        attn = self.dropout(attn)
        output = torch.bmm(attn, V)  # (B*h, T_q, d_k)

        # Concat heads
        output = output.view(B, self.n_head, T_q, self.d_k).permute(0, 2, 1, 3).reshape(B, T_q, self.d_model)
        return output


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_inner, dropout=0.1):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(d_model)
        self.conv1 = nn.Conv1d(d_model, d_inner, 1)
        self.conv2 = nn.Conv1d(d_inner, d_model, 1)
        self.bn2 = nn.BatchNorm1d(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: (B, T, d_model)"""
        residual = x
        # BatchNorm over feature dim: (B, T, C) -> (B, C, T)
        out = self.bn1(x.transpose(1, 2)).transpose(1, 2)
        # Conv1d: (B, C, T)
        out = out.transpose(1, 2)
        out = F.relu(self.conv1(out))
        out = self.conv2(out)
        out = out.transpose(1, 2)  # back to (B, T, C)
        out = self.dropout(out)
        out = out + residual
        out = self.bn2(out.transpose(1, 2)).transpose(1, 2)
        return out


class PRMModel(nn.Module):
    def __init__(self, feature_size, eb_dim, max_time_len,
                 itm_spar_num, itm_dens_num,
                 d_model=64, d_inner_hid=128, n_head=1, n_layers=4, dropout=0.2):
        super().__init__()
        self.max_time_len = max_time_len
        self.itm_spar_num = itm_spar_num
        self.itm_dens_num = itm_dens_num
        self.d_model = d_model
        self.eb_dim = eb_dim

        # Feature dimension after embedding concat + dense
        self.ft_num = itm_spar_num * eb_dim + itm_dens_num

        # Embedding table: +1 to reserve index 0 for padding
        self.emb_mtx = nn.Embedding(feature_size + 1, eb_dim, padding_idx=0)
        nn.init.trunc_normal_(self.emb_mtx.weight, std=0.05)
        with torch.no_grad():
            self.emb_mtx.weight[0].zero_()  # re-zero padding row after init

        # Learnable position embedding
        pos_dim = self.ft_num
        self.pos_mtx = nn.Parameter(torch.randn(max_time_len, pos_dim) * 0.05)

        # Project input features to d_model
        self.input_proj = nn.Linear(self.ft_num, d_model, bias=True)

        # Stacked Transformer encoder blocks
        self.n_layers = n_layers
        self.attention_layers = nn.ModuleList(
            [MultiHeadAttention(d_model, n_head, dropout) for _ in range(n_layers)])
        self.ffn_layers = nn.ModuleList(
            [PositionwiseFeedForward(d_model, d_inner_hid, dropout) for _ in range(n_layers)])

        # Output head: BN -> FC(d_model, d_model, ReLU) -> Dropout -> FC(d_model, 1)
        self.out_bn = nn.BatchNorm1d(d_model)
        self.out_fc1 = nn.Linear(d_model, d_model)
        self.out_fc2 = nn.Linear(d_model, 1)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, itm_spar, itm_dens, seq_length):
        """
        itm_spar: (B, max_time_len, itm_spar_num) — int sparse feature indices
        itm_dens: (B, max_time_len, itm_dens_num) — float dense features
        seq_length: (B,) — actual list length for each sample

        Returns: y_pred (B, max_time_len) — softmax scores masked by seq_length
        """
        B = itm_spar.size(0)

        # Embedding lookup for sparse features
        spar_emb = self.emb_mtx(itm_spar)  # (B, T, itm_spar_num, eb_dim)
        spar_emb = spar_emb.view(B, self.max_time_len, self.itm_spar_num * self.eb_dim)

        # Concat sparse embeddings with dense features
        item_seq = torch.cat([spar_emb, itm_dens], dim=-1)  # (B, T, ft_num)

        # Add position embedding
        item_seq = item_seq + self.pos_mtx.unsqueeze(0)  # (B, T, ft_num)

        # Create key/query mask from seq_length
        # mask: (B, T) with 1 for valid positions, 0 for padding
        positions = torch.arange(self.max_time_len, device=itm_spar.device).unsqueeze(0)  # (1, T)
        key_mask = (positions < seq_length.unsqueeze(1)).float()  # (B, T)

        # Project to d_model
        item_seq = self.input_proj(item_seq)  # (B, T, d_model)

        # Stacked encoder blocks
        for attn, ffn in zip(self.attention_layers, self.ffn_layers):
            item_seq = attn(item_seq, item_seq, key_mask)
            item_seq = ffn(item_seq)

        # Apply sequence mask before output head
        mask_3d = key_mask.unsqueeze(-1)  # (B, T, 1)
        seq_rep = item_seq * mask_3d

        # Output head
        y_pred = self._output_head(seq_rep, key_mask)
        return y_pred

    def _output_head(self, seq_rep, seq_mask):
        """
        seq_rep: (B, T, d_model) — masked sequence representation
        seq_mask: (B, T) — 1 for valid, 0 for padding

        Returns: (B, T) softmax scores masked by seq_mask
        """
        B, T, C = seq_rep.size()
        # BatchNorm: (B, T, C) -> (B*T, C) -> bn -> (B, T, C)
        out = seq_rep.reshape(B * T, C)
        out = self.out_bn(out)
        out = out.view(B, T, C)

        out = F.relu(self.out_fc1(out))
        out = self.out_dropout(out)
        out = self.out_fc2(out).squeeze(-1)  # (B, T)

        # Softmax over list positions
        score = F.softmax(out, dim=-1)

        # Mask padding positions
        y_pred = score * seq_mask
        return y_pred
