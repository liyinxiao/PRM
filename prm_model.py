import torch
import torch.nn as nn
import torch.nn.functional as F


class PRMModel(nn.Module):
    def __init__(self, feature_size, eb_dim, max_time_len,
                 itm_spar_num, itm_dens_num,
                 d_model=64, d_inner_hid=128, n_head=1, n_layers=4, dropout=0.2):
        super().__init__()
        self.max_time_len = max_time_len
        self.itm_spar_num = itm_spar_num
        self.itm_dens_num = itm_dens_num
        self.eb_dim = eb_dim
        self.d_model = d_model
        self.ft_num = itm_spar_num * eb_dim + itm_dens_num

        # Embedding table: +1 to reserve index 0 for padding
        self.emb_mtx = nn.Embedding(feature_size + 1, eb_dim, padding_idx=0)
        nn.init.trunc_normal_(self.emb_mtx.weight, std=0.05)
        with torch.no_grad():
            self.emb_mtx.weight[0].zero_()  # re-zero padding row after init

        # Learnable position embedding
        self.pos_mtx = nn.Parameter(torch.randn(max_time_len, self.ft_num) * 0.05)

        # Project input features to d_model
        self.input_proj = nn.Linear(self.ft_num, d_model)

        # Transformer encoder (handles multi-head attention + FFN + residual + norm)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_inner_hid,
            dropout=dropout,
            activation='relu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output head: single linear projection
        self.out_head = nn.Linear(d_model, 1)

    def forward(self, itm_spar, itm_dens, seq_length):
        """
        itm_spar: (B, T, itm_spar_num) int
        itm_dens: (B, T, itm_dens_num) float
        seq_length: (B,) int
        Returns: (B, T) softmax scores masked by seq_length
        """
        B, T, _ = itm_spar.shape

        # Sparse embedding → concat with dense
        spar_emb = self.emb_mtx(itm_spar).view(B, T, -1)       # (B, T, spar_num * eb_dim)
        item_seq = torch.cat([spar_emb, itm_dens], dim=-1)      # (B, T, ft_num)

        # Add position embedding
        item_seq = item_seq + self.pos_mtx

        # Project to d_model
        item_seq = self.input_proj(item_seq)                     # (B, T, d_model)

        # Padding mask: True = ignore (PyTorch convention)
        pad_mask = torch.arange(T, device=itm_spar.device).unsqueeze(0) >= seq_length.unsqueeze(1)  # (B, T)

        # Transformer encoder
        item_seq = self.encoder(item_seq, src_key_padding_mask=pad_mask)  # (B, T, d_model)

        # Output head
        out = self.out_head(item_seq).squeeze(-1)  # (B, T)

        # Softmax over positions, then mask padding
        score = F.softmax(out, dim=-1)
        seq_mask = ~pad_mask  # True = valid
        return score * seq_mask.float()
