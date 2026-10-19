import torch
from torch import nn
from pretrain_trfm import PositionalEncoding

class Prop2SmilesModel(nn.Module):
    def __init__(self,
                 prop_dim=3,
                 latent_dim=128,
                 vocab_size=5000,
                 hidden_size=256,
                 n_head=4,
                 n_dec_layers=4,
                 seq_len=220,
                 prop_recon_weight=1.0,
                 dropout=0.1):
        super().__init__()
        # 属性自编码
        self.prop_encoder = nn.Sequential(
            nn.Linear(prop_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        self.prop_decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, prop_dim)
        )
        # 投影到 memory
        self.memory_len = seq_len // 10
        self.mem_proj = nn.Linear(latent_dim, self.memory_len * hidden_size)
        # Decoder
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.pos_enc   = PositionalEncoding(hidden_size, dropout, max_len=seq_len)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=n_head,
            dim_feedforward=hidden_size*4,
            dropout=dropout)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_dec_layers,
                                             norm=nn.LayerNorm(hidden_size))
        self.out_proj = nn.Linear(hidden_size, vocab_size)
        # 损失
        self.ce = nn.CrossEntropyLoss(ignore_index=0)
        self.mse= nn.MSELoss()
        self.w  = prop_recon_weight

    def forward(self, props, seq):
        B, L = seq.shape
        # ——— 属性自编码损失 ———
        z = self.prop_encoder(props)                  # [B,latent_dim]
        rp= self.prop_decoder(z)                      # [B,3]
        Lp= self.mse(rp, props)
        # ——— memory ———
        m = self.mem_proj(z).view(B, self.memory_len, -1)  # [B,mem, H]
        mem = m.permute(1,0,2)                        # [mem,B,H]
        # ——— decoder 输入 ———
        emb = self.embedding(seq).permute(1,0,2)      # [L,B,H]
        emb = self.pos_enc(emb)
        mask= nn.Transformer.generate_square_subsequent_mask(L).to(seq.device)
        out = self.decoder(tgt=emb, memory=mem, tgt_mask=mask)  # [L,B,H]
        logits = self.out_proj(out).permute(1,2,0)     # [B,V,L]
        # ——— 交叉熵损失 ———
        tgt = seq[:,1:].contiguous()                  # 右移标签
        logits= logits[:,:,:L-1]
        Ls = self.ce(logits, tgt)
        return Ls + self.w*Lp, Ls, Lp
