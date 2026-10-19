import os
import json
import joblib
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from scripts.build_vocab import WordVocab,Vocab
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from scripts.utils import split, validity, sample   
from scripts.pretrain_trfm import PositionalEncoding
import pickle

class Prop2SmilesDataset(Dataset):
    def __init__(self, csv_path, smiles_col, prop_cols, vocab_path, seq_len=220):
        df = pd.read_csv(csv_path)
        self.props = torch.tensor(df[prop_cols].values.astype(float), dtype=torch.float)
        self.smiles = df[smiles_col].astype(str).tolist()
        self.seq_len = seq_len
        with open(vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)
    def __len__(self):
        return len(self.props)
    def __getitem__(self, idx):
        prop = self.props[idx]
        sm = self.smiles[idx]
        toks = split(sm).split()
        ids = [self.vocab.stoi.get(t, self.vocab.unk_index) for t in toks]
        seq = [self.vocab.sos_index] + ids + [self.vocab.eos_index]
        if len(seq) < self.seq_len:
            seq += [self.vocab.pad_index] * (self.seq_len - len(seq))
        else:
            seq = seq[:self.seq_len]
        return prop, torch.tensor(seq, dtype=torch.long)

class Prop2SmilesModel(nn.Module):
    def __init__(self, prop_dim=3, latent_dim=128, vocab_size=5000,
                 hidden_size=256, n_head=4, n_dec_layers=4,
                 seq_len=220, prop_recon_weight=0.1, dropout=0.1):
        super().__init__()
        self.prop_encoder = nn.Sequential(
            nn.Linear(prop_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        self.prop_decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, prop_dim)
        )
        self.memory_len = seq_len // 10
        self.mem_proj = nn.Linear(latent_dim, self.memory_len * hidden_size)
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.pos_enc = PositionalEncoding(hidden_size, dropout, max_len=seq_len)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size, nhead=n_head,
            dim_feedforward=hidden_size*4, dropout=dropout)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_dec_layers,
                                             norm=nn.LayerNorm(hidden_size))
        self.out_proj = nn.Linear(hidden_size, vocab_size)
        self.ce = nn.CrossEntropyLoss(ignore_index=0)
        self.mse= nn.MSELoss()
        self.w  = prop_recon_weight
    def forward(self, props, seq):
        B, L = seq.shape
        z = self.prop_encoder(props)
        rp = self.prop_decoder(z)
        Lp = self.mse(rp, props)
        m = self.mem_proj(z).view(B, self.memory_len, -1)
        mem = m.permute(1,0,2)
        emb = self.embedding(seq).permute(1,0,2)
        emb = self.pos_enc(emb)
        mask = nn.Transformer.generate_square_subsequent_mask(L).to(seq.device)
        out = self.decoder(tgt=emb, memory=mem, tgt_mask=mask)
        logits = self.out_proj(out).permute(1,2,0)
        tgt = seq[:,1:].contiguous()
        logits = logits[:,:,:L-1]
        Ls = self.ce(logits, tgt)
        return Ls + self.w*Lp, Ls, Lp

# 实例化测试
ds = Prop2SmilesDataset(r'D:\workspace\pycharm\smiles-transformer\data\WPT,AJ,ZYM\datasets\train\data_scaled.csv', 'processed_smiles', ['lam_abs','lam_em','eps'], r'D:\workspace\pycharm\smiles-transformer\data\WPT,AJ,ZYM\datasets\train\chem_data\vocab.pkl')
print('Dataset size:', len(ds))


model = Prop2SmilesModel(
    prop_dim=3, latent_dim=128, vocab_size=len(ds.vocab),
    hidden_size=256, n_head=4, n_dec_layers=4,
    seq_len=ds.seq_len, prop_recon_weight=0.1
).to('cuda' if torch.cuda.is_available() else 'cpu')
print(model)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
scaler = joblib.load(r'D:\workspace\pycharm\smiles-transformer\data\WPT,AJ,ZYM\datasets\train\prop_scaler.pkl')
with open(r'D:\workspace\pycharm\smiles-transformer\data\WPT,AJ,ZYM\datasets\train\chem_data\vocab.pkl','rb') as f:
    vocab = pickle.load(f)
    vovab_size = len(vocab)
print("词表大小:", len(vocab))
print("示例 token:", list(vocab.stoi.keys())[:10])
print("特殊标记:", vocab.pad_index, vocab.sos_index, vocab.eos_index, vocab.unk_index)

model.load_state_dict(torch.load(r'D:\workspace\pycharm\smiles-transformer\output\proc2smiles\proc2smiles_data_chem.pt'))
model.eval()

def generate_smiles(raw_prop, max_len=300):
    prop_scaled = scaler.transform([raw_prop])[0]
    z = model.prop_encoder(torch.tensor(prop_scaled, dtype=torch.float).to(device)[None])
    m = model.mem_proj(z).view(1, model.memory_len, -1).permute(1,0,2)
    seq = [vocab.sos_index]
    for _ in range(max_len):
        inp = torch.tensor([seq], device=device)
        emb = model.embedding(inp).permute(1,0,2)
        emb = model.pos_enc(emb)
        mask = nn.Transformer.generate_square_subsequent_mask(len(seq)).to(device)
        out = model.decoder(tgt=emb, memory=m, tgt_mask=mask)
        logits = model.out_proj(out)[-1,0]
        idx = logits.argmax().item()
        if idx == vocab.eos_index:
            break
        seq.append(idx)
    print("Token indices:", seq)
    print("Tokens:", [vocab.itos[i] if i < len(vocab.itos) else "<OOV>" for i in seq])

    return ''.join(vocab.itos[i] for i in seq[1:])

# 测试
test_raw = [722, 754, 109647]
print('Generated SMILES:', generate_smiles(test_raw))
print("Vocab size:", len(vocab.itos))
print("model.out_proj.weight:", model.out_proj.weight.shape)
