import pandas as pd
import torch
from torch.utils.data import Dataset
from utils import split     # 或者 utils.split，取决于你原来项目里的分词函数
import pickle
from build_vocab import WordVocab

class Prop2SmilesDataset(Dataset):
    def __init__(self, csv_path, smiles_col, prop_cols, vocab_path, seq_len=220, transform=None):
        df = pd.read_csv(csv_path)
        # 属性向量
        self.props = torch.tensor(df[prop_cols].values.astype(float), dtype=torch.float)  # [N,3]
        # SMILES 原始字符串
        self.smiles = df[smiles_col].astype(str).tolist()
        self.seq_len = seq_len
        self.transform = transform
        
        # 加载词表
        with open(vocab_path, "rb") as f:
            self.vocab = pickle.load(f)
    def __len__(self): return len(self.props)
    def __getitem__(self, idx):
        prop = self.props[idx]            # [3]
        sm  = self.smiles[idx]
        # 分子切分
        toks = (self.transform(sm).split() if self.transform else split(sm).split())
        ids  = [self.vocab.stoi.get(t, self.vocab.unk_index) for t in toks]
        seq  = [self.vocab.sos_index] + ids + [self.vocab.eos_index]
        # pad 或截断
        if len(seq)<self.seq_len:
            seq += [self.vocab.pad_index]*(self.seq_len-len(seq))
        else:
            seq = seq[:self.seq_len]
        return prop, torch.tensor(seq, dtype=torch.long)
