import random
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from enumerator import SmilesEnumerator
from utils import split

PAD = 0
MAX_LEN = 220

class Randomizer(object):

    def __init__(self):
        self.sme = SmilesEnumerator()
    
    def __call__(self, sm):
        sm_r = self.sme.randomize_smiles(sm) # Random transoform
        if sm_r is None:
            sm_spaced = split(sm) # Spacing
        else:
            sm_spaced = split(sm_r) # Spacing
        sm_split = sm_spaced.split()
        if len(sm_split)<=MAX_LEN - 2:
            return sm_split # List
        else:
            return split(sm).split()

    def random_transform(self, sm):
        '''
        function: Random transformation for SMILES. It may take some time.
        input: A SMILES
        output: A randomized SMILES
        '''
        return self.sme.randomize_smiles(sm)

class Seq2seqDataset(Dataset):

    def __init__(self, smiles, vocab, seq_len=220, transform=Randomizer()):
        self.smiles = smiles
        self.vocab = vocab
        self.seq_len = seq_len
        self.transform = transform

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, item):
        sm = self.smiles[item]
        sm = self.transform(sm) # List
        content = [self.vocab.stoi.get(token, self.vocab.unk_index) for token in sm]
        X = [self.vocab.sos_index] + content + [self.vocab.eos_index]
        padding = [self.vocab.pad_index]*(self.seq_len - len(X))
        X.extend(padding)
        return torch.tensor(X)

class SmilesMultiRegDataset(Dataset):
    """
    用于多目标回归（lam_abs, lam_em, eps）的 Dataset。
    CSV 里必须有 processed_smiles 列，以及三列标签。
    """
    def __init__(self,
                 csv_path: str,
                 smiles_col: str,
                 label_cols: list,
                 vocab,            # 可以是 pickle 路径，也可以是已经 load 的 vocab 对象
                 seq_len: int = 220,
                 transform=None):
        # 1. 载入数据
        df = pd.read_csv(csv_path)
        self.smiles_list = df[smiles_col].astype(str).tolist()
        self.labels = df[label_cols].values.astype(float)
        # 2. 载入 vocab
        if isinstance(vocab, str):
            import pickle
            with open(vocab, "rb") as f:
                self.vocab = pickle.load(f)
        else:
            self.vocab = vocab
        # 3. 其它参数
        self.seq_len = seq_len
        self.transform = transform or Randomizer()

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        sm = self.smiles_list[idx]
        tokens = self.transform(sm)   # List of token strings
        # token -> id
        content = [ self.vocab.stoi.get(tok, self.vocab.unk_index) for tok in tokens ]
        X = [self.vocab.sos_index] + content + [self.vocab.eos_index]
        # padding
        pad_len = self.seq_len - len(X)
        if pad_len > 0:
            X = X + [self.vocab.pad_index] * pad_len
        else:
            X = X[:self.seq_len]
        # 返回 (seq_tensor, label_tensor)
        return torch.tensor(X, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float)

    @staticmethod
    def collate_fn(batch):
        """
        batch: list of (seq_tensor, label_tensor)
        返回： src: LongTensor [B, L]
               src_mask: BoolTensor [B, L] (True 表示 pad)
               labels: FloatTensor [B,3]
        """
        seqs, labs = zip(*batch)
        src = torch.stack(seqs, dim=0)            # [B, L]
        labels = torch.stack(labs, dim=0)         # [B, 3]
        # mask for padding tokens (pad_index==0)
        src_mask = (src == 0)                     # [B, L]
        return src, src_mask, labels