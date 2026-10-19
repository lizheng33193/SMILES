import torch, torch.optim as optim
from torch.utils.data import DataLoader
from dataset_prop2smiles import Prop2SmilesDataset
from model_prop2smiles   import Prop2SmilesModel
from build_vocab import WordVocab
import sys
sys.modules['__main__'].WordVocab = WordVocab
# —— 配置 —— 
CSV_PATH   = r"D:\workspace\pycharm\smiles-transformer\tests\datasets\proc\proc2.csv"
VOCAB_PATH = r"D:\workspace\pycharm\smiles-transformer\tests\datasets\proc\vocab.pkl"
BATCH      = 32
LR         = 1e-4
EPOCHS     = 20
SEQ_LEN    = 220

# —— 数据 & 模型 —— 
ds = Prop2SmilesDataset(CSV_PATH,
                        smiles_col="processed_smiles",
                        prop_cols=["lam_abs","lam_em","eps"],
                        vocab_path=VOCAB_PATH,
                        seq_len=SEQ_LEN)
loader = DataLoader(ds, batch_size=BATCH, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Prop2SmilesModel(
    prop_dim=3, latent_dim=128,
    vocab_size=len(ds.vocab),
    hidden_size=256, n_head=4, n_dec_layers=4,
    seq_len=SEQ_LEN,
    prop_recon_weight=1.0
).to(device)

opt = optim.Adam(model.parameters(), lr=LR)

# —— 训练循环 —— 
for epoch in range(1, EPOCHS+1):
    model.train()
    total = 0
    for props, seq in loader:
        props, seq = props.to(device), seq.to(device)
        opt.zero_grad()
        loss, Ls, Lp = model(props, seq)
        loss.backward()
        opt.step()
        total += loss.item()
    print(f"Epoch {epoch:02d} | Loss: {total/len(loader):.4f}")
    # （可加保存 checkpoint）
