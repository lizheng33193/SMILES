import os, pickle, numpy as np, pandas as pd
import torch
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from pretrain_trfm import TrfmSeq2seq
from build_vocab import WordVocab
from dataset import SmilesMultiRegDataset
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt

VOCAB_PKL   = r"D:\workspace\pycharm\smiles-transformer\data\WPT,AJ,ZYM\datasets\vocab.pkl"
ENC_PTH     = r"D:\workspace\pycharm\smiles-transformer\.save\trfm_new_9_0.pkl"
CSV_PATH    = r"D:\workspace\pycharm\smiles-transformer\data\WPT,AJ,ZYM\datasets\data.csv"
CHECKPOINT_DIR = r"D:\workspace\pycharm\smiles-transformer\output\myproc.pkl"
SCALER_PKL      = r"D:\workspace\pycharm\smiles-transformer\scripts\checkpoints\scaler.pkl"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
device = torch.device("cpu")


# 2.1 加载 WordVocab
vocab: WordVocab = WordVocab.load_vocab(VOCAB_PKL)
print(f"词表大小 = {len(vocab)}")

# 2.2 实例化模型（CPU）
model = TrfmSeq2seq(
    in_size     = len(vocab),
    hidden_size = 256,
    out_size    = len(vocab),
    n_layers    = 4,
    dropout     = 0.1
)
# 2.3 加载权重
state = torch.load(ENC_PTH, map_location="cpu")
model.load_state_dict(state)
model.eval()
print(f"加载自编码器权重: {ENC_PTH}")

# 词表大小 = 45
# 加载自编码器权重: D:\workspace\pycharm\smiles-transformer\.save\trfm_new_9_0.pkl
# d:\ProgramData\anaconda3\envs\DECIMER\lib\site-packages\torch\nn\modules\transformer.py:379: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
#   warnings.warn(
# C:\Users\Admin\AppData\Local\Temp\ipykernel_21988\561195607.py:14: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
#   state = torch.load(ENC_PTH, map_location="cpu")



# 3.1 构造 Dataset
full_ds = SmilesMultiRegDataset(
    csv_path   = CSV_PATH,
    smiles_col = "processed_smiles",
    label_cols = ["lam_abs","lam_em","eps"],
    vocab      = vocab,
    seq_len    = 100
)
print(f"总样本数 = {len(full_ds)}")

# 3.2 拆分 Train/Val (80/20)
n = len(full_ds)
train_n = int(n*0.8)
val_n   = n - train_n
train_ds, val_ds = random_split(full_ds, [train_n, val_n], generator=torch.Generator().manual_seed(42))
print(f"train={len(train_ds)}, val={len(val_ds)}")

# 3.3 DataLoader
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                          collate_fn=SmilesMultiRegDataset.collate_fn)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False,
                          collate_fn=SmilesMultiRegDataset.collate_fn)
# 总样本数 = 130
# train=104, val=26



# Step 4：标准化标签并保存 Scaler
from sklearn.preprocessing import StandardScaler
import joblib

# 4.1 收集训练集所有标签
all_labels = []
for _, _, labels in train_loader:
    all_labels.append(labels.numpy())
Y_train = np.vstack(all_labels)

# 4.2 构建 StandardScaler 并保存到本地
scaler = StandardScaler()
scaler.fit(Y_train)
joblib.dump(scaler, "scaler.pkl")  # 保存为 scaler.pkl
print("Saved scaler to scaler.pkl")




## Step 5：定义回归模型

# 5.1 回归模型定义
class TrfmRegressor(nn.Module):
    def __init__(self, encoder, hidden_size=256):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(4*hidden_size, 128),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 3)
        )
    def forward(self, src, mask):
        feats = self.encoder.encode(src.transpose(0,1))  # numpy (B,4H)
        feats = torch.from_numpy(feats).float()          # CPU Tensor
        return self.head(feats)



regressor = TrfmRegressor(model, hidden_size=256)
optimizer = torch.optim.Adam(regressor.parameters(), lr=1e-4)
best_val_mse = float('inf')
num_epochs = 300

for epoch in range(1, num_epochs+1):
    # 训练
    regressor.train()
    total_train = 0
    for src, mask, labels in train_loader:
        y = scaler.transform(labels.numpy())
        y = torch.from_numpy(y).float()
        preds = regressor(src, mask)
        loss = F.mse_loss(preds, y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_train += loss.item()
    avg_train = total_train/len(train_loader)

    # 验证
    regressor.eval()
    total_val = 0
    with torch.no_grad():
        for src, mask, labels in val_loader:
            y = scaler.transform(labels.numpy())
            y = torch.from_numpy(y).float()
            preds = regressor(src, mask)
            total_val += F.mse_loss(preds, y).item()
    avg_val = total_val/len(val_loader)

    print(f"Epoch {epoch}/{num_epochs}  train MSE={avg_train:.4f}  val MSE={avg_val:.4f}")
    if avg_val < best_val_mse:
        best_val_mse = avg_val
        torch.save(regressor.state_dict(), os.path.join(CHECKPOINT_DIR, 'best_regressor.pth'))
        print("  ▶ New best saved")

print("Training done. Best val MSE (std labels)=", best_val_mse)




# 6.1 加载最优权重
regressor.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR,'best_regressor.pth')))
regressor.eval()

# 6.2 收集 preds & trues
y_true, y_pred_std = [], []
for src, mask, labels in val_loader:
    preds_std = regressor(src, mask).detach().numpy()
    y_pred_std.append(preds_std)
    y_true.append(labels.numpy())
Y_true = np.vstack(y_true)
Y_pred_std = np.vstack(y_pred_std)

# 6.3 反标准化
scaler = joblib.load(r"D:\workspace\pycharm\smiles-transformer\scripts\checkpoints\scaler.pkl")  # 直接使用文件名，不依赖变量
Y_pred = scaler.inverse_transform(Y_pred_std)

# 6.4 计算 RMSE
rmse = np.sqrt(((Y_true - Y_pred)**2).mean(axis=0))
print(f"Final RMSE [lam_abs, lam_em, eps] = {rmse}")

# Final RMSE [lam_abs, lam_em, eps] = [   79.237045   114.203545 47348.72    ]
# C:\Users\Admin\AppData\Local\Temp\ipykernel_21988\3878661677.py:2: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
#   regressor.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR,'best_regressor.pth')))


new_smiles = [
    "C1=CC=C(C=C1)C2=CC(=C3N=C4C(=CC(=[N+]4[B-](F)(F)N23)C5=CC=CC=C5)C6=CC=C(C=C6)CN7CCOCC7)C8=CC=C(C=C8)CN9CCOCC9",
    "C 1 = C C = C ( C = C 1 ) C 2 = C C ( = C 3 N = C 4 C ( = C C ( = [ N+ ] 4 [ B - ] ( F ) ( F ) N 2 3 ) C 5 = C C = C C = C 5 ) C 6 = C C = C ( C = C 6 ) C N 7 C C O C C 7 ) C 8 = C C = C ( C = C 8 ) C N 9 C C O C C 9",
    "CCN(CC)C1=CC=C(C=C1)C2=N/C(=N\C3=C(C4=CC=CC=C4)C5=C(C6=C(CC5)C=C(C=C6)OC)N3[B-](F)(F)F)/C(=C2)C7=CC=CC=C7",
    "CC1=CC=C(C=C1)S(=O)(=O)OC2=CC=C(C=C2)C3=CC(=C4N=C5C(=CC(=[N+]5[B-](F)(F)N34)C6=CC=C(C=C6)OS(=O)(=O)C7=CC=C(C)C=C7)C8=CC=CC=C8)C9=CC=CC=C9",
    "C C N 1 C 2 = C ( C = C C = C 2 ) C 3 = C C ( = C C = C 3 1 ) C 4 = C ( C ( = [ N+ ] 5 C 4 = N C 6 = C ( C 7 = C C 8 = C ( C = C 7 ) N ( C C ) C 9 = C 8 C = C C = C 9 ) C ( = C ( C % 1 0 = C C = C ( C = C % 1 0 ) Br ) N 6 [ B - ] 5 ( F ) F ) I ) C % 1 1 = C C = C ( C = C % 1 1 ) Br ) I"
]
# 临时 CSV
pd.DataFrame({
    'processed_smiles':new_smiles,
    'lam_abs':0, 'lam_em':0, 'eps':0
}).to_csv('tmp_pred.csv',index=False)

# DataLoader
ds_pred = SmilesMultiRegDataset('tmp_pred.csv','processed_smiles',
                                ['lam_abs','lam_em','eps'], vocab, 100)
loader_pred = DataLoader(ds_pred,batch_size=16,shuffle=False,
                         collate_fn=SmilesMultiRegDataset.collate_fn)

# 预测并反标准化
preds=[]
for src,mask,_ in loader_pred:
    pred_std = regressor(src,mask).detach().numpy()
    pred = scaler.inverse_transform(pred_std)
    preds.append(pred)
preds = np.vstack(preds)

# 打印结果
for smi,(a,e,p) in zip(new_smiles,preds):
    print(f"{smi} -> lam_abs={a:.1f}, lam_em={e:.1f}, eps={p:.1f}")


