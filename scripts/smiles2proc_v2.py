
"""
综合修复版 - 结合两个分析的关键修复点
主要修复：
1. ✅ 恢复 log10(eps) 变换（最关键）
2. ✅ 修复 tokenization（使用 NoOpTransform）
3. ✅ SEQ_LEN 增加到 220
4. ✅ 调度器只监控 lam_abs/lam_em
5. ✅ 恢复原始特征提取方式
6. ✅ 默认冻结 encoder
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import joblib
from copy import deepcopy
import random
from pathlib import Path


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


class Config:
    BASE_DIR = Path(__file__).resolve().parents[1]
    DATA_DIR = os.path.join(BASE_DIR, "data", "model_v2_datasets", "data_198", "data_with_processed", "train")
    OUTPUT_DIR = os.path.join(BASE_DIR, "output", "improved_model_fixed")
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
    VOCAB_PKL = os.path.join(DATA_DIR, "vocab.pkl")
    ENCODER_PTH = os.path.join(BASE_DIR, "model_pretrain_trfm", ".save", "trfm_new_9_0.pkl")
    CSV_PATH = os.path.join(DATA_DIR, "data_train.csv")
    # SCALER_PKL = os.path.join(DATA_DIR, "train_scaler.pkl")

    HIDDEN_SIZE = 256
    N_LAYERS = 4
    DROPOUT = 0.1
    SEQ_LEN = 250  # 🔧 修复3：从100增加到220

    BATCH_SIZE = 32
    N_EPOCHS = 300
    INITIAL_LR = 1e-4
    WEIGHT_DECAY = 1e-5
    GRAD_CLIP = 1.0
    PATIENCE = 50

    N_FOLDS = 5
    USE_KFOLD = True
    FINETUNE_ENCODER = False  # 默认冻结

    NUM_WORKERS = 4
    PIN_MEMORY = True

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

from build_vocab import WordVocab
from dataset import SmilesMultiRegDataset


# 🔧 修复2：NoOpTransform for processed_smiles
class NoOpTransform:
    """用于已分词的 processed_smiles"""
    def __call__(self, sm):
        if isinstance(sm, str):
            return sm.split()  # 直接按空格分词
        return sm


class OriginalStyleEncoder:
    """恢复原始编码器"""
    def __init__(self, pretrained_model, device='cpu'):
        self.model = pretrained_model
        self.device = device
        self.model.to(self.device)
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        self.hidden_size = pretrained_model.hidden_size

    def _encode(self, src):
        with torch.no_grad():
            embedded = self.model.embed(src)
            embedded = self.model.pe(embedded)

            output = embedded
            for i in range(self.model.trfm.encoder.num_layers - 1):
                output = self.model.trfm.encoder.layers[i](output, None)
            penul = output.clone()

            output = self.model.trfm.encoder.layers[-1](output, None)
            if self.model.trfm.encoder.norm:
                output = self.model.trfm.encoder.norm(output)

            output_np = output.cpu().numpy()
            penul_np = penul.cpu().numpy()

            features = np.hstack([
                np.mean(output_np, axis=0),
                np.max(output_np, axis=0),
                output_np[0, :, :],
                penul_np[0, :, :]
            ])

        return features

    def encode_batch(self, src, src_mask=None):
        src = src.transpose(0, 1).to(self.device)
        batch_size = src.shape[1]

        if batch_size <= 100:
            return torch.from_numpy(self._encode(src)).float()

        results = []
        for i in range(0, batch_size, 100):
            end = min(i + 100, batch_size)
            feats = self._encode(src[:, i:end])
            results.append(feats)
        return torch.from_numpy(np.concatenate(results, axis=0)).float()


class SimpleRegressionHead(nn.Module):
    def __init__(self, input_size=1024, hidden_size=128, n_outputs=3, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_outputs)
        )

    def forward(self, x):
        return self.net(x)


class TrfmRegressor(nn.Module):
    def __init__(self, encoder_wrapper):
        super().__init__()
        self.encoder_wrapper = encoder_wrapper
        self.head = SimpleRegressionHead(
            input_size=4 * encoder_wrapper.hidden_size,
            hidden_size=128,
            n_outputs=3,
            dropout=0.1
        )

    def forward(self, src, mask=None):
        feats = self.encoder_wrapper.encode_batch(src, mask)
        feats = feats.to(next(self.head.parameters()).device)
        return self.head(feats)


def prepare_data(csv_path, vocab_pkl, seq_len=220):
    df = pd.read_csv(csv_path)
    print(f"原始样本数: {len(df)}")

    # 检查数据
    print(f"\n标签统计:")
    for col in ['lam_abs', 'lam_em', 'eps']:
        valid = (~df[col].isna()).sum()
        print(f"  {col}: {valid}/{len(df)} 有效")
        if col == 'eps':
            eps_vals = df[col].dropna()
            print(f"    eps范围: [{eps_vals.min():.0f}, {eps_vals.max():.0f}]")
            print(f"    eps均值: {eps_vals.mean():.0f}")

    # 🔧 修复2：使用NoOpTransform
    full_ds = SmilesMultiRegDataset(
        csv_path=csv_path,
        smiles_col='processed_smiles',
        label_cols=['lam_abs', 'lam_em', 'eps'],
        vocab=vocab_pkl,
        seq_len=seq_len,
        transform=NoOpTransform()  # 关键修复
    )

    return full_ds, df


def train_epoch(model, dataloader, optimizer, scaler_list, device, grad_clip=1.0):
    model.train()
    total_loss = 0

    for src, mask, labels in dataloader:
        src = src.to(device)
        labels_np = labels.numpy().copy()

        # 🔧 修复1：对eps做log10变换
        labels_np[:, 2] = np.log10(labels_np[:, 2] + 1e-6)  # 避免log(0)

        # 标准化
        labels_scaled = np.full_like(labels_np, np.nan, dtype=np.float32)
        for i, scaler in enumerate(scaler_list):
            col = labels_np[:, i]
            valid = np.isfinite(col)
            if valid.any():
                labels_scaled[valid, i] = scaler.transform(
                    col[valid].reshape(-1, 1)
                ).ravel()

        labels_scaled = torch.from_numpy(labels_scaled).to(device)

        optimizer.zero_grad()
        preds = model(src, mask)

        target = torch.nan_to_num(labels_scaled, nan=0.0)
        loss_elem = F.mse_loss(preds, target, reduction='none')
        valid_mask = torch.isfinite(labels_scaled).float()
        loss = (loss_elem * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)

        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, scaler_list, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for src, mask, labels in dataloader:
            src = src.to(device)
            labels_np = labels.numpy().copy()

            # 🔧 修复1：对eps做log10变换
            labels_np_log = labels_np.copy()
            labels_np_log[:, 2] = np.log10(labels_np_log[:, 2] + 1e-6)

            # 预测（在log空间）
            preds_scaled = model(src, mask).cpu().numpy()

            # 反标准化（得到log空间的预测）
            preds_log = np.full_like(preds_scaled, np.nan)
            for i, scaler in enumerate(scaler_list):
                col_label = labels_np_log[:, i]
                valid = np.isfinite(col_label)
                if valid.any():
                    preds_log[valid, i] = scaler.inverse_transform(
                        preds_scaled[valid, i].reshape(-1, 1)
                    ).ravel()

            # 🔧 修复1：eps从log空间还原
            preds_original = preds_log.copy()
            preds_original[:, 2] = np.power(10.0, preds_log[:, 2])

            all_preds.append(preds_original)
            all_labels.append(labels_np)  # 原始尺度的label

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    # 计算指标（在原始尺度）
    rmse, mae, r2 = [], [], []
    dim_names = ['lam_abs', 'lam_em', 'eps']

    for i in range(3):
        pred_col = all_preds[:, i]
        label_col = all_labels[:, i]
        valid = np.isfinite(label_col) & np.isfinite(pred_col)

        if valid.sum() > 0:
            p, l = pred_col[valid], label_col[valid]
            rmse.append(np.sqrt(((p - l) ** 2).mean()))
            mae.append(np.abs(p - l).mean())
            try:
                r2.append(r2_score(l, p))
            except:
                r2.append(0.0)
        else:
            rmse.append(np.nan)
            mae.append(np.nan)
            r2.append(np.nan)

    return np.array(rmse), np.array(mae), r2, all_preds, all_labels


def train_model(model, train_loader, val_loader, scaler_list, config, fold=None):
    device = config.DEVICE
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.INITIAL_LR,
        weight_decay=config.WEIGHT_DECAY
    )

    # 🔧 修复4：调度器只监控lam_abs和lam_em
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, verbose=True
    )

    best_val_metric = float('inf')
    patience_counter = 0
    best_model_state = None

    fold_str = f"Fold {fold}" if fold is not None else "Training"

    for epoch in range(1, config.N_EPOCHS + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler_list, device, config.GRAD_CLIP
        )

        val_rmse, val_mae, val_r2, _, _ = evaluate(
            model, val_loader, scaler_list, device
        )

        # 🔧 修复4：只用lam_abs和lam_em的平均RMSE
        avg_val_metric = np.nanmean(val_rmse[:2])

        scheduler.step(avg_val_metric)
        current_lr = optimizer.param_groups[0]['lr']

        if epoch % 10 == 0 or epoch == 1:
            print(f"{fold_str} Epoch {epoch}/{config.N_EPOCHS}")
            print(f"  Loss: {train_loss:.4f} | Val RMSE(abs,em): {avg_val_metric:.2f}")
            print(f"  RMSE [abs, em, eps]: {val_rmse}")
            print(f"  R²: {val_r2} | LR: {current_lr:.2e}")

        if avg_val_metric < best_val_metric:
            best_val_metric = avg_val_metric
            patience_counter = 0
            best_model_state = deepcopy(model.state_dict())

            save_name = f'best_regressor_fold{fold}.pth' if fold else 'best_regressor.pth'
            torch.save(best_model_state, os.path.join(config.CHECKPOINT_DIR, save_name))

            if epoch % 10 == 0:
                print(f"  ✓ 新最佳")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"\n早停! 最佳RMSE(abs,em): {best_val_metric:.2f}")
                break

    model.load_state_dict(best_model_state)
    return model, best_val_metric


def k_fold_training(full_dataset, vocab, pretrained_model_path, config):
    kfold = KFold(n_splits=config.N_FOLDS, shuffle=True, random_state=42)
    fold_results = []
    all_indices = list(range(len(full_dataset)))

    for fold, (train_idx, val_idx) in enumerate(kfold.split(all_indices), 1):
        print(f"\n{'=' * 60}")
        print(f"Fold {fold}/{config.N_FOLDS}")
        print(f"训练: {len(train_idx)} | 验证: {len(val_idx)}")
        print(f"{'=' * 60}\n")

        train_subset = torch.utils.data.Subset(full_dataset, train_idx)
        val_subset = torch.utils.data.Subset(full_dataset, val_idx)

        train_loader = DataLoader(
            train_subset, batch_size=config.BATCH_SIZE, shuffle=True,
            collate_fn=SmilesMultiRegDataset.collate_fn,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY
        )
        val_loader = DataLoader(
            val_subset, batch_size=config.BATCH_SIZE, shuffle=False,
            collate_fn=SmilesMultiRegDataset.collate_fn,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY
        )

        # 准备scaler（在log空间）
        train_labels = []
        for _, _, labels in train_loader:
            train_labels.append(labels.numpy())
        Y_train = np.vstack(train_labels).copy()

        # 🔧 修复1：对eps做log10变换
        Y_train[:, 2] = np.log10(Y_train[:, 2] + 1e-6)

        scaler_list = []
        for i in range(3):
            col = Y_train[:, i]
            valid = np.isfinite(col)
            if valid.sum() > 0:
                scaler = StandardScaler()
                scaler.fit(col[valid].reshape(-1, 1))
                scaler_list.append(scaler)
                print(f"  {'lam_abs,lam_em,log10(eps)'.split(',')[i]}: "
                      f"mean={scaler.mean_[0]:.2f}, std={scaler.scale_[0]:.2f}")
            else:
                scaler = StandardScaler()
                scaler.mean_, scaler.scale_ = np.array([0.]), np.array([1.])
                scaler_list.append(scaler)

        joblib.dump(scaler_list, os.path.join(config.CHECKPOINT_DIR, f'scaler_fold{fold}.pkl'))

        # 加载encoder
        from pretrain_trfm import TrfmSeq2seq
        encoder = TrfmSeq2seq(
            in_size=len(vocab),
            hidden_size=config.HIDDEN_SIZE,
            out_size=len(vocab),
            n_layers=config.N_LAYERS,
            dropout=config.DROPOUT
        )
        state = torch.load(pretrained_model_path, map_location='cpu', weights_only=False)
        encoder.load_state_dict(state)

        encoder_wrapper = OriginalStyleEncoder(encoder, device=config.DEVICE)
        model = TrfmRegressor(encoder_wrapper)

        # 训练
        model, best_metric = train_model(
            model, train_loader, val_loader, scaler_list, config, fold=fold
        )

        # 最终评估
        val_rmse, val_mae, val_r2, _, _ = evaluate(
            model, val_loader, scaler_list, config.DEVICE
        )

        fold_results.append({
            'fold': fold,
            'best_metric': best_metric,
            'final_rmse': val_rmse,
            'final_mae': val_mae,
            'final_r2': val_r2
        })

        print(f"\nFold {fold} 结果:")
        print(f"  RMSE [abs, em, eps]: {val_rmse}")
        print(f"  R²: {val_r2}")

    print(f"\n{'=' * 60}")
    print("K折交叉验证汇总:")
    print(f"{'=' * 60}")
    avg_rmse = np.nanmean([r['final_rmse'] for r in fold_results], axis=0)
    std_rmse = np.nanstd([r['final_rmse'] for r in fold_results], axis=0)
    avg_r2 = np.nanmean([r['final_r2'] for r in fold_results], axis=0)
    print(f"平均RMSE [abs, em, eps]: {avg_rmse}")
    print(f"标准差: {std_rmse}")
    print(f"平均R²: {avg_r2}")

    return fold_results


# 🔧 额外：UNK检测脚本
def check_unk_ratio(dataloader, vocab):
    """检查UNK token比例"""
    cnt_unk, cnt_all = 0, 0
    for src, _, _ in dataloader:
        cnt_unk += (src == vocab.unk_index).sum().item()
        cnt_all += (src != vocab.pad_index).sum().item()

    unk_ratio = cnt_unk / max(cnt_all, 1)
    print(f"\n🔍 UNK检测: {unk_ratio*100:.2f}% ({cnt_unk}/{cnt_all})")
    if unk_ratio > 0.2:
        print("⚠️  警告: UNK比例过高，请检查tokenization!")
    return unk_ratio


if __name__ == "__main__":
    print("=" * 60)
    print("综合修复版训练脚本")
    print("=" * 60)

    print("\n[1] 加载词汇表...")
    vocab = WordVocab.load_vocab(Config.VOCAB_PKL)
    print(f"词汇表大小: {len(vocab)}")

    print("\n[2] 准备数据...")
    full_ds, df_original = prepare_data(
        Config.CSV_PATH, Config.VOCAB_PKL, Config.SEQ_LEN
    )

    # UNK检测
    print("\n[3] UNK比例检测...")
    temp_loader = DataLoader(full_ds, batch_size=32, collate_fn=SmilesMultiRegDataset.collate_fn)
    check_unk_ratio(temp_loader, vocab)

    if Config.USE_KFOLD:
        print(f"\n[4] {Config.N_FOLDS}折交叉验证...")
        results = k_fold_training(
            full_ds, vocab, Config.ENCODER_PTH, Config
        )

    print(f"\n完成! 模型保存于: {Config.CHECKPOINT_DIR}")