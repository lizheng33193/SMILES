"""
终极修复版 - 解决均值崩塌和缺失值预测问题
核心改进：
1. ✅ 加权损失函数（abs=5, em=2, eps=1）
2. ✅ 引入存在性判别头（presence gate）
3. ✅ 两阶段训练（先abs，再em，最后eps）
4. ✅ 严格的fold-scaler对齐
5. ✅ 常数解检测机制
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
    OUTPUT_DIR = os.path.join(BASE_DIR, "output", "improved_model_v3")
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
    VOCAB_PKL = os.path.join(DATA_DIR, "vocab.pkl")
    ENCODER_PTH = os.path.join(BASE_DIR, "model_pretrain_trfm", ".save", "trfm_new_9_0.pkl")
    CSV_PATH = os.path.join(DATA_DIR, "data_train.csv")

    HIDDEN_SIZE = 256
    N_LAYERS = 4
    DROPOUT = 0.1
    SEQ_LEN = 250

    BATCH_SIZE = 32
    N_EPOCHS = 300
    INITIAL_LR = 1e-4
    WEIGHT_DECAY = 1e-5
    GRAD_CLIP = 1.0
    PATIENCE = 50

    # 🔧 两阶段训练参数
    STAGE1_EPOCHS = 30  # 只训练abs
    STAGE2_EPOCHS = 50  # abs+em
    # 之后全部任务一起训练

    # 🔧 任务权重（在标准化空间）
    TASK_WEIGHTS = [5.0, 2.0, 1.0]  # [abs, em, eps]

    # 🔧 存在性判别阈值
    PRESENCE_THRESHOLD = 0.5
    PRESENCE_LOSS_WEIGHT = 0.5

    N_FOLDS = 5
    USE_KFOLD = True
    FINETUNE_ENCODER = False

    NUM_WORKERS = 4
    PIN_MEMORY = True

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

from build_vocab import WordVocab
from dataset import SmilesMultiRegDataset


class NoOpTransform:
    """用于已分词的processed_smiles"""

    def __call__(self, sm):
        if isinstance(sm, str):
            return sm.split()
        return sm


class OriginalStyleEncoder:
    """原始风格的编码器"""

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


class TrfmRegressor(nn.Module):
    """🔧 新增存在性判别头的回归模型"""

    def __init__(self, encoder_wrapper):
        super().__init__()
        D = 4 * encoder_wrapper.hidden_size
        self.encoder_wrapper = encoder_wrapper

        # 回归头（输出标准化空间的预测）
        self.reg_head = nn.Sequential(
            nn.Linear(D, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 3)  # [abs, em, log10(eps)]
        )

        # 🔧 存在性判别头（只对em和eps）
        self.presence_head = nn.Sequential(
            nn.Linear(D, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # logits for [em_exists, eps_exists]
        )

    def forward(self, src, mask=None):
        feats = self.encoder_wrapper.encode_batch(src, mask)
        feats = feats.to(next(self.reg_head.parameters()).device)

        y_pred = self.reg_head(feats)  # [B, 3] 标准化空间
        presence_logits = self.presence_head(feats)  # [B, 2]

        return y_pred, presence_logits


def prepare_data(csv_path, vocab_pkl, seq_len=250):
    df = pd.read_csv(csv_path)
    print(f"原始样本数: {len(df)}")

    print(f"\n标签统计:")
    for col in ['lam_abs', 'lam_em', 'eps']:
        valid = (~df[col].isna()).sum()
        print(f"  {col}: {valid}/{len(df)} 有效")
        if col == 'eps':
            eps_vals = df[col].dropna()
            print(f"    eps范围: [{eps_vals.min():.0f}, {eps_vals.max():.0f}]")
            print(f"    eps均值: {eps_vals.mean():.0f}")

    full_ds = SmilesMultiRegDataset(
        csv_path=csv_path,
        smiles_col='processed_smiles',
        label_cols=['lam_abs', 'lam_em', 'eps'],
        vocab=vocab_pkl,
        seq_len=seq_len,
        transform=NoOpTransform()
    )

    return full_ds, df


def train_epoch(model, dataloader, optimizer, scaler_list, device, config, current_stage=3):
    """
    🔧 两阶段训练
    current_stage: 1=只训练abs, 2=abs+em, 3=全部
    """
    model.train()
    total_loss = 0
    total_reg_loss = 0
    total_presence_loss = 0

    # 🔧 根据阶段设置任务权重
    if current_stage == 1:
        task_weights = torch.tensor([1.0, 0.0, 0.0], device=device)
    elif current_stage == 2:
        task_weights = torch.tensor([5.0, 2.0, 0.0], device=device)
    else:
        task_weights = torch.tensor(config.TASK_WEIGHTS, device=device)

    task_weights = task_weights.view(1, 3)

    for src, mask, labels in dataloader:
        src = src.to(device)
        labels_np = labels.numpy().copy()

        # 🔧 对eps做log10变换
        labels_np[:, 2] = np.log10(labels_np[:, 2] + 1e-6)

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
        y_pred, presence_logits = model(src, mask)

        # 🔧 加权回归损失（在标准化空间）
        target = torch.nan_to_num(labels_scaled, nan=0.0)
        valid_mask = torch.isfinite(labels_scaled).float()
        loss_elem = (y_pred - target) ** 2

        reg_loss = (loss_elem * valid_mask * task_weights).sum() / \
                   (valid_mask * task_weights).sum().clamp(min=1.0)

        # 🔧 存在性判别损失（只对em和eps）
        presence_target = torch.stack([
            torch.isfinite(labels_scaled[:, 1]).float(),  # em存在性
            torch.isfinite(labels_scaled[:, 2]).float()  # eps存在性
        ], dim=1)

        presence_loss = F.binary_cross_entropy_with_logits(
            presence_logits, presence_target
        )

        # 总损失
        loss = reg_loss + config.PRESENCE_LOSS_WEIGHT * presence_loss

        loss.backward()
        if config.GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item()
        total_reg_loss += reg_loss.item()
        total_presence_loss += presence_loss.item()

    n = len(dataloader)
    return total_loss / n, total_reg_loss / n, total_presence_loss / n


def evaluate(model, dataloader, scaler_list, device, config, check_constant=False):
    """
    🔧 评估函数，支持存在性判别
    check_constant: 是否检测常数解
    """
    model.eval()
    all_preds, all_labels = [], []
    all_preds_std = []  # 用于检测常数解

    with torch.no_grad():
        for src, mask, labels in dataloader:
            src = src.to(device)
            labels_np = labels.numpy().copy()

            # log10变换
            labels_np_log = labels_np.copy()
            labels_np_log[:, 2] = np.log10(labels_np_log[:, 2] + 1e-6)

            # 预测
            y_pred_std, presence_logits = model(src, mask)
            y_pred_std_np = y_pred_std.cpu().numpy()
            presence_probs = torch.sigmoid(presence_logits).cpu().numpy()

            if check_constant:
                all_preds_std.append(y_pred_std_np)

            # 反标准化
            preds_log = np.full_like(y_pred_std_np, np.nan)
            for i, scaler in enumerate(scaler_list):
                col_label = labels_np_log[:, i]
                valid = np.isfinite(col_label)
                if valid.any():
                    preds_log[valid, i] = scaler.inverse_transform(
                        y_pred_std_np[valid, i].reshape(-1, 1)
                    ).ravel()

            # 🔧 eps从log空间还原
            preds_original = preds_log.copy()
            preds_original[:, 2] = np.power(10.0, preds_log[:, 2])

            # 🔧 应用存在性判别（对em和eps）
            em_missing = presence_probs[:, 0] < config.PRESENCE_THRESHOLD
            eps_missing = presence_probs[:, 1] < config.PRESENCE_THRESHOLD

            preds_original[em_missing, 1] = np.nan  # 或设为极小值如250
            preds_original[eps_missing, 2] = 1.0  # eps极小值

            all_preds.append(preds_original)
            all_labels.append(labels_np)

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    # 🔧 常数解检测
    if check_constant and len(all_preds_std) > 0:
        all_preds_std = np.vstack(all_preds_std)
        print("\n🔍 常数解检测（标准化空间）:")
        for i, name in enumerate(['abs', 'em', 'log10(eps)']):
            mean = np.nanmean(all_preds_std[:, i])
            std = np.nanstd(all_preds_std[:, i])
            print(f"  {name}: mean={mean:.4f}, std={std:.4f}", end="")
            if std < 0.01:
                print(" ⚠️ 常数解!")
            else:
                print(" ✓")

    # 计算指标
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
    """🔧 两阶段训练流程"""
    device = config.DEVICE
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.INITIAL_LR,
        weight_decay=config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, verbose=True
    )

    best_val_metric = float('inf')
    patience_counter = 0
    best_model_state = None

    fold_str = f"Fold {fold}" if fold is not None else "Training"

    print(f"\n{'=' * 60}")
    print(f"{fold_str} - 两阶段训练")
    print(f"{'=' * 60}")

    for epoch in range(1, config.N_EPOCHS + 1):
        # 🔧 确定当前训练阶段
        if epoch <= config.STAGE1_EPOCHS:
            current_stage = 1
            stage_name = "Stage1(abs only)"
        elif epoch <= config.STAGE1_EPOCHS + config.STAGE2_EPOCHS:
            current_stage = 2
            stage_name = "Stage2(abs+em)"
        else:
            current_stage = 3
            stage_name = "Stage3(all)"

        train_loss, reg_loss, pres_loss = train_epoch(
            model, train_loader, optimizer, scaler_list, device, config, current_stage
        )

        # 每10轮或阶段切换时评估
        if epoch % 10 == 0 or epoch == 1 or \
                epoch == config.STAGE1_EPOCHS or \
                epoch == config.STAGE1_EPOCHS + config.STAGE2_EPOCHS:

            check_const = (epoch % 30 == 0)  # 每30轮检测常数解
            val_rmse, val_mae, val_r2, _, _ = evaluate(
                model, val_loader, scaler_list, device, config, check_const
            )

            # 🔧 只用abs和em的平均RMSE作为早停指标
            avg_val_metric = np.nanmean(val_rmse[:2])

            scheduler.step(avg_val_metric)
            current_lr = optimizer.param_groups[0]['lr']

            print(f"\n{fold_str} Epoch {epoch}/{config.N_EPOCHS} [{stage_name}]")
            print(f"  Loss: {train_loss:.4f} (Reg: {reg_loss:.4f}, Pres: {pres_loss:.4f})")
            print(f"  Val RMSE(abs,em): {avg_val_metric:.2f}")
            print(f"  RMSE [abs={val_rmse[0]:.1f}, em={val_rmse[1]:.1f}, eps={val_rmse[2]:.0f}]")
            print(f"  R²: {val_r2}")
            print(f"  LR: {current_lr:.2e}")

            if avg_val_metric < best_val_metric:
                best_val_metric = avg_val_metric
                patience_counter = 0
                best_model_state = deepcopy(model.state_dict())

                save_name = f'best_regressor_fold{fold}.pth' if fold else 'best_regressor.pth'
                torch.save(best_model_state, os.path.join(config.CHECKPOINT_DIR, save_name))
                print(f"  ✓ 新最佳模型")
            else:
                patience_counter += 10  # 因为每10轮才评估一次

            if patience_counter >= config.PATIENCE:
                print(f"\n早停! 最佳RMSE(abs,em): {best_val_metric:.2f}")
                break

    model.load_state_dict(best_model_state)
    return model, best_val_metric


def k_fold_training(full_dataset, vocab, pretrained_model_path, config):
    """K折交叉验证"""
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
            model, val_loader, scaler_list, config.DEVICE, config, check_constant=True
        )

        fold_results.append({
            'fold': fold,
            'best_metric': best_metric,
            'final_rmse': val_rmse,
            'final_mae': val_mae,
            'final_r2': val_r2
        })

        print(f"\nFold {fold} 最终结果:")
        print(f"  RMSE [abs={val_rmse[0]:.1f}, em={val_rmse[1]:.1f}, eps={val_rmse[2]:.0f}]")
        print(f"  R²: {val_r2}")

    print(f"\n{'=' * 60}")
    print("K折交叉验证汇总:")
    print(f"{'=' * 60}")
    avg_rmse = np.nanmean([r['final_rmse'] for r in fold_results], axis=0)
    std_rmse = np.nanstd([r['final_rmse'] for r in fold_results], axis=0)
    avg_r2 = np.nanmean([r['final_r2'] for r in fold_results], axis=0)
    print(f"平均RMSE [abs={avg_rmse[0]:.1f}, em={avg_rmse[1]:.1f}, eps={avg_rmse[2]:.0f}]")
    print(f"标准差: {std_rmse}")
    print(f"平均R²: {avg_r2}")

    return fold_results


if __name__ == "__main__":
    print("=" * 60)
    print("终极修复版训练脚本 v3")
    print("=" * 60)

    print("\n[1] 加载词汇表...")
    vocab = WordVocab.load_vocab(Config.VOCAB_PKL)
    print(f"词汇表大小: {len(vocab)}")

    print("\n[2] 准备数据...")
    full_ds, df_original = prepare_data(
        Config.CSV_PATH, Config.VOCAB_PKL, Config.SEQ_LEN
    )

    if Config.USE_KFOLD:
        print(f"\n[3] {Config.N_FOLDS}折交叉验证（两阶段训练）...")
        results = k_fold_training(
            full_ds, vocab, Config.ENCODER_PTH, Config
        )

    print(f"\n✅ 完成! 模型保存于: {Config.CHECKPOINT_DIR}")