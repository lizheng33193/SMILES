"""
最终修复版 SMILES 多目标回归训练脚本
已修复所有问题:
1. ✅ 微调时去掉no_grad,允许梯度回传
2. ✅ head输入维度从encoder_wrapper读取
3. ✅ K折每折独立encoder副本
4. ✅ score_smiles分词一致性
5. ✅ 优先级权重调整
6. ✅ K折输出eps原始尺度
7. ✅ non_blocking异步拷贝
8. ✅ 修复scaler变量名不一致
9. ✅ 修复train_epoch的NaN处理
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
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from copy import deepcopy
import random
from pathlib import Path


# 固定随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


# ================== 配置 ==================
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
    SEQ_LEN = 100

    BATCH_SIZE = 32
    N_EPOCHS = 300
    INITIAL_LR = 1e-3
    ENCODER_LR = 1e-5
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP = 1.0
    PATIENCE = 50

    N_FOLDS = 5
    USE_KFOLD = True

    FINETUNE_ENCODER = False
    FINETUNE_LAYERS = 2

    NUM_WORKERS = 4
    PIN_MEMORY = True
    USE_MULTI_HEAD = True
    USE_SMOOTHL1 = True

    # 修复5: 调整优先级权重
    SCORE_WEIGHTS = {'lam_abs': 1.0, 'lam_em': 0.23, 'eps': 0.005}

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

from build_vocab import WordVocab
from dataset import SmilesMultiRegDataset
from utils import split  # 用于分词一致性


# ================== 修复1: 支持微调的Encoder ==================
class ImprovedEncoder:
    """支持条件性梯度计算的编码器"""

    def __init__(self, pretrained_model, device='cpu', enable_grad=False):
        self.model = pretrained_model
        self.device = device
        self.enable_grad = enable_grad  # 新增: 控制是否计算梯度
        self.model.to(self.device)

        # 根据enable_grad设置模式
        if not enable_grad:
            self.model.eval()

        self.hidden_size = pretrained_model.hidden_size

    def encode_with_attention_pooling(self, src, src_mask):
        """
        修复1: 条件性no_grad
        修复NaN: 内部重算PAD mask,确保极性正确
        """
        src = src.to(self.device)
        src_before_transpose = src  # 保存用于计算mask

        # 关键修复: 内部重算PAD mask (PAD token index=0)
        pad_mask = (src_before_transpose == 0)  # True=PAD, False=有效token

        # 全PAD样本保护: 强制解开至少一个位置
        all_pad = pad_mask.all(dim=1)  # (B,)
        if all_pad.any():
            pad_mask[all_pad, 0] = False  # 解开第0个位置
            all_pad_count = all_pad.sum().item()
            if all_pad_count > 0:
                print(f"Warning: {all_pad_count} 全PAD样本被修复")

        # 一次性自检 (仅第一个batch)
        if not hasattr(self, '_mask_checked'):
            ext_mask = src_mask.to(self.device).bool() if src_mask is not None else pad_mask
            mismatch = (ext_mask ^ pad_mask).float().mean().item()
            all_pad_rate = pad_mask.all(dim=1).float().mean().item()
            print(f"[Mask自检] 外部vs内部不一致率: {mismatch:.4f}")
            print(f"[Mask自检] 全PAD样本比例: {all_pad_rate:.4f}")
            if mismatch > 0.01:
                print("  ⚠ 极性可能相反,已切换到内部计算的pad_mask")
            self._mask_checked = True

        src = src_before_transpose.transpose(0, 1)  # (L, B)

        # 修复1: 仅在不需要梯度时使用no_grad
        if self.enable_grad:
            # 微调模式: 允许梯度
            embedded = self.model.embed(src)
            embedded = self.model.pe(embedded)

            output = embedded
            for i in range(self.model.trfm.encoder.num_layers - 1):
                output = self.model.trfm.encoder.layers[i](
                    output, src_key_padding_mask=pad_mask  # 使用内部计算的pad_mask
                )
            penultimate = output.clone()

            output = self.model.trfm.encoder.layers[-1](
                output, src_key_padding_mask=pad_mask
            )
            if self.model.trfm.encoder.norm:
                output = self.model.trfm.encoder.norm(output)
        else:
            # 冻结模式: 不计算梯度
            with torch.no_grad():
                embedded = self.model.embed(src)
                embedded = self.model.pe(embedded)

                output = embedded
                for i in range(self.model.trfm.encoder.num_layers - 1):
                    output = self.model.trfm.encoder.layers[i](
                        output, src_key_padding_mask=pad_mask  # 使用内部计算的pad_mask
                    )
                penultimate = output.clone()

                output = self.model.trfm.encoder.layers[-1](
                    output, src_key_padding_mask=pad_mask
                )
                if self.model.trfm.encoder.norm:
                    output = self.model.trfm.encoder.norm(output)

        output = output.transpose(0, 1)
        penultimate = penultimate.transpose(0, 1)

        # 使用内部计算的pad_mask进行池化
        valid_mask = (~pad_mask).unsqueeze(-1).float()

        sum_output = (output * valid_mask).sum(dim=1)
        sum_penul = (penultimate * valid_mask).sum(dim=1)
        count = valid_mask.sum(dim=1).clamp(min=1)
        mean_output = sum_output / count
        mean_penul = sum_penul / count

        masked_output = output.masked_fill(pad_mask.unsqueeze(-1), -1e9)
        masked_penul = penultimate.masked_fill(pad_mask.unsqueeze(-1), -1e9)
        max_output = masked_output.max(dim=1)[0]
        max_penul = masked_penul.max(dim=1)[0]

        features = torch.cat([
            mean_output, max_output, mean_penul, max_penul
        ], dim=-1)

        return features

    def encode_batch(self, src, src_mask):
        batch_size = src.shape[0]
        if batch_size <= 100:
            return self.encode_with_attention_pooling(src, src_mask)

        results = []
        for i in range(0, batch_size, 100):
            end_idx = min(i + 100, batch_size)
            batch_features = self.encode_with_attention_pooling(
                src[i:end_idx], src_mask[i:end_idx]
            )
            results.append(batch_features)
        return torch.cat(results, dim=0)


# ================== 回归头 ==================
class ImprovedRegressionHead(nn.Module):
    def __init__(self, input_size=1024, hidden_sizes=[512, 256, 128],
                 n_outputs=3, dropout=0.2, use_multi_head=True):
        super().__init__()
        self.use_multi_head = use_multi_head

        layers = []
        in_features = input_size
        for hidden in hidden_sizes[:-1]:
            layers.extend([
                nn.Linear(in_features, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            in_features = hidden
        self.backbone = nn.Sequential(*layers)

        if use_multi_head:
            self.heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(in_features, hidden_sizes[-1]),
                    nn.LayerNorm(hidden_sizes[-1]),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_sizes[-1], 1)
                ) for _ in range(n_outputs)
            ])
        else:
            self.output_layer = nn.Sequential(
                nn.Linear(in_features, hidden_sizes[-1]),
                nn.LayerNorm(hidden_sizes[-1]),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_sizes[-1], n_outputs)
            )

    def forward(self, x):
        shared = self.backbone(x)
        if self.use_multi_head:
            outputs = [head(shared) for head in self.heads]
            return torch.cat(outputs, dim=-1)
        else:
            return self.output_layer(shared)


# ================== 修复2: 从encoder_wrapper读取hidden_size ==================
class TrfmRegressor(nn.Module):
    def __init__(self, encoder_wrapper, use_multi_head=True):
        super().__init__()
        self.encoder_wrapper = encoder_wrapper

        # 修复2: 从encoder_wrapper获取真实的hidden_size
        actual_hidden_size = encoder_wrapper.hidden_size

        self.head = ImprovedRegressionHead(
            input_size=4 * actual_hidden_size,  # 使用真实维度
            hidden_sizes=[512, 256, 128],
            n_outputs=3,
            dropout=0.2,
            use_multi_head=use_multi_head
        )

    def forward(self, src, mask):
        feats = self.encoder_wrapper.encode_batch(src, mask)
        feats = feats.to(next(self.head.parameters()).device)
        return self.head(feats)


# ================== 数据准备 ==================
def prepare_data_with_log_transform(csv_path, vocab_pkl, seq_len=100):
    df = pd.read_csv(csv_path)
    print(f"原始样本数: {len(df)}")

    # 数据侧过滤空列
    from utils import split
    valid_indices = []
    empty_count = 0

    for idx, row in df.iterrows():
        smiles_str = str(row['processed_smiles']).strip()
        if not smiles_str or smiles_str == 'nan':
            empty_count += 1
            continue

        try:
            tokens = split(smiles_str).split()
            if len(tokens) == 0:
                empty_count += 1
                continue
            valid_indices.append(idx)
        except:
            empty_count += 1
            continue

    df = df.iloc[valid_indices].reset_index(drop=True)
    print(f"过滤掉 {empty_count} 个空/无效样本")
    print(f"有效样本数: {len(df)}")

    # 统计缺失情况
    print(f"\n标签完整性统计:")
    print(f"  lam_abs: {(~df['lam_abs'].isna()).sum()}/{len(df)} 有效")
    print(f"  lam_em:  {(~df['lam_em'].isna()).sum()}/{len(df)} 有效")
    print(f"  eps:     {(~df['eps'].isna()).sum()}/{len(df)} 有效")

    df['eps_log'] = np.log1p(df['eps'])

    processed_csv = csv_path.replace('.csv', '_processed.csv')
    df[['processed_smiles', 'lam_abs', 'lam_em', 'eps_log']].to_csv(
        processed_csv, index=False
    )

    full_ds = SmilesMultiRegDataset(
        csv_path=processed_csv,
        smiles_col='processed_smiles',
        label_cols=['lam_abs', 'lam_em', 'eps_log'],
        vocab=vocab_pkl,
        seq_len=seq_len
    )

    return full_ds, df


# ================== 修复8&9: 训练函数使用scaler_list + 掩码损失 ==================
def train_epoch(model, dataloader, optimizer, scaler_list, device,
                use_smoothl1=True, grad_clip=1.0):
    model.train()
    total_loss = 0

    for src, mask, labels in dataloader:
        src = src.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        # 逐维标准化（缺失保留为NaN）
        labels_np = labels.cpu().numpy()
        labels_scaled = np.full_like(labels_np, np.nan, dtype=np.float32)
        for i, s in enumerate(scaler_list):
            col = labels_np[:, i]
            valid = np.isfinite(col)
            if valid.any():
                labels_scaled[valid, i] = s.transform(col[valid].reshape(-1, 1)).ravel()

        labels_scaled = torch.from_numpy(labels_scaled).to(device, non_blocking=True)

        optimizer.zero_grad()
        preds = model(src, mask)

        # 掩码化的逐元素损失：NaN处不计入损失
        target_filled = torch.nan_to_num(labels_scaled, nan=0.0)
        if use_smoothl1:
            loss_elem = F.smooth_l1_loss(preds, target_filled, reduction='none')
        else:
            loss_elem = F.mse_loss(preds, target_filled, reduction='none')

        valid_mask = torch.isfinite(labels_scaled).float()
        denom = valid_mask.sum().clamp(min=1.0)
        loss = (loss_elem * valid_mask).sum() / denom

        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, scaler_list, device):
    """
    修复: 逐维评估,只对有标签的样本计算指标
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for src, mask, labels in dataloader:
            src = src.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # 逐维标准化
            labels_scaled = torch.zeros_like(labels)
            for i, scaler in enumerate(scaler_list):
                col = labels[:, i].cpu().numpy()
                valid_mask = np.isfinite(col)
                if valid_mask.any():
                    col_scaled = np.full_like(col, np.nan)
                    col_scaled[valid_mask] = scaler.transform(col[valid_mask].reshape(-1, 1)).flatten()
                    labels_scaled[:, i] = torch.from_numpy(col_scaled)

            labels_scaled = labels_scaled.to(device, non_blocking=True)
            preds_scaled = model(src, mask).cpu().numpy()
            labels_np = labels.cpu().numpy()

            # 逐维反标准化
            preds_original = np.full_like(preds_scaled, np.nan)
            for i, scaler in enumerate(scaler_list):
                col_pred = preds_scaled[:, i]
                col_label = labels_np[:, i]
                valid_mask = np.isfinite(col_label)
                if valid_mask.any():
                    preds_original[valid_mask, i] = scaler.inverse_transform(
                        col_pred[valid_mask].reshape(-1, 1)
                    ).flatten()

            all_preds.append(preds_original)
            all_labels.append(labels_np)

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    # 逐维计算指标
    rmse, mae, r2 = [], [], []
    dim_names = ['lam_abs', 'lam_em', 'eps_log']

    for i in range(3):
        pred_col = all_preds[:, i]
        label_col = all_labels[:, i]
        valid_mask = np.isfinite(label_col) & np.isfinite(pred_col)
        n_valid = valid_mask.sum()

        if n_valid > 0:
            pred_valid = pred_col[valid_mask]
            label_valid = label_col[valid_mask]

            rmse_i = np.sqrt(((pred_valid - label_valid) ** 2).mean())
            mae_i = np.abs(pred_valid - label_valid).mean()
            try:
                r2_i = r2_score(label_valid, pred_valid)
            except:
                r2_i = 0.0

            rmse.append(rmse_i)
            mae.append(mae_i)
            r2.append(r2_i)
        else:
            print(f"  ⚠ {dim_names[i]} 本折无有效标签,跳过该维指标")
            rmse.append(np.nan)
            mae.append(np.nan)
            r2.append(np.nan)

    return np.array(rmse), np.array(mae), r2, all_preds, all_labels


# ================== 主训练 ==================
def train_model(model, train_loader, val_loader, scaler_list, config, fold=None):
    device = config.DEVICE
    model = model.to(device)

    if config.FINETUNE_ENCODER:
        # 修复1: 微调时设置encoder为训练模式
        encoder_params = []
        num_layers = len(model.encoder_wrapper.model.trfm.encoder.layers)
        for i in range(num_layers - config.FINETUNE_LAYERS, num_layers):
            layer = model.encoder_wrapper.model.trfm.encoder.layers[i]
            layer.train()  # 设为训练模式
            encoder_params.extend(layer.parameters())

        optimizer = torch.optim.AdamW([
            {'params': encoder_params, 'lr': config.ENCODER_LR},
            {'params': model.head.parameters(), 'lr': config.INITIAL_LR}
        ], weight_decay=config.WEIGHT_DECAY)

        for param in encoder_params:
            param.requires_grad = True
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.INITIAL_LR,
            weight_decay=config.WEIGHT_DECAY
        )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, verbose=True
    )

    best_val_rmse = float('inf')
    patience_counter = 0
    best_model_state = None

    history = {
        'train_loss': [], 'val_rmse': [], 'val_mae': [], 'val_r2': [], 'lr': []
    }

    fold_str = f"Fold {fold}" if fold is not None else "Training"

    for epoch in range(1, config.N_EPOCHS + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler_list, device,
            config.USE_SMOOTHL1, config.GRAD_CLIP
        )

        val_rmse, val_mae, val_r2, _, _ = evaluate(
            model, val_loader, scaler_list, device
        )
        # 计算平均RMSE (忽略NaN维度)
        avg_val_rmse = np.nanmean(val_rmse)

        scheduler.step(avg_val_rmse)
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['val_rmse'].append(val_rmse)
        history['val_mae'].append(val_mae)
        history['val_r2'].append(val_r2)
        history['lr'].append(current_lr)

        if epoch % 10 == 0 or epoch == 1:
            print(f"{fold_str} Epoch {epoch}/{config.N_EPOCHS}")
            print(f"  Loss: {train_loss:.4f} | Val RMSE: {avg_val_rmse:.4f}")
            print(f"  RMSE [abs, em, eps_log]: {val_rmse}")
            print(f"  R²: {val_r2} | LR: {current_lr:.2e}")

        if avg_val_rmse < best_val_rmse:
            best_val_rmse = avg_val_rmse
            patience_counter = 0
            best_model_state = deepcopy(model.state_dict())

            save_name = f'best_regressor_fold{fold}.pth' if fold else 'best_regressor.pth'
            torch.save(best_model_state, os.path.join(config.CHECKPOINT_DIR, save_name))

            if epoch % 10 == 0:
                print(f"  ✓ 新最佳 (RMSE: {best_val_rmse:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"\n早停! 最佳RMSE: {best_val_rmse:.4f}")
                break

    model.load_state_dict(best_model_state)
    return model, history, best_val_rmse


# ================== 修复3&6&8: K折独立encoder + eps原始尺度 + scaler_list ==================
def k_fold_training(full_dataset, vocab, pretrained_model_path, config):
    """修复3: 每折独立加载encoder"""
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

        # Fit scaler逐维 (只用非缺失样本)
        train_labels = []
        for _, _, labels in train_loader:
            train_labels.append(labels.cpu().numpy())
        Y_train = np.vstack(train_labels)

        scaler_list = []
        for i, dim_name in enumerate(['lam_abs', 'lam_em', 'eps_log']):
            col = Y_train[:, i]
            valid_mask = np.isfinite(col)
            n_valid = valid_mask.sum()

            if n_valid > 0:
                scaler_i = StandardScaler()
                scaler_i.fit(col[valid_mask].reshape(-1, 1))
                scaler_list.append(scaler_i)
                print(f"  {dim_name}: {n_valid}/{len(col)} 有效样本用于fit scaler")
            else:
                # 无有效样本,创建虚拟scaler
                scaler_i = StandardScaler()
                scaler_i.mean_ = np.array([0.0])
                scaler_i.scale_ = np.array([1.0])
                scaler_list.append(scaler_i)
                print(f"  ⚠ {dim_name}: 0 有效样本,使用虚拟scaler")

        scaler_path = os.path.join(config.CHECKPOINT_DIR, f'scaler_fold{fold}.pkl')
        joblib.dump(scaler_list, scaler_path)

        # 修复3: 每折重新加载独立的encoder副本
        from pretrain_trfm import TrfmSeq2seq
        fresh_encoder = TrfmSeq2seq(
            in_size=len(vocab),
            hidden_size=config.HIDDEN_SIZE,
            out_size=len(vocab),
            n_layers=config.N_LAYERS,
            dropout=config.DROPOUT
        )
        state = torch.load(pretrained_model_path, map_location='cpu', weights_only=False)
        fresh_encoder.load_state_dict(state)
        fresh_encoder.eval()

        encoder_wrapper = ImprovedEncoder(
            fresh_encoder,
            device=config.DEVICE,
            enable_grad=config.FINETUNE_ENCODER
        )
        model = TrfmRegressor(encoder_wrapper, use_multi_head=config.USE_MULTI_HEAD)

        # 修复8: 传入scaler_list而非scaler
        model, history, best_rmse = train_model(
            model, train_loader, val_loader, scaler_list, config, fold=fold
        )

        # 评估
        val_rmse, val_mae, val_r2, val_preds, val_labels = evaluate(
            model, val_loader, scaler_list, config.DEVICE
        )

        # 修复6: K折也输出eps原始尺度
        eps_log_preds = val_preds[:, 2]
        eps_log_true = val_labels[:, 2]
        valid_eps = np.isfinite(eps_log_true) & np.isfinite(eps_log_preds)

        if valid_eps.any():
            eps_preds = np.expm1(eps_log_preds[valid_eps])
            eps_true = np.expm1(eps_log_true[valid_eps])
            eps_rmse_orig = np.sqrt(((eps_preds - eps_true) ** 2).mean())
            eps_r2_orig = r2_score(eps_true, eps_preds)
        else:
            eps_rmse_orig = np.nan
            eps_r2_orig = np.nan

        fold_results.append({
            'fold': fold,
            'best_rmse': best_rmse,
            'final_rmse': val_rmse,
            'final_mae': val_mae,
            'final_r2': val_r2,
            'eps_rmse_original': eps_rmse_orig,
            'eps_r2_original': eps_r2_orig
        })

        print(f"\nFold {fold} 结果:")
        print(f"  RMSE [abs, em, eps_log]: {val_rmse}")
        print(f"  R²: {val_r2}")
        if not np.isnan(eps_rmse_orig):
            print(f"  EPS原始尺度 - RMSE: {eps_rmse_orig:.1f}, R²: {eps_r2_orig:.4f}")

    # 汇总
    print(f"\n{'=' * 60}")
    print("K折交叉验证汇总:")
    print(f"{'=' * 60}")
    avg_rmse = np.nanmean([r['final_rmse'] for r in fold_results], axis=0)
    std_rmse = np.nanstd([r['final_rmse'] for r in fold_results], axis=0)

    valid_eps_rmse = [r['eps_rmse_original'] for r in fold_results if not np.isnan(r['eps_rmse_original'])]
    if valid_eps_rmse:
        avg_eps_rmse = np.mean(valid_eps_rmse)
        std_eps_rmse = np.std(valid_eps_rmse)
        print(f"平均RMSE [abs, em, eps_log]: {avg_rmse}")
        print(f"标准差: {std_rmse}")
        print(f"EPS原始尺度 - 平均RMSE: {avg_eps_rmse:.1f} ± {std_eps_rmse:.1f}")

    return fold_results


# ================== 修复4: 分词一致性的打分器 ==================
def score_smiles(smiles_list, model, scaler_list, vocab, config, given_props=None):
    """
    修复4: 确保分词与训练时一致
    """
    # 使用与训练时相同的分词方法
    processed_smiles = []
    for sm in smiles_list:
        # 使用utils.split进行分词 (与SmilesMultiRegDataset一致)
        sm_spaced = split(sm)
        processed_smiles.append(sm_spaced)

    # 创建临时CSV
    temp_csv = 'temp_scoring.csv'
    pd.DataFrame({
        'processed_smiles': processed_smiles,  # 已分词的SMILES
        'lam_abs': 0, 'lam_em': 0, 'eps_log': 0
    }).to_csv(temp_csv, index=False)

    ds = SmilesMultiRegDataset(
        temp_csv, 'processed_smiles',
        ['lam_abs', 'lam_em', 'eps_log'],
        vocab, config.SEQ_LEN
    )
    loader = DataLoader(
        ds, batch_size=32, shuffle=False,
        collate_fn=SmilesMultiRegDataset.collate_fn
    )

    model.eval()
    all_preds = []

    with torch.no_grad():
        for src, mask, _ in loader:
            src = src.to(config.DEVICE, non_blocking=True)
            mask = mask.to(config.DEVICE, non_blocking=True)
            preds_scaled = model(src, mask).cpu().numpy()

            # 逐维反标准化
            preds_original = np.full_like(preds_scaled, np.nan, dtype=np.float32)
            for i, s in enumerate(scaler_list):
                preds_original[:, i] = s.inverse_transform(
                    preds_scaled[:, i].reshape(-1, 1)
                ).ravel()

            all_preds.append(preds_original)

    predictions = np.vstack(all_preds)
    predictions[:, 2] = np.expm1(predictions[:, 2])  # eps_log -> eps

    # 修复5: 使用调整后的权重
    if given_props is not None:
        weights = config.SCORE_WEIGHTS
        prop_names = ['lam_abs', 'lam_em', 'eps']

        scores = np.zeros(len(smiles_list))
        for i, name in enumerate(prop_names):
            if name in given_props:
                target = given_props[name]
                errors = np.abs(predictions[:, i] - target)
                scores += weights[name] * errors
    else:
        scores = np.zeros(len(smiles_list))

    os.remove(temp_csv)
    return scores, predictions


# ================== 主执行 ==================
if __name__ == "__main__":
    print("=" * 60)
    print("最终修复版训练")
    print("=" * 60)

    print("\n[1] 加载词汇表...")
    vocab = WordVocab.load_vocab(Config.VOCAB_PKL)
    print(f"词汇表: {len(vocab)}")

    print("\n[2] 准备数据...")
    full_ds, df_original = prepare_data_with_log_transform(
        Config.CSV_PATH, Config.VOCAB_PKL, Config.SEQ_LEN
    )

    if Config.USE_KFOLD:
        print(f"\n[3] {Config.N_FOLDS}折交叉验证...")
        results = k_fold_training(
            full_ds, vocab, Config.ENCODER_PTH, Config
        )
    else:
        print("\n[3] 单次训练...")
        from pretrain_trfm import TrfmSeq2seq

        pretrained_model = TrfmSeq2seq(
            in_size=len(vocab),
            hidden_size=Config.HIDDEN_SIZE,
            out_size=len(vocab),
            n_layers=Config.N_LAYERS,
            dropout=Config.DROPOUT
        )
        state = torch.load(Config.ENCODER_PTH, map_location='cpu', weights_only=False)
        pretrained_model.load_state_dict(state)
        pretrained_model.eval()

        n = len(full_ds)
        train_n = int(n * 0.8)
        val_n = n - train_n

        train_ds, val_ds = torch.utils.data.random_split(
            full_ds, [train_n, val_n],
            generator=torch.Generator().manual_seed(42)
        )

        train_loader = DataLoader(
            train_ds, batch_size=Config.BATCH_SIZE, shuffle=True,
            collate_fn=SmilesMultiRegDataset.collate_fn,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY
        )
        val_loader = DataLoader(
            val_ds, batch_size=Config.BATCH_SIZE, shuffle=False,
            collate_fn=SmilesMultiRegDataset.collate_fn,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY
        )

        # 聚合标签
        train_labels = []
        for _, _, labels in train_loader:
            train_labels.append(labels.cpu().numpy())
        Y_train = np.vstack(train_labels)

        # 逐维fit scaler
        scaler_list = []
        for i, dim_name in enumerate(['lam_abs', 'lam_em', 'eps_log']):
            col = Y_train[:, i]
            valid = np.isfinite(col)
            if valid.any():
                s = StandardScaler().fit(col[valid].reshape(-1, 1))
            else:
                s = StandardScaler()
                s.mean_ = np.array([0.0])
                s.scale_ = np.array([1.0])
            scaler_list.append(s)
        joblib.dump(scaler_list, Config.SCALER_PKL)

        encoder_wrapper = ImprovedEncoder(
            pretrained_model,
            device=Config.DEVICE,
            enable_grad=Config.FINETUNE_ENCODER
        )
        regressor = TrfmRegressor(encoder_wrapper, use_multi_head=Config.USE_MULTI_HEAD)

        regressor, history, best_rmse = train_model(
            regressor, train_loader, val_loader, scaler_list, Config
        )

        val_rmse, val_mae, val_r2, val_preds, val_labels = evaluate(
            regressor, val_loader, scaler_list, Config.DEVICE
        )

        print("\n" + "=" * 60)
        print("最终结果:")
        print("=" * 60)
        print(f"RMSE [abs, em, eps_log]: {val_rmse}")
        print(f"R²: {val_r2}")

        valid_eps = np.isfinite(val_labels[:, 2])
        if valid_eps.any():
            eps_preds = np.expm1(val_preds[valid_eps, 2])
            eps_true = np.expm1(val_labels[valid_eps, 2])
            eps_rmse = np.sqrt(((eps_preds - eps_true) ** 2).mean())
            eps_r2 = r2_score(eps_true, eps_preds)
            print(f"\nEPS原始尺度 - RMSE: {eps_rmse:.1f}, R²: {eps_r2:.4f}")

        print("\n打分器示例:")
        test_smiles = ["CC(C)CC1=CC=C(C=C1)C"]
        scores, preds = score_smiles(
            test_smiles, regressor, scaler_list, vocab, Config,
            given_props={'lam_abs': 700}
        )
        print(f"预测: abs={preds[0, 0]:.1f}, em={preds[0, 1]:.1f}, eps={preds[0, 2]:.0f}")
        print(f"分数: {scores[0]:.2f}")

        print(f"\n完成! 模型: {Config.CHECKPOINT_DIR}")