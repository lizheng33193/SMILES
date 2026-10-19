"""
推理脚本 - 用于测试训练好的模型
支持单个样本预测和批量测试集预测
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
from pathlib import Path
import argparse


class Config:
    BASE_DIR = Path(__file__).resolve().parents[1]
    DATA_DIR = os.path.join(BASE_DIR, "data", "model_v2_datasets", "data_198", "data_with_processed")
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    TEST_DIR = os.path.join(DATA_DIR, "test")
    OUTPUT_DIR = os.path.join(BASE_DIR, "output", "improved_model_fixed")
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
    RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")

    VOCAB_PKL = os.path.join(TRAIN_DIR, "vocab.pkl")
    ENCODER_PTH = os.path.join(BASE_DIR, "model_pretrain_trfm", ".save", "trfm_new_9_0.pkl")
    TEST_CSV = os.path.join(TEST_DIR, "data_test.csv")

    HIDDEN_SIZE = 256
    N_LAYERS = 4
    DROPOUT = 0.1
    SEQ_LEN = 250
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    PIN_MEMORY = True

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


os.makedirs(Config.RESULTS_DIR, exist_ok=True)

from build_vocab import WordVocab
from dataset import SmilesMultiRegDataset


class NoOpTransform:
    """用于已分词的 processed_smiles"""
    def __call__(self, sm):
        if isinstance(sm, str):
            return sm.split()
        return sm


class OriginalStyleEncoder:
    """原始编码器"""
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


def load_model(model_path, scaler_path, vocab, encoder_path, device):
    """加载训练好的模型"""
    print(f"\n加载模型: {model_path}")

    # 加载预训练encoder
    from pretrain_trfm import TrfmSeq2seq
    encoder = TrfmSeq2seq(
        in_size=len(vocab),
        hidden_size=Config.HIDDEN_SIZE,
        out_size=len(vocab),
        n_layers=Config.N_LAYERS,
        dropout=Config.DROPOUT
    )
    state = torch.load(encoder_path, map_location='cpu', weights_only=False)
    encoder.load_state_dict(state)

    # 创建模型
    encoder_wrapper = OriginalStyleEncoder(encoder, device=device)
    model = TrfmRegressor(encoder_wrapper)

    # 加载微调后的权重
    model_state = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    # 加载scaler
    scaler_list = joblib.load(scaler_path)

    print(f"✓ 模型加载成功")
    return model, scaler_list


def predict_batch(model, dataloader, scaler_list, device):
    """批量预测"""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for src, mask, labels in dataloader:
            src = src.to(device)
            labels_np = labels.numpy().copy()

            # 预测（在标准化的log空间）
            preds_scaled = model(src, mask).cpu().numpy()

            # 反标准化到log空间
            preds_log = np.full_like(preds_scaled, np.nan)
            for i, scaler in enumerate(scaler_list):
                preds_log[:, i] = scaler.inverse_transform(
                    preds_scaled[:, i].reshape(-1, 1)
                ).ravel()

            # eps从log空间还原到原始空间
            preds_original = preds_log.copy()
            preds_original[:, 2] = np.power(10.0, preds_log[:, 2])

            all_preds.append(preds_original)
            all_labels.append(labels_np)

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    return all_preds, all_labels


def calculate_metrics(y_true, y_pred, target_names=['lam_abs', 'lam_em', 'eps']):
    """计算评估指标"""
    metrics = {}

    for i, name in enumerate(target_names):
        pred_col = y_pred[:, i]
        true_col = y_true[:, i]

        # 只计算有效值
        valid = np.isfinite(true_col) & np.isfinite(pred_col)

        if valid.sum() > 0:
            p, t = pred_col[valid], true_col[valid]

            metrics[name] = {
                'RMSE': np.sqrt(mean_squared_error(t, p)),
                'MAE': mean_absolute_error(t, p),
                'R2': r2_score(t, p),
                'n_samples': valid.sum()
            }
        else:
            metrics[name] = {
                'RMSE': np.nan,
                'MAE': np.nan,
                'R2': np.nan,
                'n_samples': 0
            }

    return metrics


def predict_single_smiles(smiles, model, scaler_list, vocab, device, seq_len=220):
    """预测单个SMILES"""
    # 创建临时数据集（这里假设已经是processed_smiles格式）
    tokens = smiles.split() if isinstance(smiles, str) else smiles

    # 转换为indices
    indices = [vocab.stoi.get(token, vocab.unk_index) for token in tokens]
    indices = indices[:seq_len]  # 截断

    # Padding
    if len(indices) < seq_len:
        indices += [vocab.pad_index] * (seq_len - len(indices))

    # 转换为tensor
    src = torch.tensor([indices], dtype=torch.long).to(device)

    # 预测
    model.eval()
    with torch.no_grad():
        pred_scaled = model(src, None).cpu().numpy()

    # 反标准化
    pred_log = np.zeros(3)
    for i, scaler in enumerate(scaler_list):
        pred_log[i] = scaler.inverse_transform(pred_scaled[0, i].reshape(-1, 1))[0, 0]

    # eps还原
    pred_original = pred_log.copy()
    pred_original[2] = np.power(10.0, pred_log[2])

    return {
        'lam_abs': pred_original[0],
        'lam_em': pred_original[1],
        'eps': pred_original[2]
    }


def test_on_dataset(model, scaler_list, vocab, test_csv, device, config):
    """在测试集上评估"""
    print(f"\n{'='*60}")
    print("测试集评估")
    print(f"{'='*60}")

    # 加载测试数据
    test_ds = SmilesMultiRegDataset(
        csv_path=test_csv,
        smiles_col='processed_smiles',
        label_cols=['lam_abs', 'lam_em', 'eps'],
        vocab=vocab,
        seq_len=config.SEQ_LEN,
        transform=NoOpTransform()
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=SmilesMultiRegDataset.collate_fn,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    print(f"测试样本数: {len(test_ds)}")

    # 预测
    print("\n正在预测...")
    y_pred, y_true = predict_batch(model, test_loader, scaler_list, device)

    # 计算指标
    metrics = calculate_metrics(y_true, y_pred)

    # 打印结果
    print(f"\n{'='*60}")
    print("测试结果:")
    print(f"{'='*60}")
    for name, m in metrics.items():
        print(f"\n{name}:")
        print(f"  RMSE: {m['RMSE']:.4f}")
        print(f"  MAE:  {m['MAE']:.4f}")
        print(f"  R²:   {m['R2']:.4f}")
        print(f"  样本数: {m['n_samples']}")

    # 保存预测结果
    df_test = pd.read_csv(test_csv)
    df_results = df_test.copy()
    df_results['pred_lam_abs'] = y_pred[:, 0]
    df_results['pred_lam_em'] = y_pred[:, 1]
    df_results['pred_eps'] = y_pred[:, 2]

    output_path = os.path.join(config.RESULTS_DIR, 'test_predictions.csv')
    df_results.to_csv(output_path, index=False)
    print(f"\n预测结果已保存: {output_path}")

    return metrics, df_results


def ensemble_predict(models_info, vocab, test_csv, device, config):
    """集成多个fold的模型进行预测"""
    print(f"\n{'='*60}")
    print("集成预测 (多个fold平均)")
    print(f"{'='*60}")

    # 加载测试数据
    test_ds = SmilesMultiRegDataset(
        csv_path=test_csv,
        smiles_col='processed_smiles',
        label_cols=['lam_abs', 'lam_em', 'eps'],
        vocab=vocab,
        seq_len=config.SEQ_LEN,
        transform=NoOpTransform()
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=SmilesMultiRegDataset.collate_fn,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    # 收集所有fold的预测
    all_fold_preds = []

    for fold_info in models_info:
        model, scaler_list = load_model(
            fold_info['model_path'],
            fold_info['scaler_path'],
            vocab,
            Config.ENCODER_PTH,
            device
        )

        y_pred, y_true = predict_batch(model, test_loader, scaler_list, device)
        all_fold_preds.append(y_pred)
        print(f"✓ Fold {fold_info['fold']} 预测完成")

    # 平均预测结果
    y_pred_ensemble = np.mean(all_fold_preds, axis=0)

    # 计算指标
    metrics = calculate_metrics(y_true, y_pred_ensemble)

    # 打印结果
    print(f"\n{'='*60}")
    print("集成预测结果:")
    print(f"{'='*60}")
    for name, m in metrics.items():
        print(f"\n{name}:")
        print(f"  RMSE: {m['RMSE']:.4f}")
        print(f"  MAE:  {m['MAE']:.4f}")
        print(f"  R²:   {m['R2']:.4f}")

    # 保存结果
    df_test = pd.read_csv(test_csv)
    df_results = df_test.copy()
    df_results['pred_lam_abs'] = y_pred_ensemble[:, 0]
    df_results['pred_lam_em'] = y_pred_ensemble[:, 1]
    df_results['pred_eps'] = y_pred_ensemble[:, 2]

    output_path = os.path.join(config.RESULTS_DIR, 'test_predictions_ensemble.csv')
    df_results.to_csv(output_path, index=False)
    print(f"\n集成预测结果已保存: {output_path}")

    return metrics, df_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='模型推理脚本')
    parser.add_argument('--mode', type=str, default='single',
                        choices=['single', 'test', 'ensemble'],
                        help='推理模式: single(单fold), test(测试集), ensemble(多fold集成)')
    parser.add_argument('--fold', type=int, default=1,
                        help='使用哪个fold的模型 (mode=single时)')
    parser.add_argument('--smiles', type=str, default=None,
                        help='单个SMILES进行预测 (需要是processed格式)')

    args = parser.parse_args()

    print("="*60)
    print("模型推理脚本")
    print("="*60)

    # 加载词汇表
    print("\n加载词汇表...")
    vocab = WordVocab.load_vocab(Config.VOCAB_PKL)
    print(f"词汇表大小: {len(vocab)}")

    if args.mode == 'single':
        # 单个fold预测
        model_path = os.path.join(Config.CHECKPOINT_DIR, f'best_regressor_fold{args.fold}.pth')
        scaler_path = os.path.join(Config.CHECKPOINT_DIR, f'scaler_fold{args.fold}.pkl')

        model, scaler_list = load_model(
            model_path, scaler_path, vocab, Config.ENCODER_PTH, Config.DEVICE
        )

        if args.smiles:
            # 预测单个SMILES
            print(f"\n预测SMILES: {args.smiles[:50]}...")
            result = predict_single_smiles(
                args.smiles, model, scaler_list, vocab, Config.DEVICE, Config.SEQ_LEN
            )
            print("\n预测结果:")
            print(f"  λ_abs: {result['lam_abs']:.2f} nm")
            print(f"  λ_em:  {result['lam_em']:.2f} nm")
            print(f"  ε:     {result['eps']:.0f}")
        else:
            # 在测试集上评估
            metrics, df_results = test_on_dataset(
                model, scaler_list, vocab, Config.TEST_CSV, Config.DEVICE, Config
            )

    elif args.mode == 'test':
        # 使用单个fold在测试集上评估
        model_path = os.path.join(Config.CHECKPOINT_DIR, f'best_regressor_fold{args.fold}.pth')
        scaler_path = os.path.join(Config.CHECKPOINT_DIR, f'scaler_fold{args.fold}.pkl')

        model, scaler_list = load_model(
            model_path, scaler_path, vocab, Config.ENCODER_PTH, Config.DEVICE
        )

        metrics, df_results = test_on_dataset(
            model, scaler_list, vocab, Config.TEST_CSV, Config.DEVICE, Config
        )

    elif args.mode == 'ensemble':
        # 集成多个fold
        models_info = []
        for fold in range(1, 6):  # 假设有5个fold
            model_path = os.path.join(Config.CHECKPOINT_DIR, f'best_regressor_fold{fold}.pth')
            scaler_path = os.path.join(Config.CHECKPOINT_DIR, f'scaler_fold{fold}.pkl')

            if os.path.exists(model_path) and os.path.exists(scaler_path):
                models_info.append({
                    'fold': fold,
                    'model_path': model_path,
                    'scaler_path': scaler_path
                })

        print(f"\n找到 {len(models_info)} 个fold的模型")

        if models_info:
            metrics, df_results = ensemble_predict(
                models_info, vocab, Config.TEST_CSV, Config.DEVICE, Config
            )
        else:
            print("错误: 未找到任何训练好的模型!")

    print("\n完成!")


