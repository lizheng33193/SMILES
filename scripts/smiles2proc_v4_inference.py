"""
推理脚本 - 用于预测SMILES分子的光学性质
支持单个分子预测和批量预测
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings

warnings.filterwarnings('ignore')

import pickle
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import joblib
from pathlib import Path


class Config:
    BASE_DIR = Path(__file__).resolve().parents[1]
    DATA_DIR = os.path.join(BASE_DIR, "data", "model_v2_datasets", "data_198", "data_with_processed")
    OUTPUT_DIR = os.path.join(BASE_DIR, "output", "improved_model_v3")
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
    VOCAB_PKL = os.path.join(DATA_DIR, "train", "vocab.pkl")
    ENCODER_PTH = os.path.join(BASE_DIR, "model_pretrain_trfm", ".save", "trfm_new_9_0.pkl")

    HIDDEN_SIZE = 256
    N_LAYERS = 4
    DROPOUT = 0.1
    SEQ_LEN = 250
    BATCH_SIZE = 64
    PRESENCE_THRESHOLD = 0.5

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    """回归模型（含存在性判别头）"""

    def __init__(self, encoder_wrapper):
        super().__init__()
        D = 4 * encoder_wrapper.hidden_size
        self.encoder_wrapper = encoder_wrapper

        # 回归头
        self.reg_head = nn.Sequential(
            nn.Linear(D, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 3)  # [abs, em, log10(eps)]
        )

        # 存在性判别头
        self.presence_head = nn.Sequential(
            nn.Linear(D, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # [em_exists, eps_exists]
        )

    def forward(self, src, mask=None):
        feats = self.encoder_wrapper.encode_batch(src, mask)
        feats = feats.to(next(self.reg_head.parameters()).device)

        y_pred = self.reg_head(feats)
        presence_logits = self.presence_head(feats)

        return y_pred, presence_logits


class SMILESPredictor:
    """SMILES分子性质预测器"""

    def __init__(self, vocab_path, encoder_path, model_paths, scaler_paths, config):
        """
        Args:
            vocab_path: 词汇表路径
            encoder_path: 预训练encoder路径
            model_paths: 模型checkpoint路径列表（支持多个fold的ensemble）
            scaler_paths: scaler路径列表
            config: 配置对象
        """
        self.config = config
        self.device = config.DEVICE

        print(f"🔧 使用设备: {self.device}")

        # 加载词汇表
        print("📖 加载词汇表...")
        self.vocab = WordVocab.load_vocab(vocab_path)
        print(f"   词汇表大小: {len(self.vocab)}")

        # 加载预训练encoder
        print("🧠 加载预训练encoder...")
        from pretrain_trfm import TrfmSeq2seq
        encoder = TrfmSeq2seq(
            in_size=len(self.vocab),
            hidden_size=config.HIDDEN_SIZE,
            out_size=len(self.vocab),
            n_layers=config.N_LAYERS,
            dropout=config.DROPOUT
        )
        state = torch.load(encoder_path, map_location='cpu', weights_only=False)
        encoder.load_state_dict(state)

        # 加载回归模型（支持ensemble）
        print(f"🎯 加载回归模型 ({len(model_paths)} folds)...")
        self.models = []
        for i, model_path in enumerate(model_paths):
            encoder_wrapper = OriginalStyleEncoder(encoder, device=self.device)
            model = TrfmRegressor(encoder_wrapper)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            self.models.append(model)
            print(f"   ✓ Fold {i + 1} 模型已加载")

        # 加载scalers
        print("📊 加载标准化器...")
        self.scalers = []
        for scaler_path in scaler_paths:
            scaler_list = joblib.load(scaler_path)
            self.scalers.append(scaler_list)
        print(f"   ✓ {len(self.scalers)} 个scaler已加载")

    def predict_single(self, smiles_processed):
        """
        预测单个SMILES分子

        Args:
            smiles_processed: 已处理的SMILES字符串（用空格分隔的token）

        Returns:
            dict: {'lam_abs': float, 'lam_em': float/None, 'eps': float}
        """
        # 创建临时DataFrame
        df = pd.DataFrame({
            'processed_smiles': [smiles_processed],
            'lam_abs': [np.nan],
            'lam_em': [np.nan],
            'eps': [np.nan]
        })

        results = self.predict_batch(df)
        return results.iloc[0].to_dict()

    def predict_batch(self, df_input):
        """
        批量预测

        Args:
            df_input: DataFrame，必须包含'processed_smiles'列

        Returns:
            DataFrame: 包含预测结果的DataFrame
        """
        # 创建临时CSV（如果需要）
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_csv = f.name
            df_input.to_csv(temp_csv, index=False)

        try:
            # 创建Dataset
            dataset = SmilesMultiRegDataset(
                csv_path=temp_csv,
                smiles_col='processed_smiles',
                label_cols=['lam_abs', 'lam_em', 'eps'],
                vocab=self.vocab,
                seq_len=self.config.SEQ_LEN,
                transform=NoOpTransform()
            )

            dataloader = DataLoader(
                dataset,
                batch_size=self.config.BATCH_SIZE,
                shuffle=False,
                collate_fn=SmilesMultiRegDataset.collate_fn,
                num_workers=0,
                pin_memory=False
            )

            # 收集所有模型的预测
            all_fold_preds = []
            all_fold_presence = []

            for model, scaler_list in zip(self.models, self.scalers):
                fold_preds = []
                fold_presence = []

                with torch.no_grad():
                    for src, mask, _ in dataloader:
                        src = src.to(self.device)

                        # 预测
                        y_pred_std, presence_logits = model(src, mask)
                        y_pred_std_np = y_pred_std.cpu().numpy()
                        presence_probs = torch.sigmoid(presence_logits).cpu().numpy()

                        # 反标准化
                        preds_log = np.full_like(y_pred_std_np, np.nan)
                        for i, scaler in enumerate(scaler_list):
                            preds_log[:, i] = scaler.inverse_transform(
                                y_pred_std_np[:, i].reshape(-1, 1)
                            ).ravel()

                        # eps从log空间还原
                        preds_original = preds_log.copy()
                        preds_original[:, 2] = np.power(10.0, preds_log[:, 2])

                        fold_preds.append(preds_original)
                        fold_presence.append(presence_probs)

                all_fold_preds.append(np.vstack(fold_preds))
                all_fold_presence.append(np.vstack(fold_presence))

            # Ensemble平均
            final_preds = np.mean(all_fold_preds, axis=0)
            final_presence = np.mean(all_fold_presence, axis=0)

            # 应用存在性判别
            em_missing = final_presence[:, 0] < self.config.PRESENCE_THRESHOLD
            eps_missing = final_presence[:, 1] < self.config.PRESENCE_THRESHOLD

            final_preds[em_missing, 1] = np.nan
            final_preds[eps_missing, 2] = 1.0

            # 构建结果DataFrame
            result_df = df_input.copy()
            result_df['pred_lam_abs'] = final_preds[:, 0]
            result_df['pred_lam_em'] = final_preds[:, 1]
            result_df['pred_eps'] = final_preds[:, 2]
            result_df['em_confidence'] = final_presence[:, 0]
            result_df['eps_confidence'] = final_presence[:, 1]

            return result_df

        finally:
            # 清理临时文件
            if os.path.exists(temp_csv):
                os.remove(temp_csv)


def load_predictor(config, fold_list=None):
    """
    加载预测器

    Args:
        config: 配置对象
        fold_list: 要加载的fold列表（None表示加载所有可用的fold）

    Returns:
        SMILESPredictor实例
    """
    # 查找可用的checkpoint
    if fold_list is None:
        model_paths = sorted(Path(config.CHECKPOINT_DIR).glob("best_regressor_fold*.pth"))
        fold_list = [int(p.stem.split('fold')[-1]) for p in model_paths]
    else:
        model_paths = [
            os.path.join(config.CHECKPOINT_DIR, f"best_regressor_fold{f}.pth")
            for f in fold_list
        ]

    scaler_paths = [
        os.path.join(config.CHECKPOINT_DIR, f"scaler_fold{f}.pkl")
        for f in fold_list
    ]

    # 检查文件是否存在
    for path in model_paths + scaler_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到文件: {path}")

    print(f"📦 准备加载 {len(model_paths)} 个fold的模型进行ensemble预测\n")

    predictor = SMILESPredictor(
        vocab_path=config.VOCAB_PKL,
        encoder_path=config.ENCODER_PTH,
        model_paths=model_paths,
        scaler_paths=scaler_paths,
        config=config
    )

    return predictor


def main():
    parser = argparse.ArgumentParser(description="SMILES分子光学性质预测")
    parser.add_argument('--input', type=str, required=True,
                        help='输入CSV文件路径（必须包含processed_smiles列）')
    parser.add_argument('--output', type=str, default=None,
                        help='输出CSV文件路径（默认为input_predictions.csv）')
    parser.add_argument('--folds', type=int, nargs='+', default=None,
                        help='使用的fold编号（默认使用所有可用fold）')
    parser.add_argument('--smiles', type=str, default=None,
                        help='单个SMILES字符串（已处理，用于快速测试）')

    args = parser.parse_args()

    print("=" * 60)
    print("SMILES分子光学性质预测系统")
    print("=" * 60)

    # 加载预测器
    predictor = load_predictor(Config, fold_list=args.folds)

    print("\n" + "=" * 60)
    print("开始预测...")
    print("=" * 60 + "\n")

    # 单个SMILES预测
    if args.smiles:
        print(f"输入SMILES: {args.smiles}\n")
        result = predictor.predict_single(args.smiles)

        print("预测结果:")
        print(f"  λ_abs: {result['pred_lam_abs']:.1f} nm")
        if pd.notna(result['pred_lam_em']):
            print(f"  λ_em:  {result['pred_lam_em']:.1f} nm (置信度: {result['em_confidence']:.2%})")
        else:
            print(f"  λ_em:  无发射 (置信度: {result['em_confidence']:.2%})")
        print(f"  ε:     {result['pred_eps']:.0f} (置信度: {result['eps_confidence']:.2%})")
        return

    # 批量预测
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"输入文件不存在: {args.input}")

    print(f"📁 读取输入文件: {args.input}")
    df_input = pd.read_csv(args.input)

    if 'processed_smiles' not in df_input.columns:
        raise ValueError("输入CSV必须包含'processed_smiles'列")

    print(f"   共 {len(df_input)} 个分子待预测\n")

    # 预测
    result_df = predictor.predict_batch(df_input)

    # 保存结果
    output_path = args.output or args.input.replace('.csv', '_predictions.csv')
    result_df.to_csv(output_path, index=False)

    print(f"\n✅ 预测完成!")
    print(f"📄 结果已保存至: {output_path}")

    # 显示统计信息
    print(f"\n📊 预测统计:")
    print(f"  λ_abs范围: [{result_df['pred_lam_abs'].min():.1f}, {result_df['pred_lam_abs'].max():.1f}] nm")

    has_em = result_df['pred_lam_em'].notna().sum()
    print(f"  有发射的分子: {has_em}/{len(result_df)} ({has_em / len(result_df):.1%})")

    if has_em > 0:
        em_values = result_df['pred_lam_em'].dropna()
        print(f"  λ_em范围: [{em_values.min():.1f}, {em_values.max():.1f}] nm")

    print(f"  ε范围: [{result_df['pred_eps'].min():.0f}, {result_df['pred_eps'].max():.0f}]")


if __name__ == "__main__":
    main()


# python smiles-transformer/scripts/smiles2proc_v4_inference.py

# python smiles-transformer/scripts/smiles2proc_v4_inference.py --input smiles-transformer/data/model_v2_datasets/data_198/data_with_processed/train/data_train.csv --output smiles-transformer/output/improved_model_v3/