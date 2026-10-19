"""
数据和模型诊断脚本
用于检查训练数据质量和模型行为
"""

import pandas as pd
import numpy as np
import torch
from build_vocab import WordVocab
from dataset import SmilesMultiRegDataset, Randomizer
from utils import split


def diagnose_dataset(csv_path, vocab_pkl):
    """诊断数据集问题"""
    print("=" * 60)
    print("数据集诊断")
    print("=" * 60)

    df = pd.read_csv(csv_path)
    vocab = WordVocab.load_vocab(vocab_pkl)

    print(f"\n1. 基本统计:")
    print(f"   总样本数: {len(df)}")
    print(f"   词汇表大小: {len(vocab)}")

    print(f"\n2. 标签分布:")
    for col in ['lam_abs', 'lam_em', 'eps']:
        if col in df.columns:
            valid = df[col].notna()
            print(f"   {col}:")
            print(f"     - 有效样本: {valid.sum()}/{len(df)}")
            print(f"     - 均值: {df[col].mean():.2f}")
            print(f"     - 标准差: {df[col].std():.2f}")
            print(f"     - 范围: [{df[col].min():.2f}, {df[col].max():.2f}]")

    print(f"\n3. SMILES分词检查:")
    randomizer = Randomizer()
    no_transform = lambda x: x.split() if isinstance(x, str) else x

    # 检查前5个样本
    for i in range(min(5, len(df))):
        sm = str(df.iloc[i]['processed_smiles'])

        # 原始分词
        tokens_orig = sm.split()

        # Randomizer分词
        tokens_rand = randomizer(sm)

        # 检查差异
        if tokens_orig != tokens_rand:
            print(f"\n   样本 {i} - Randomizer改变了结构!")
            print(f"     原始: {' '.join(tokens_orig[:10])}...")
            print(f"     随机: {' '.join(tokens_rand[:10])}...")

        # 检查token是否在vocab中
        unk_count = sum(1 for tok in tokens_orig if tok not in vocab.stoi)
        if unk_count > 0:
            print(f"   样本 {i}: {unk_count}/{len(tokens_orig)} 未知tokens")

    print(f"\n4. 数据质量检查:")
    # 检查空SMILES
    empty_smiles = df['processed_smiles'].isna() | (df['processed_smiles'] == '')
    print(f"   空SMILES: {empty_smiles.sum()}")

    # 检查全缺失标签
    all_na = df[['lam_abs', 'lam_em', 'eps']].isna().all(axis=1)
    print(f"   全缺失标签样本: {all_na.sum()}")

    # 检查异常值
    for col in ['lam_abs', 'lam_em', 'eps']:
        if col in df.columns:
            q99 = df[col].quantile(0.99)
            outliers = df[col] > q99 * 3
            if outliers.any():
                print(f"   {col}异常值(>3*P99): {outliers.sum()}")


def diagnose_model_output(model_path, vocab_pkl, test_smiles):
    """诊断模型输出"""
    print("\n" + "=" * 60)
    print("模型输出诊断")
    print("=" * 60)

    # 加载模型和scaler
    import joblib
    try:
        model = torch.load(model_path, map_location='cpu')
        print(f"\n✓ 成功加载模型: {model_path}")
    except Exception as e:
        print(f"\n✗ 加载模型失败: {e}")
        return

    # 测试不同输入
    print(f"\n测试输入变化对输出的影响:")

    from dataset import SmilesMultiRegDataset
    vocab = WordVocab.load_vocab(vocab_pkl)

    # 测试1: 相同输入多次
    print(f"\n1. 相同输入重复3次:")
    for i in range(3):
        # 创建临时dataset
        temp_csv = 'temp_test.csv'
        pd.DataFrame({
            'processed_smiles': test_smiles[:1],
            'lam_abs': 0, 'lam_em': 0, 'eps': 0
        }).to_csv(temp_csv, index=False)

        ds = SmilesMultiRegDataset(temp_csv, 'processed_smiles',
                                   ['lam_abs', 'lam_em', 'eps'],
                                   vocab, 100)

        src, mask, _ = ds[0]
        print(f"   运行 {i + 1}: src[:10] = {src[:10].tolist()}")

    # 测试2: 不同SMILES
    print(f"\n2. 不同SMILES输入:")
    for sm in test_smiles:
        print(f"   {sm[:30]}... -> 检查tokenization")

    print(f"\n提示: 如果相同输入的src不同,说明Randomizer在起作用!")


def check_scaler_sanity(scaler_path):
    """检查scaler是否合理"""
    print("\n" + "=" * 60)
    print("Scaler诊断")
    print("=" * 60)

    import joblib
    try:
        scaler_list = joblib.load(scaler_path)
        print(f"\n✓ 成功加载scaler: {scaler_path}")

        for i, (name, scaler) in enumerate(zip(['lam_abs', 'lam_em', 'eps'], scaler_list)):
            print(f"\n{name}:")
            print(f"  mean: {scaler.mean_[0]:.2f}")
            print(f"  std: {scaler.scale_[0]:.2f}")

            # 检查是否是虚拟scaler
            if scaler.mean_[0] == 0 and scaler.scale_[0] == 1:
                print(f"  ⚠ 这是虚拟scaler，可能该维度无有效训练数据!")

    except Exception as e:
        print(f"\n✗ 加载scaler失败: {e}")


if __name__ == "__main__":
    # 配置路径
    CSV_PATH = r"D:\workspace\pycharm\smiles-transformer\data\model_v2_datasets\data_198\data_with_processed\train\data_train.csv"
    VOCAB_PKL = r"D:\workspace\pycharm\smiles-transformer\data\model_v2_datasets\data_198\data_with_processed\train\vocab.pkl"

    # 运行诊断
    diagnose_dataset(CSV_PATH, VOCAB_PKL)

    # 测试SMILES
    test_smiles = [
        "CN(C)C1=CC=C(C=C1)C2=CC(=C3N=C4C(=CC(=[N+]4[B-](F)(F)N23)C5=CC=CC=C5)C6=CC=C(C=C6)CN7CCOCC7)C8=CC=C(C=C8)CN9CCOCC9",
        "CCN(CC)C1=CC=C(C=C1)C2=N/C(=N\\C3=C(C4=CC=CC=C4)C5=C(C6=C(CC5)C=C(C=C6)OC)N3[B-](F)(F)F)/C(=C2)C7=CC=CC=C7"
    ]

    # 如果模型已训练，可以取消注释
    # diagnose_model_output(
    #     "output/improved_model/checkpoints/best_regressor_fold1.pth",
    #     VOCAB_PKL,
    #     test_smiles
    # )

    # 如果scaler已保存，可以取消注释
    # check_scaler_sanity("output/improved_model/checkpoints/scaler_fold1.pkl")