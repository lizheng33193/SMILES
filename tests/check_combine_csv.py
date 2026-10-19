#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

def merge_smiles(data_one_path: str,
                 image2smiles_path: str,
                 output_path: str) -> None:
    """
    读取两个 CSV 文件，根据 'number' 列合并 smiles 信息，
    并将结果保存到 output_path。
    """
    # 读取原始表格
    df1 = pd.read_csv(data_one_path)
    df2 = pd.read_csv(image2smiles_path)
    
    # 确保 df2 是以 number 为索引的
    smiles_map = df2.set_index('number')['smiles']
    
    # 将 df2 中的 smiles 覆盖到 df1 中
    # 如果 df1 中已有 smiles 列，会被覆盖；否则会新建该列
    df1['smiles'] = df1['number'].map(smiles_map)
    
    # 将合并后的 DataFrame 写出
    df1.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"已保存合并结果到: {output_path}")

if __name__ == "__main__":
    # 文件名根据实际路径修改
    data_one_csv = r"D:\workspace\pycharm\smiles-transformer\data\model_v2_datasets\data_198\data_label.csv"
    image2smiles_csv = r"D:\workspace\pycharm\smiles-transformer\data\model_v2_datasets\data_198\image2smiles_validity.csv"
    output_csv = r"D:\workspace\pycharm\smiles-transformer\data\model_v2_datasets\data_198\data.csv"
    
    merge_smiles(data_one_csv, image2smiles_csv, output_csv)
