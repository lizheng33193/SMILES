import argparse
import pandas as pd
from tqdm import tqdm

from utils import split

def main():
    parser = argparse.ArgumentParser(description='Build a corpus file')
    parser.add_argument('--in_path', '-i', type=str, default=r'D:\workspace\pycharm\smiles-transformer\data\model_v2_datasets\data_198\data_with_processed\train\data_train.csv', help='input file')
    parser.add_argument('--out_path', '-o', type=str, default=r'D:\workspace\pycharm\smiles-transformer\data\model_v2_datasets\data_198\data_with_processed\train\vocab.txt', help='output file')
    args = parser.parse_args()

    smiles = pd.read_csv(args.in_path)['processed_smiles'].values
    with open(args.out_path, 'a') as f:
        for sm in tqdm(smiles):
            f.write(split(sm)+'\n')
    print('Built a corpus file!')

if __name__=='__main__':
    main()



