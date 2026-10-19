from utils import split
import pandas as pd
import pickle
from build_vocab import WordVocab

def build_tokenizer_vocab(smiles_csv=r'D:\workspace\pycharm\smiles-transformer\data\WPT,AJ,ZYM\datasets\data.csv', smiles_col='processed_smiles', vocab_path=r'D:\workspace\pycharm\smiles-transformer\data\WPT,AJ,ZYM\test\vocab_split.pkl'):
    corpus_file = r'D:\workspace\pycharm\smiles-transformer\data\WPT,AJ,ZYM\test\corpus_split.txt'
    df = pd.read_csv(smiles_csv)
    with open(corpus_file, 'w') as f:
        for sm in df[smiles_col].astype(str):
            f.write(split(sm) + '\n')
    vocab = WordVocab(corpus_file, min_freq=1)
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print(f"Tokenizer vocab saved to: {vocab_path} (size={len(vocab)})")

if __name__ == '__main__':
    build_tokenizer_vocab()
