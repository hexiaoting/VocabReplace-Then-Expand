import argparse
import sentencepiece as spm
import pandas as pd
import os

from tokenizers import ByteLevelBPETokenizer

parser = argparse.ArgumentParser()
parser.add_argument('-input_path',type=str,required=True)
parser.add_argument('-dataset',type=str,required=True)
parser.add_argument('-dump_dir', required=True, type=str,default='VocabFiles') 
parser.add_argument('-column', type=str, default='Data', help='Column name in CSV file containing text data')
args = parser.parse_args()

print(f'\n------------------------------\nStarting for {args.dataset}...')

if args.input_path.endswith('.txt'):
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(args.input_path,vocab_size=256000,show_progress=True)
    tokenizer.save_model(args.dump_dir)

elif args.input_path.endswith('.csv'):
    df = pd.read_csv(args.input_path)
    df = df.dropna()
    df = df[df.iloc[:,0].str.strip() != '']
    
    texts = df.iloc[:,0].tolist()
    texts_freq_normalized = []
    
    for idx,item in enumerate(texts):
        count = df.iloc[idx,1]
        texts_freq_normalized.extend([item] * count)
    
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(texts_freq_normalized,vocab_size=256000,show_progress=True)
    tokenizer.save_model(args.dump_dir)

elif args.input_path.endswith('.jsonl'):
    df = pd.read_json(args.input_path, lines=True)
    df = df.dropna()
    df = df[df[args.column].str.strip() != '']
    
    texts = df[args.column].tolist()
    
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(texts,vocab_size=256000,show_progress=True)
    tokenizer.save_model(args.dump_dir)