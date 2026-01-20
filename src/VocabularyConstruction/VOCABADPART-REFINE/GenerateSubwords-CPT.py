#!/usr/bin/env python
import argparse
from transformers import AutoTokenizer
import json
import pandas as pd
import os
import re
import string
from pathlib import Path          # optional, just nice syntax

parser = argparse.ArgumentParser() 
parser.add_argument('-dataset',default='CPT',type=str)
parser.add_argument('-split_more_than',default=1,type=int) 
parser.add_argument('-v_size',type=int,required=True)
parser.add_argument('-vpath',type=str,required=True)
parser.add_argument('-PAC_path',type=str,required=True)
parser.add_argument('-model_id',type=str,required=True) 
args = parser.parse_args()

print(f'\n------------------------------\nStarting for {args.dataset} {args.v_size//1000}K...')

tok = AutoTokenizer.from_pretrained(args.model_id)
org_vocab = tok.get_vocab()

list_PM_All = list()
vocab_PAC = json.load(open(args.PAC_path))
for term,idx in vocab_PAC.items():
        if idx < 256: continue
        if term in org_vocab: continue
        if len(tok.tokenize(term)) == 1: continue
        list_PM_All.append(term)

print('V_PAC not in V_PLM',len(list_PM_All))

PAC_topK = [x for x in list_PM_All if len(tok.tokenize(x.replace("Ġ",' '))) > args.split_more_than]

pattern = r"^[A-Za-zĠ]+$"
del_idx = list()
for idx,v in enumerate(PAC_topK):
    if re.match(pattern,v.strip().split()[0]): 
        continue
    else: del_idx.append(idx)

for idx in del_idx[::-1]: del PAC_topK[idx]

allowed = set(string.ascii_letters + string.digits + string.punctuation)
allowed.add("Ġ")                  # keep the token marker – remove if you don’t need it

PAC_topK = [x for x in PAC_topK if all(ch in allowed for ch in x)]
PAC_topK = PAC_topK[:args.v_size]

if not os.path.exists(f'{args.vpath}'): os.makedirs(f'{args.vpath}')

dump_path = f'{args.vpath}/{args.v_size/1000}_.txt'

with open(dump_path,'w') as f:
    for word in PAC_topK:
        f.write(f"{word}\n")
f.close()

print(f'FINAL Vocab size Dump: {len(PAC_topK)}')
