#!/bin/bash

mkdir -p VocabFiles
mkdir -p VocabFiles/CPT-SupremeCourtCases/
mkdir -p VocabFiles/CPT-SupremeCourtCases/CPT-SupremeCourtCases/
mkdir -p VocabFiles/CPT-SupremeCourtCases/CPT-SupremeCourtCases-Qwen2.5_Vocab/
mkdir -p VocabFiles/CPT-SupremeCourtCases/CPT-SupremeCourtCases-Llama3.1_Vocab/

python GenerateVocabulary.py -input_path "../../data/legal_pretrain_train_100M_tokens.txt" \
                             -dataset CPT-SupremeCourtCases \
                             -dump_dir VocabFiles/CPT-SupremeCourtCases/CPT-SupremeCourtCases/

for v_size in 5000 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000
do
    python GenerateSubwords-CPT.py -v_size $v_size \
                                 -vpath ./VocabFiles/CPT-SupremeCourtCases/CPT-SupremeCourtCases-Llama3.1_Vocab/ \
                                 -PAC_path ./VocabFiles/CPT-SupremeCourtCases/CPT-SupremeCourtCases/vocab.json \
                                 -model_id meta-llama/Llama-3.1-8B
done

for v_size in 5000 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000
do
    python GenerateSubwords-CPT.py -v_size $v_size \
                                 -vpath ./VocabFiles/CPT-SupremeCourtCases/CPT-SupremeCourtCases-Qwen2.5_Vocab/ \
                                 -PAC_path ./VocabFiles/CPT-SupremeCourtCases/CPT-SupremeCourtCases/vocab.json \
                                 -model_id Qwen/Qwen2.5-7B
done