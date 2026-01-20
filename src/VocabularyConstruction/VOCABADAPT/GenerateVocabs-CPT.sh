#!/bin/bash

mkdir -p VocabFiles-PostNER
mkdir -p VocabFiles-PostNER/CPT-SupremeCourtCase/
mkdir -p VocabFiles-PostNER/CPT-SupremeCourtCase/CPT-SupremeCourtCase/
mkdir -p VocabFiles-PostNER/CPT-SupremeCourtCase/CPT-SupremeCourtCase-Qwen2.5_Vocab/
mkdir -p VocabFiles-PostNER/CPT-SupremeCourtCase/CPT-SupremeCourtCase-Llama3.1_Vocab/

python GenerateVocabulary.py -input_path "../../data/Legal_Train_OOV_Tokens_Cleaned.csv" -dataset CPT-SupremeCourtCase -dump_dir VocabFiles-PostNER/CPT-SupremeCourtCase/CPT-SupremeCourtCase/

for v_size in 5000 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000
do
    python GenerateSubwords-CPT.py -v_size $v_size \
                                 -vpath ./VocabFiles-PostNER/CPT-SupremeCourtCase/CPT-SupremeCourtCase-Llama3.1_Vocab/ \
                                 -PAC_path ./VocabFiles-PostNER/CPT-SupremeCourtCase/CPT-SupremeCourtCase/vocab.json \
                                 -model_id meta-llama/Llama-3.1-8B
done


for v_size in 5000 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000
do
    python GenerateSubwords-CPT.py -v_size $v_size \
                                 -vpath ./VocabFiles-PostNER/CPT-SupremeCourtCase/CPT-SupremeCourtCase-Qwen2.5_Vocab/ \
                                 -PAC_path ./VocabFiles-PostNER/CPT-SupremeCourtCase/CPT-SupremeCourtCase/vocab.json \
                                 -model_id Qwen/Qwen2.5-7B
done