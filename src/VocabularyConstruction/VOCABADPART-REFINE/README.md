## Execution Steps

1. Organize data as a txt file
2. Run GenerateVocabs-CPT.sh. This will generate all the necessary vocabulary files. Right now it generates all vocabulary sizes from 5000 to 100K. In out paper, we use a default setting of 10k.
3. if you want to explore, expansion without replacement, just run AddMergeRules-{Llama/Qwen}.ipynb.
4. If you want to explore, expansion with replacement, after runinn AddMergeRules-{Llama/Qwen}.ipynb run ReplaceTokens-xx.ipynb. This will generate the final vocabulary files for adaptation.
