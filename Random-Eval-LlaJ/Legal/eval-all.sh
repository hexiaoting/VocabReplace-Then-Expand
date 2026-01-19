python detect_confab.py \
    --model_path /path/to/MedGemma/ \
    --input_file /path/to/input/file.csv \
    --eval_column {BASE/CPTOnly/VA} \
    --prompt_file_prefix 'coherence' \
    --batch_size 1 \
    --device_id 0 

python detect_confab.py \
    --model_path /path/to/MedGemma/ \
    --input_file /path/to/input/file.csv \
    --eval_column {BASE/CPTOnly/VA} \
    --prompt_file_prefix 'relevance' \
    --batch_size 1 \
    --device_id 0 &

python detect_confab.py \
    --model_path /path/to/MedGemma/ \
    --input_file /path/to/input/file.csv \
    --eval_column {BASE/CPTOnly/VA} \
    --prompt_file_prefix '' \
    --batch_size 1 \
    --device_id 0 &
