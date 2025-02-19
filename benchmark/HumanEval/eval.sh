# 
LANG="python"
OUPUT_DIR="output"
MODEL="deepseek-coder-1.3b-instruct"

CUDA_VISIBLE_DEVICES=0,1 python eval_instruct.py \
    --model "deepseek-ai/$MODEL" \
    --output_path "$OUPUT_DIR/${LANG}.$MODEL.jsonl" \
    --language $LANG \
    --temp_dir $OUPUT_DIR