#!/bin/bash


BASE_DIR="../runs"

OUTPUT_BASE="./results/"


LORA_DIRS=(
)


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TQDM_DISABLE=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6

for dir in "${LORA_DIRS[@]}"; do
    ckpt_dir=$(find "${BASE_DIR}/${dir}" -mindepth 1 -maxdepth 1 -type d | head -n 1)
    echo "Running for $dir with checkpoint ${ckpt_dir} ..."
    
    python get_rewards_reasoning_step.py \
        --eval_data_dir=./eval_data \
        --output_dir="${OUTPUT_BASE}${dir}" \
        --eval_model_config=prm_models/model_config.json \
        --mode=context\
        --prm_name=ContextPRM \
		--lora_path="${ckpt_dir}" \
        
done





