


BASE_DIR="../model_train/runs/"

OUTPUT_BASE="./results/"



export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TQDM_DISABLE=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6


python get_rewards_reasoning_step.py \
    --eval_data_dir=./eval_data \
    --output_dir="${OUTPUT_BASE}_baseline/" \
    --eval_model_config=prm_models/model_config.json \
    --mode=origin\
    --prm_name=ContextPRM \

        




