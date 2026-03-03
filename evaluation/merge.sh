
#!/bin/bash


OUTPUT_BASE="./results/"


LORA_DIRS=(
    "qwen_baseline"
    "qwen_lora"
)

for dir in "${LORA_DIRS[@]}"; do
    echo "Merging results for ${dir} ..."
    eval_dir="${OUTPUT_BASE}${dir}/eval_data_with_ContextPRM_reward"

    if [ -d "$eval_dir" ]; then
        python check_pre_merge_files.py \
          --rewards_dir "$eval_dir" \
          --prm_name ContextPRM
        python merge_output.py \
            --input_dir "$eval_dir" \
            --output_file "$eval_dir/merged/merge_rewards.json"


        python check.py --rewards_dir "$eval_dir/merged/"
        python calculate_metric_by_category.py --rewards_dir "$eval_dir/merged/" --save_dir "${OUTPUT_BASE}/final_results/${dir}" --prm_name llama_lora
    else
        echo "Warning: ${eval_dir} not found, skipped."
    fi
done



