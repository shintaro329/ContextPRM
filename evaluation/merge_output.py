import os
import json
from collections import defaultdict

def load_json_files_from_dir(dir_path):
    all_data = []
    for filename in os.listdir(dir_path):
        if filename.endswith('.json'):
            with open(os.path.join(dir_path, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data.extend(data)
    return all_data

def move_parsed_answer_to_cot(data):
    for entry in data:
        top_parsed_answer = entry.pop('parsed_answer', None)
        top_parsed_answer_correctness = entry.pop('parsed_answer_correctness', None)

        for cot in entry['chain_of_thoughts']:
            if 'parsed_answer' not in cot and top_parsed_answer is not None:
                cot['parsed_answer'] = top_parsed_answer
            if 'parsed_answer_correctness' not in cot and top_parsed_answer_correctness is not None:
                if isinstance(top_parsed_answer_correctness, str):
                    cot['parsed_answer_correctness'] = (top_parsed_answer_correctness.lower() == 'true')
                else:
                    cot['parsed_answer_correctness'] = bool(top_parsed_answer_correctness)
    return data

def merge_cots_by_question(data):
    merged = {}
    for entry in data:
        qid = entry['id']
        if qid not in merged:
    
            merged[qid] = {
                'question': entry['question'],
                'answer': entry['answer'],
                'metadata': {
                    'category': entry['category'],
                    'src': entry['src']
                },
                'id': qid,
                'chain_of_thoughts': []
            }

        for cot in entry['chain_of_thoughts']:
            if not isinstance(cot, dict):
                print(f"Warning: cot is not a dict for question_id={entry['id']}, skipping this cot.")
                continue


            if 'cot_id' not in cot and 'cot_id' in entry:
                cot['cot_id'] = entry['cot_id']


            if 'prm_reward' not in cot:
                print(f"Warning: 'prm_reward' missing in cot_id={cot.get('cot_id', 'unknown')} for question_id={entry['id']}. Setting default [0].")
                cot['prm_reward'] = [0]


            if 'parsed_answer' not in cot:
                cot['parsed_answer'] = entry.get('parsed_answer', '')

            if 'parsed_answer_correctness' not in cot:
                pac = entry.get('parsed_answer_correctness', False)
                if isinstance(pac, str):
                    pac = pac.lower() == 'true'
                cot['parsed_answer_correctness'] = pac

            merged[qid]['chain_of_thoughts'].append(cot)

    return list(merged.values())

def save_merged_data(data, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True) 
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def main(input_dir, output_file):
    print(f"Loading JSON files from {input_dir}...")
    raw_data = load_json_files_from_dir(input_dir)
    print(f"Loaded {len(raw_data)} total entries.")

    print("Moving parsed_answer and parsed_answer_correctness into each cot...")
    data_fixed = move_parsed_answer_to_cot(raw_data)

    print("Merging chain_of_thoughts by question id...")
    merged_data = merge_cots_by_question(data_fixed)
    print(f"Merged into {len(merged_data)} unique questions.")

    print(f"Saving merged data to {output_file} ...")
    save_merged_data(merged_data, output_file)
    print("Done.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Merge and fix JSON CoT data for evaluation.")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing JSON files")
    parser.add_argument("--output_file", type=str, required=True, help="Output merged JSON file path")
    args = parser.parse_args()

    main(args.input_dir, args.output_file)
