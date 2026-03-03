import json
import os
import argparse
import sys

def validate_reward_file(file_path):
    """
    Checks a single (pre-merge) reward file.
    It iterates through all questions and all their CoTs,
    and checks if 'prm_reward' is missing or an empty list [].
    """
    
    print(f"\n--- Validating file: {file_path} ---")
    
    total_questions = 0
    total_cots = 0
    empty_reward_cots = 0
    missing_reward_cots = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[FATAL ERROR] Could not read or parse file: {e}")
        return 1, 0, 0, 0 # file_error, total_q, total_c, empty_c

    if not isinstance(data, list):
        print(f"[FATAL ERROR] Root structure is not a list.")
        return 1, 0, 0, 0

    total_questions = len(data)
    
    # Loop over each question entry
    for i, question in enumerate(data):
        try:
            # Get all CoT solutions for this question
            cot_solutions = question['chain_of_thoughts']
            
            if not cot_solutions:
                continue # Skip questions with no CoTs

            total_cots += len(cot_solutions)
            
            # Loop through each CoT solution
            for j, cot in enumerate(cot_solutions):
                
                # Check 1: Is the key missing?
                if 'prm_reward' not in cot:
                    missing_reward_cots += 1
                
                # Check 2: Is the key present but the value is empty? (e.g., [])
                # 'not cot['prm_reward']' will be True for []
                elif not cot['prm_reward']: 
                    empty_reward_cots += 1
                        
        except KeyError:
            print(f"  [WARN] Question index {i} is missing 'chain_of_thoughts'.")
            continue
        except Exception as e:
            print(f"  [WARN] Error processing question index {i}: {e}")
            continue
    
    # --- Print Summary for this file ---
    if missing_reward_cots > 0 or empty_reward_cots > 0:
        print(f"[FAILURE] Found {total_questions} questions with {total_cots} CoTs.")
        if missing_reward_cots > 0:
            print(f"  - CoTs with MISSING 'prm_reward' key: {missing_reward_cots}")
        if empty_reward_cots > 0:
            print(f"  - CoTs with EMPTY 'prm_reward' list ([]): {empty_reward_cots}")
        return 1, total_questions, total_cots, empty_reward_cots # 1 file error
    else:
        print(f"[SUCCESS] All {total_cots} CoTs in {total_questions} questions have valid, non-empty 'prm_reward' fields.")
        return 0, total_questions, total_cots, 0 # 0 file errors

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Check pre-merge reward JSON files for empty rewards.")
    parser.add_argument('--rewards_dir', type=str, required=True, help="Directory of the prm rewards files (e.g., .../eval_data_with_ContextPRM_reward)")
    parser.add_argument('--prm_name', type=str, required=True, help="PRM name (e.g., ContextPRM), used to find files.")
    args = parser.parse_args()

    if not os.path.isdir(args.rewards_dir):
        print(f"Error: Directory not found: {args.rewards_dir}")
        sys.exit(1)

    # Find files based on the naming convention from your main() script
    file_suffix = f'_with_{args.prm_name}_rewards.json'
    file_list = [
        f for f in os.listdir(args.rewards_dir) 
        if os.path.isfile(os.path.join(args.rewards_dir, f)) and f.endswith(file_suffix)
    ]

    if not file_list:
        print(f"No '...{file_suffix}' files found in directory: {args.rewards_dir}")
        print("Please check your --rewards_dir and --prm_name arguments.")
        sys.exit(0)

    print(f"Found {len(file_list)} files matching '{file_suffix}' to check...")
    
    total_files_with_errors = 0
    grand_total_questions = 0
    grand_total_cots = 0
    grand_total_empty_cots = 0
    
    for filename in file_list:
        file_path = os.path.join(args.rewards_dir, filename)
        file_errors, num_questions, num_cots, num_empty = validate_reward_file(file_path)
        
        total_files_with_errors += file_errors
        grand_total_questions += num_questions
        grand_total_cots += num_cots
        grand_total_empty_cots += num_empty

    print("\n--- Overall Summary ---")
    print(f"Checked {len(file_list)} files.")
    print(f"Total questions processed: {grand_total_questions}")
    print(f"Total CoTs processed: {grand_total_cots}")
    if total_files_with_errors == 0:
        print("All files look good! All CoTs have a non-empty 'prm_reward'.")
    else:
        print(f"Found {total_files_with_errors} files containing errors.")
        print(f"Total CoTs with EMPTY 'prm_reward' (value '[]'): {grand_total_empty_cots}")
        print("\nThis confirms the 'main()' script (calculate_rewards.py) is saving empty rewards from the model.")
