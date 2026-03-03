import json
import os
import argparse
import sys

def validate_json_data(file_path):
    """
    Goes through a single JSON file and checks for the exact condition
    that causes the IndexError in the original script.
    
    The error condition is:
    1. A 'question' (entry) in the data list.
    2. Has a 'chain_of_thoughts' list.
    3. A 'cot' (solution) in that list.
    4. Has a 'prm_reward' list.
    5. That 'prm_reward' list is EMPTY ([]).
       This causes prm_rewards[-1] to fail.
    
    This function will also catch structural errors (missing keys).
    """
    
    print(f"\n--- Validating file: {file_path} ---")
    
    abnormal_question_count = 0
    total_questions = 0
    abnormal_details = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[FATAL ERROR] Could not parse JSON file. Error: {e}")
        return 1  # Return 1 error
    except FileNotFoundError:
        print(f"[FATAL ERROR] File not found: {file_path}")
        return 1
    
    if not isinstance(data, list):
        print(f"[FATAL ERROR] Root structure of JSON is not a list.")
        return 1

    total_questions = len(data)

    # Loop over each question in the JSON file
    for i, question in enumerate(data):
        is_abnormal = False
        
        try:
            # 1. Check for 'chain_of_thoughts'
            cot_solutions = question['chain_of_thoughts']
            
            if not isinstance(cot_solutions, list):
                is_abnormal = True
                abnormal_details.append(
                    f"  [FAIL] Question index {i}: 'chain_of_thoughts' is not a list (type: {type(cot_solutions)})."
                )
                continue # Go to next question

            # 2. Loop through each CoT solution
            for j, cot in enumerate(cot_solutions):
                
                try:
                    # 3. Check for 'prm_reward'
                    prm_rewards = cot['prm_reward']
                    
                    if not isinstance(prm_rewards, list):
                        is_abnormal = True
                        abnormal_details.append(
                            f"  [FAIL] Question index {i}, CoT index {j}: 'prm_reward' is not a list (type: {type(prm_rewards)})."
                        )
                        break # This CoT is bad, no need to check others in this question

                    # 4. Replicate the logic from the original script
                    # prm_rewards = prm_rewards[:-1] if len(prm_rewards) > 1 else prm_rewards
                    # This logic is what's checked:
                    
                    processed_prm_rewards = prm_rewards[:-1] if len(prm_rewards) > 1 else prm_rewards

                    # 5. Check if the *processed* list is empty.
                    # This only happens if the original prm_rewards was [].
                    # If it was [0.5], processed_prm_rewards is [0.5] (safe).
                    # If it was [0.1, 0.2], processed_prm_rewards is [0.1] (safe).
                    # If it was [], processed_prm_rewards is [] (CRASH).
                    
                    if not processed_prm_rewards:
                        is_abnormal = True
                        abnormal_details.append(
                            f"  [CRASH CONDITION] Question index {i}, CoT index {j}: 'prm_reward' list is {prm_rewards}. "
                            f"After processing, it becomes {processed_prm_rewards}, which will cause 'IndexError: list index out of range' "
                            "when 'method' is 'last'."
                        )
                        break # This CoT is bad, no need to check others in this question

                except KeyError:
                    is_abnormal = True
                    abnormal_details.append(
                        f"  [FAIL] Question index {i}, CoT index {j}: Missing 'prm_reward' key in 'chain_of_thoughts' entry."
                    )
                    break # This CoT is bad
                except TypeError:
                     is_abnormal = True
                     abnormal_details.append(
                        f"  [FAIL] Question index {i}, CoT index {j}: 'cot' entry is not a dictionary (type: {type(cot)})."
                    )
                     break # This CoT is bad

        except KeyError:
            is_abnormal = True
            abnormal_details.append(
                f"  [FAIL] Question index {i}: Missing 'chain_of_thoughts' key in question entry."
            )
        except TypeError:
            is_abnormal = True
            abnormal_details.append(
                f"  [FAIL] Question index {i}: 'question' entry is not a dictionary (type: {type(question)})."
            )
        
        if is_abnormal:
            abnormal_question_count += 1

    # --- Print Summary for the file ---
    if abnormal_question_count == 0:
        print(f"[SUCCESS] All {total_questions} entries in this file appear safe and will not cause the IndexError.")
    else:
        print(f"[FAILURE] Found {abnormal_question_count} out of {total_questions} questions that will cause a crash.")
        print("--- Error Details (first 20) ---")
        for detail in abnormal_details[:20]:
            print(detail)
        if len(abnormal_details) > 20:
            print(f"...and {len(abnormal_details) - 20} more abnormal entries.")
    
    return abnormal_question_count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Check reward JSON files for data that will cause an IndexError in the main script."
    )
    parser.add_argument('--rewards_dir', type=str, required=True, help='Directory of the prm rewards files')
    args = parser.parse_args()

    if not os.path.isdir(args.rewards_dir):
        print(f"Error: Directory not found: {args.rewards_dir}")
        sys.exit(1)

    file_list = [
        f for f in os.listdir(args.rewards_dir) 
        if os.path.isfile(os.path.join(args.rewards_dir, f)) and f.endswith('rewards.json')
    ]

    if not file_list:
        print(f"No '...rewards.json' files found in directory: {args.rewards_dir}")
        sys.exit(0)

    print(f"Found {len(file_list)} '...rewards.json' files to check.")
    
    total_errors = 0
    for filename in file_list:
        file_path = os.path.join(args.rewards_dir, filename)
        total_errors += validate_json_data(file_path)

    print("\n--- Overall Summary ---")
    if total_errors == 0:
        print("All files checked, and all entries look safe.")
    else:
        print(f"Found a total of {total_errors} abnormal questions across all files.")
        print("Please fix the data in the files listed above before re-running the main script.")
