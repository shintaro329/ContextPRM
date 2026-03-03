import json
from tqdm import tqdm
from torch.utils.data import Dataset
from copy import deepcopy


def merge_dicts(dict_list):
    merged_dict = deepcopy(dict_list[0])
    for d in dict_list[1:]:
        for key, value in d.items():
            merged_dict[key].extend(value)
    return merged_dict

def tokenize_step(cot_step, label, tokenizer, label_mask_token_id=-100, label_last_n=None):
    cot_step_tokenized = tokenizer(cot_step, add_special_tokens=False)
    if label_last_n is None or label_last_n > len(cot_step_tokenized.input_ids):
        cot_step_labels = [label] * len(cot_step_tokenized.input_ids)
    else:
        cot_step_labels = [label_mask_token_id] * (len(cot_step_tokenized.input_ids) - label_last_n) + [label] * label_last_n
    
    cot_step_tokenized['labels'] = cot_step_labels
    return cot_step_tokenized


def tokenize_one_cot_origin(question_tokenized, data, tokenizer, **kwargs):
    label_mask_token_id = kwargs.get('label_mask_token_id', -100)
    label_last_n = kwargs.get('label_last_n')
    max_length = kwargs.get('max_length')
    use_augs = kwargs.get('use_augs', True)

    if 'labels' not in data or data['labels'] is None:
        return []
    
    labels = data['labels']
    cot_steps_tokenized = []

    for i,step in enumerate(data['steps']):
        cot_step = f'{step} \n\n\n\n'
        label = 1 if labels[i] == 1 else 0
        cot_step_tokenized = tokenize_step(cot_step, label=label, tokenizer=tokenizer, label_mask_token_id=label_mask_token_id, label_last_n=label_last_n)
        cot_steps_tokenized.append(cot_step_tokenized)
        if label == 0:
            break
            
    augs = []
    if use_augs:
        for aug in data['augs']:
            aug_idx = aug['aug_idx']
            aug_step_content = aug['aug_step']
            aug_step = f'{aug_step_content} \n\n\n\n'
            aug_label = 1 if aug['aug_type'] == 1 or aug['aug_type'] == 0 else 0
            aug_step_tokenized = tokenize_step(aug_step, label=aug_label, tokenizer=tokenizer, label_mask_token_id=label_mask_token_id, label_last_n=label_last_n)
            augs.append((aug_step_tokenized, aug_idx))

    tokenized = []
    chosen_tokenized = merge_dicts([question_tokenized] + cot_steps_tokenized)
    if max_length is None or len(chosen_tokenized['input_ids']) <= max_length:
        tokenized.append(chosen_tokenized)

    for cot_step_tokenized in cot_steps_tokenized:
        cot_step_tokenized['labels'] = [label_mask_token_id] * len(cot_step_tokenized['labels'])

    for aug_step_tokenized, aug_idx in augs:
        aug_tokenized = merge_dicts([question_tokenized] + cot_steps_tokenized[:aug_idx] + [aug_step_tokenized])
        if max_length is None or len(aug_tokenized['input_ids']) <= max_length:
            tokenized.append(aug_tokenized)

    return tokenized
    

def tokenize_one_cot_context_independent(question_tokenized, question_text, data, tokenizer, **kwargs):
    """
    构造与 origin 模式结构一致的 context 模式数据，用于高效的统一推理。
    """
    label_mask_token_id = kwargs.get('label_mask_token_id', -100)
    label_last_n = kwargs.get('label_last_n')
    max_length = kwargs.get('max_length')
    use_augs = kwargs.get('use_augs', True)

    # if 'context_consistent_labels' not in data or data['context_consistent_labels'] is None:
    #     return []
    if 'labels' not in data or data['labels'] is None:
        return []
    # labels = data['context_consistent_labels']
    labels = data['labels']
    steps = data['steps']
    cot_steps_tokenized = []
    
    for i, current_step_text in enumerate(steps):
        context_text = question_text if i == 0 else steps[i-1]
        cot_step = f"[CONTEXT]: {context_text}\n\n[CURRENT STEP]: {current_step_text} \n\n\n\n"
        label = 1 if labels[i] == 1 else 0
        
        cot_step_tokenized = tokenize_step(cot_step, label=label, tokenizer=tokenizer, label_mask_token_id=label_mask_token_id, label_last_n=label_last_n)
        cot_steps_tokenized.append(cot_step_tokenized)
        if label == 0:
            break
            
    augs = []
    if use_augs and 'augs' in data:
        for aug in data['augs']:
            aug_idx = aug['aug_idx']
            aug_step_content = aug['aug_step']
            aug_context_text = question_text if aug_idx == 0 else steps[aug_idx - 1]
            aug_step = f"[CONTEXT]: {aug_context_text}\n\n[CURRENT STEP]: {aug_step_content} \n\n\n\n"
            aug_label = 0
            aug_step_tokenized = tokenize_step(aug_step, label=aug_label, tokenizer=tokenizer, label_mask_token_id=label_mask_token_id, label_last_n=label_last_n)
            augs.append((aug_step_tokenized, aug_idx))

    tokenized = []
    chosen_tokenized = merge_dicts([question_tokenized] + cot_steps_tokenized)
    if max_length is None or len(chosen_tokenized['input_ids']) <= max_length:
        tokenized.append(chosen_tokenized)

    for cot_step_tokenized in cot_steps_tokenized:
        cot_step_tokenized['labels'] = [label_mask_token_id] * len(cot_step_tokenized['labels'])

    for aug_step_tokenized, aug_idx in augs:
        aug_tokenized = merge_dicts([question_tokenized] + cot_steps_tokenized[:aug_idx] + [aug_step_tokenized])
        if max_length is None or len(aug_tokenized['input_ids']) <= max_length:
            tokenized.append(aug_tokenized)

    return tokenized


def tokenize_one_question(data, tokenizer, mode='origin', **kwargs):
    question = data['question']
    question_tokenized = tokenizer(f'{question} \n\n')
    question_tokenized['labels'] = [kwargs.get('label_mask_token_id', -100)] * len(question_tokenized['input_ids'])
    
    all_tokenized_samples = []


    cot_kwargs = kwargs.copy()
    cot_kwargs['question_tokenized'] = question_tokenized
    cot_kwargs['tokenizer'] = tokenizer

    if mode == 'origin':
        cot_tokenizer_func = tokenize_one_cot_origin
    elif mode == 'context_independent':
        cot_tokenizer_func = tokenize_one_cot_context_independent
        cot_kwargs['question_text'] = question 
    else:
        raise ValueError(f"Mode must be 'origin' or 'context_independent', but got {mode}")

    for cot in data['chain_of_thoughts']:
        cot_kwargs['data'] = cot
        samples = cot_tokenizer_func(**cot_kwargs)
        all_tokenized_samples.extend(samples)
    
    return all_tokenized_samples

def read_json(d):
    if d.endswith('jsonl'):
        with open(d, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]
    elif d.endswith('json'):
        with open(d, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        raise NotImplementedError('currently only supports json and jsonl files')


def tokenize_data(data_path, tokenizer, mode='origin', **kwargs):
    if isinstance(data_path, list):
        text_data = []
        for d in data_path:
            text_data.extend(read_json(d))
    else:
        text_data = read_json(data_path)
    
    tokenized_data = []
    for d in tqdm(text_data, desc=f"Tokenizing in '{mode}' mode"):
        tokenized_data.extend(tokenize_one_question(d, tokenizer, mode=mode, **kwargs))
        
    return tokenized_data


class TokenizedPRMDataset(Dataset):
    def __init__(self, 
                 data_path, 
                 tokenizer, 
                 mode='origin',
                 label_mask_token_id=-100,
                 label_last_n=None,
                 max_length=None,
                 use_augs=True):

        super(TokenizedPRMDataset, self).__init__()
        
        print(f"Initializing dataset in '{mode}' mode...")
        
        tokenize_kwargs = {
            'label_mask_token_id': label_mask_token_id,
            'label_last_n': label_last_n,
            'max_length': max_length,
            'use_augs': use_augs
        }
        
        self.tokenized_data = tokenize_data(
            data_path=data_path, 
            tokenizer=tokenizer, 
            mode=mode, 
            **tokenize_kwargs
        )

        print(f"Finished tokenizing. Total samples created: {len(self.tokenized_data)}")

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, i):
        return self.tokenized_data[i]