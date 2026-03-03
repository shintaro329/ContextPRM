from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import torch.nn.functional as F
from prm_datasets import TokenizedPRMDataset
import evaluate
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.special import softmax
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import torch

def get_model(configs):
    

    model = AutoModelForCausalLM.from_pretrained(
    configs.model_id,
    # device_map="auto",
    torch_dtype=torch.bfloat16,  
)

    if 'lora_config' in configs:
        print('Using LoRA')
        lora_config = LoraConfig(**configs.lora_config)
        model = get_peft_model(model, lora_config)
        
    return model

def get_tokenizer(model_id):
        
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token #llama doesn't define pad token, so we need to do this
    tokenizer.padding_side='right' # we need to pad from right (so that we can do eval mask id trick for eval)


    return tokenizer

def get_datasets(configs, tokenizer):
    
    t_dataset = TokenizedPRMDataset(configs.train_data_path, 
                                    tokenizer,
                                    label_last_n = configs.train_label_last_n if 'train_label_last_n' in configs else None,
                                    max_length=configs.max_length if 'max_length' in configs else None,
                                    use_augs=configs.use_augs if 'use_augs' in configs else True,
                                    mode=configs.mode)
    e_dataset = TokenizedPRMDataset(configs.eval_data_path, 
                                    tokenizer,
                                    label_last_n = configs.eval_label_last_n if 'eval_label_last_n' in configs else None,
                                    max_length=configs.max_length if 'max_length' in configs else None,
                                    use_augs=configs.use_augs if 'use_augs' in configs else True,
                                    mode=configs.mode) if configs.eval_data_path is not None else None
    return t_dataset, e_dataset


def get_collate_func(tokenizer):
      
    return DataCollatorForTokenClassification(tokenizer=tokenizer, 
                                                        padding='longest', 
                                                        label_pad_token_id=-100,
                                                        return_tensors='pt')


def get_compute_loss_func(tokenizer):
      
    def compute_loss_func(outputs, labels, num_items_in_batch):

        plus_id= tokenizer.convert_tokens_to_ids("+")
        minus_id = tokenizer.convert_tokens_to_ids("-")

        logits = outputs.logits[:,:,[minus_id, plus_id]].reshape(-1,2)



        if num_items_in_batch is None:
            loss = F.cross_entropy(input=logits,
                            target=labels.flatten(),
                            ignore_index=-100)
            return loss


        loss = F.cross_entropy(input=logits,
                            target=labels.flatten(),
                            ignore_index=-100,
                            reduction='sum')

        return loss / num_items_in_batch
    
    return compute_loss_func


def get_compute_metrics(tokenizer):
    plus_id= tokenizer.convert_tokens_to_ids("+")
    minus_id = tokenizer.convert_tokens_to_ids("-")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred


        label_mask_PRM = (labels != -100)

        labels_PRM = labels[label_mask_PRM]
        logits_PRM = logits[:, :, [minus_id, plus_id]][label_mask_PRM]


        pred_PRM = np.argmax(logits_PRM, axis=-1)


        predf_PRM = softmax(logits_PRM, axis=-1)[:, 1]

        results = {
            "PRM Accuracy": accuracy_score(labels_PRM, pred_PRM),
            "PRM Precision": precision_score(labels_PRM, pred_PRM, zero_division=0.0),
            "PRM Recall": recall_score(labels_PRM, pred_PRM, zero_division=0.0),
            "PRM Specificity": recall_score(labels_PRM, pred_PRM, pos_label=0, zero_division=0.0),
            "PRM NPV": precision_score(labels_PRM, pred_PRM, pos_label=0, zero_division=0.0),
            "PRM F1": f1_score(labels_PRM, pred_PRM, zero_division=0.0),
            "PRM F1 Neg": f1_score(labels_PRM, pred_PRM, pos_label=0, zero_division=0.0),
        }


        try:
            results["PRM F1 AUC"] = roc_auc_score(labels_PRM, predf_PRM)
        except ValueError:
            results["PRM F1 AUC"] = float("nan")

        return results

    return compute_metrics
