import math
import statistics
import json
import torch
import torch.nn.functional as F
from torch.types import Device
from typing import Optional
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from .prm_interface import PRM, StepScore
from peft import PeftModel 
def read_json_file(file_path):

    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def make_step_rewards(logits, token_masks):
    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1) # bs, seq_len, num_labels
    
    all_scores_res = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i] # seq_len, num_labels
        positive_probs = sample[sample != 0].view(-1, 2)[:, 1] # valid_tokens, num_labels
        non_zero_elements_list = positive_probs.cpu().tolist()
        all_scores_res.append(non_zero_elements_list)
    return all_scores_res


class QwenMathPRM(PRM):
    def __init__(
        self,
        aggregation: str = 'full',
        quantization_config: Optional[BitsAndBytesConfig] = None,
        device: Optional[Device] = None,
        model_id: str = 'Qwen/Qwen2.5-Math-7B-PRM',
        lora_adapter_path: str = None
    ) -> None:
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.quantization_config = quantization_config
        self.model_id = model_id
        self.lora_adapter_path = lora_adapter_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)

        if self.quantization_config:
            raise NotImplementedError
        else:
            self.model = AutoModel.from_pretrained(
                self.model_id,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
            if self.lora_adapter_path:
                print(f"Loading LoRA adapter from: {self.lora_adapter_path}")
                self.model = PeftModel.from_pretrained(
                    self.model,
                    self.lora_adapter_path,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True
                )
                print("Merging LoRA weights for faster inference...")
                self.model = self.model.merge_and_unload()
        self.model.eval()
        self.aggregation = aggregation

    def __call_batch(self, steps: list[str]) -> list[StepScore]:
        """
        Batch forward, step reward calculation per sample ensures output identical
        to original __call__ + __call_single logic.
        """
        # tokenize batch (padding to max length)
        encodings = self.tokenizer(steps, return_tensors='pt', padding=True)
        input_ids = encodings['input_ids'].to(self.model.device)
        attention_mask = encodings['attention_mask'].to(self.model.device)

        # batch forward
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # step token id
        step_sep_id = self.tokenizer.encode('<extra_0>')[0]
        token_masks = (input_ids == step_sep_id)  # [batch, seq_len]

        # process each sample individually (确保输出与原逻辑一致)
        result = []
        for i, beam in enumerate(steps):
            step_reward = make_step_rewards(outputs[0][i:i+1], token_masks[i:i+1])
            step_probs = step_reward[0]

            # aggregation
            if self.aggregation == 'min':
                score = min(step_probs)
            elif self.aggregation == 'max':
                score = max(step_probs)
            elif self.aggregation == 'mean':
                score = statistics.mean(step_probs)
            elif self.aggregation == 'prod':
                score = math.prod(step_probs)
            elif self.aggregation == 'last':
                score = step_probs[-1]
            elif self.aggregation == 'full':
                score = step_probs
            else:
                raise NotImplementedError

            result.append(StepScore(step=beam, score=score))

        return result

    # 保留原 __call__ 接口，方便兼容
    def __call__(self, steps: list[str]) -> list[StepScore]:
        return self.__call_batch(steps)

# class QwenMathPRM(PRM):
#     def __init__(
#         self,
#         aggregation: str = 'full',
#         quantization_config: Optional[BitsAndBytesConfig] = None,
#         device: Optional[Device] = None,
#         model_id: str = 'Qwen/Qwen2.5-Math-7B-PRM',
#         lora_adapter_path:str=None

#     ) -> None:
#         self.device = (
#             device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
#         )
#         self.quantization_config = quantization_config
#         self.model_id = model_id
#         self.lora_adapter_path=lora_adapter_path
#         self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)

#         if self.quantization_config:
#             raise NotImplementedError
#         else:
#             self.model = AutoModel.from_pretrained(
#                 self.model_id, 
#                 device_map="auto", 
#                 torch_dtype=torch.bfloat16,
#                 trust_remote_code=True,
#                 )
#             if self.lora_adapter_path:
#                 print(f"Loading LoRA adapter from: {self.lora_adapter_path}")
#                 self.model = PeftModel.from_pretrained(
#                     self.model,
#                     self.lora_adapter_path,
#                     device_map="auto",
#                     torch_dtype=torch.bfloat16,
#                     trust_remote_code=True,
#                 )
#             self.model.eval()
#         self.aggregation = aggregation

#     def __call_single(self, single_beam: str) -> list[float]:
#         '''
#         Computes scores for each reasoning step in the single_beam.

#         Args:
#             single_beam (str): A single reasoning beam, consisting of Question + Solution.

#         Returns:
#             list[float]: The scores for each step in the Solution.
#         '''
#         ###

#         input_ids = self.tokenizer.encode(
#             single_beam, 
#             return_tensors='pt', 
#         ).to(self.model.device)

#         outputs = self.model(input_ids=input_ids)
#         step_sep_id = self.tokenizer.encode('<extra_0>')[0]
#         token_masks = (input_ids == step_sep_id)
#         # print(outputs)
#         # print(token_masks)

#         step_reward = make_step_rewards(outputs[0], token_masks)
#         # print(step_reward)
#         # input()
#         step_probs = step_reward[0]

#         ###
#         if self.aggregation == 'min':
#             return min(step_probs)
#         elif self.aggregation == 'max':
#             return max(step_probs)
#         elif self.aggregation == 'mean':
#             return statistics.mean(step_probs)
#         elif self.aggregation == 'prod':
#             return math.prod(step_probs)
#         elif self.aggregation == 'last':
#             return step_probs[-1]
#         elif self.aggregation == 'full':
#             return step_probs
#         else:
#             raise NotImplementedError


#     def __call__(self, steps: list[str]) -> list[StepScore]:
#         '''
#         Computes scores for a list of reasoning beams.

#         Args:
#             steps (list[str]): A list of reasoning beams.

#         Returns:
#             list[StepScore]: A list of StepScore objects, each containing step and score.
#         '''
#         result = []

#         for beam in steps:
#             step_score = self.__call_single(beam)
#             result.append(StepScore(step=beam, score=step_score))

#         return result