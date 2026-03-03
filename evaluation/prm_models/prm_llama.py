import math
import statistics
import json
import torch
from torch.types import Device
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from .prm_interface import PRM, StepScore
from peft import PeftModel 

from torch.nn.functional import softmax


from accelerate import Accelerator

def read_json_file(file_path):

    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def check_model_precision(model):
    for name, param in model.named_parameters():
        print(f'Parameter: {name}, dtype: {param.dtype}')


def get_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token  
    tokenizer.padding_side = 'left' 
    tokenizer.truncation_side = 'left'
    return tokenizer
    
def _aggregate_scores(aggregation_strategy, step_probs):
    if not step_probs:
        return [] if aggregation_strategy == 'full' else 0.0

    if aggregation_strategy == 'min':
        return min(step_probs)
    elif aggregation_strategy == 'max':
        return max(step_probs)
    elif aggregation_strategy == 'mean':
        return statistics.mean(step_probs)
    elif aggregation_strategy == 'prod':
        return math.prod(step_probs)
    elif aggregation_strategy == 'last':
        return step_probs[-1]
    elif aggregation_strategy == 'full':
        return step_probs
    else:
        raise NotImplementedError

class LlamaPRM(PRM):
    
    def __init__(
            self,
            model_id: str, # Path to the base model
            lora_adapter_path: Optional[str] = None, 
            aggregation: str = 'full',
            quantization_config: Optional[BitsAndBytesConfig] = None,
            device: Optional[str] = None, 
        ) -> None:
            self.device = (
                device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
            )
            self.quantization_config = quantization_config
            self.model_id = model_id
            self.lora_adapter_path = lora_adapter_path
    
            self.tokenizer = get_tokenizer(self.model_id)
    
            
            self.candidate_tokens = [12, 10]

            # --- Model Loading Logic ---
            # 3. Load the base model first
            print(f"Loading base model: {self.model_id}")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=self.quantization_config,
                torch_dtype=torch.bfloat16,
            )
    
            # 4. If a LoRA adapter path is provided, load it on top of the base model
            if self.lora_adapter_path:
                print(f"Loading LoRA adapter from: {self.lora_adapter_path}")
                self.tokenizer=get_tokenizer(self.lora_adapter_path)
                self.model = PeftModel.from_pretrained(base_model, self.lora_adapter_path)
                print("Merging LoRA weights for faster inference...")
                self.model = self.model.merge_and_unload()
            else:
                self.model = base_model # Use the base model if no adapter is given
    
            if not self.quantization_config:
                self.model.to(self.device)
            
            self.model.eval() # Set the model to evaluation mode
            self.aggregation = aggregation


    def __call_single(self, single_beam: str) -> list[float]:
        '''
        Computes scores for each reasoning step in the single_beam.

        Args:
            single_beam (str): A single reasoning beam, consisting of Question + Solution.

        Returns:
            list[float]: The scores for each step in the Solution.
        '''
        # Tokenize the entire input
        encoded = self.tokenizer.encode(single_beam, return_tensors='pt').to(self.device)
        input_ids = encoded[0]
        total_length = input_ids.size(0)

        input_id = torch.tensor([self.tokenizer.encode(single_beam)]).to(self.device)

        with torch.no_grad():
            logits = self.model(input_id).logits[:,:,self.candidate_tokens]
            # print(logits)
            scores = logits.softmax(dim=-1)[:,:,1] 
            # print(scores)
            selected_logits = logits[input_id == 23535]
            # print(selected_logits)
            step_scores = scores[input_id == 23535]
            # print(step_scores)# step tag is 22701
            step_probs  = step_scores.tolist()

        if self.aggregation == 'min':
            return min(step_probs)
        elif self.aggregation == 'max':
            return max(step_probs)
        elif self.aggregation == 'mean':
            return statistics.mean(step_probs)
        elif self.aggregation == 'prod':
            return math.prod(step_probs)
        elif self.aggregation == 'last':
            return step_probs[-1]
        elif self.aggregation == 'full':
            return step_probs
        else:
            raise NotImplementedError


    def __call__(self, steps: list[str]) -> list[StepScore]:
        '''
        Computes scores for a list of reasoning beams.

        Args:
            steps (list[str]): A list of reasoning beams.

        Returns:
            list[StepScore]: A list of StepScore objects, each containing step and score.
        '''
        result = []

        for beam in steps:
            # print('++++++++++++++++++++++++++++++++'+beam+'++++++++++++++++++++++++++++++++')
            step_score = self.__call_single(beam)
            result.append(StepScore(step=beam, score=step_score))

        return result

class LlamaContextPRM(PRM):
    
    def __init__(
            self,
            model_id: str,
            lora_adapter_path: Optional[str] = None,
            aggregation: str = 'full',
            quantization_config: Optional[BitsAndBytesConfig] = None,
            device: Optional[str] = None,
        ) -> None:
        

        self.device = (
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        self.quantization_config = quantization_config
        self.model_id = model_id
        self.lora_adapter_path = lora_adapter_path
        
        self.tokenizer = get_tokenizer(self.model_id)
        

        self.candidate_tokens = [12, 10] 
        
        print(f"Loading base model: {self.model_id}")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=self.quantization_config,
            torch_dtype=torch.bfloat16,
        )

        if self.lora_adapter_path:
            print(f"Loading LoRA adapter from: {self.lora_adapter_path}")
            self.model = PeftModel.from_pretrained(base_model, self.lora_adapter_path)
            print("Merging LoRA weights for faster inference...")
            self.model = self.model.merge_and_unload()
        else:
            self.model = base_model

        # if not self.quantization_config:
        #     self.model.to(self.device)
            
        self.model.eval()
        self.aggregation = aggregation
        print(f"LlamaContextPRM initialized.")
        self.model=self.model.to(self.device)


    def __call_single(self, single_beam: str) -> list[float]:

        parts = single_beam.split(' \n\n', 1)
        question = parts[0].strip()
        cot_string = parts[1] if len(parts) > 1 else ""
        steps = [s.strip() for s in cot_string.split('\n\n\n\n') if s.strip()]

        if not steps:
            return []


        reconstructed_parts = []
        for i, current_step in enumerate(steps):
            context = question if i == 0 else steps[i-1]

            formatted_step = f"[CONTEXT]: {context}\n\n[CURRENT STEP]: {current_step} \n\n\n\n"
            reconstructed_parts.append(formatted_step)
        
        reconstructed_cot = "".join(reconstructed_parts)
        final_input_text = f"{question} \n\n{reconstructed_cot}"


        input_id = torch.tensor([self.tokenizer.encode(final_input_text)]).to(self.device)

        with torch.no_grad():
            logits = self.model(input_id).logits[:,:,self.candidate_tokens]
            scores = logits.softmax(dim=-1)[:,:,1]
            

            step_scores = scores[input_id == 23535] 
            step_probs  = step_scores.tolist()


        if self.aggregation == 'min':
            return min(step_probs) if step_probs else 0.0
        elif self.aggregation == 'max':
            return max(step_probs) if step_probs else 0.0
        elif self.aggregation == 'mean':
            return statistics.mean(step_probs) if step_probs else 0.0
        elif self.aggregation == 'prod':
            return math.prod(step_probs)
        elif self.aggregation == 'last':
            return step_probs[-1] if step_probs else 0.0
        elif self.aggregation == 'full':
            return step_probs
        else:
            raise NotImplementedError

    def __call__(self, steps: list[str]) -> list[StepScore]:
       
        result = []
        for beam in steps:
            step_score = self.__call_single(beam)
            result.append(StepScore(step=beam, score=step_score))
        return result



