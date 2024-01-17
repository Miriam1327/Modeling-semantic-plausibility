import os
import torch
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    LlamaForSequenceClassification,
)
from accelerate import Accelerator

from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel,
    prepare_model_for_kbit_training
)

from torch.optim import AdamW
from tqdm.auto import tqdm
import torch.nn as nn
from transformers import get_scheduler
from evaluation import get_accuracy, get_precision_recall


class llama2_7b_Model_peft(object):
    def __init__(self,base_model):
        self.base_model=base_model
        self.tokenizer=self.set_all_model_need(base_model)[1]
        self.model = self.create_peft_config()
    
    def set_all_model_need(self,base_model):
        tokenizer= AutoTokenizer.from_pretrained(base_model)
        
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_quant_type="nf8",
            bnb_8bit_compute_dtype=torch.bfloat16, 
            bnb_8bit_use_double_quant=True,
        )

        model = LlamaForSequenceClassification.from_pretrained(
            base_model,
            quantization_config=quant_config,
            device_map={"": 0},
            num_labels=2
            )
        model = prepare_model_for_kbit_training(model)
        model.config.use_cache = False
        model.config.pretraining_tp = 1
        model.config.pad_token_id = model.config.eos_token_id
        print("Original number of parameters:", sum(p.numel() for p in model.parameters()))
        print(model)
        return model,tokenizer
    

    def create_peft_config(self):
        # create peft config
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules = ["q_proj", "v_proj"]
        )

        model=self.set_all_model_need(self.base_model)[0]
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        return model
    
    def do_inference(self, dataloader):
        self.model.eval()
        with torch.no_grad():
            for labels,text in dataloader:
                logits = self.model(**text).logits
                the_labels=labels
                get_accuracy(logits, the_labels,flag2='print')
                get_precision_recall(logits, the_labels,flag2='print')
                get_precision_recall(logits, the_labels,flag='false',flag2='print')





    