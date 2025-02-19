# This script exists just to load models faster
import functools
import os
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
import torch
from transformers import (AutoModelForCausalLM,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          OPTForCausalLM, Qwen2Tokenizer)

from _settings import MODEL_PATH


@functools.lru_cache()
def _load_pretrained_model(model_name, device, load_in_8bit, torch_dtype=torch.float16):
    kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters=True)]
    accelerator = Accelerator(mixed_precision="bf16", kwargs_handlers=kwargs_handlers)  
    if load_in_8bit:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', load_in_8bit=True, torch_dtype=torch.bfloat16)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', torch_dtype=torch.bfloat16)
    # model.to(device)
    return model


@functools.lru_cache()
def _load_pretrained_tokenizer(model_name, use_fast=False):
    if model_name.startswith('facebook/opt-'):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH+model_name.split("/")[1], use_fast=use_fast)
    elif model_name == "microsoft/deberta-large-mnli":
        tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
    elif model_name == "roberta-large-mnli":
        tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
    elif model_name == 'llama-7b-hf' or model_name == 'llama-13b-hf' or model_name == "llama2-7b-hf":
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODEL_PATH, model_name), cache_dir=None, use_fast=use_fast)
        tokenizer.eos_token_id = 2
        tokenizer.bos_token_id = 1
        tokenizer.eos_token = tokenizer.decode(tokenizer.eos_token_id)
        tokenizer.bos_token = tokenizer.decode(tokenizer.bos_token_id)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    elif model_name == "falcon-7b":
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODEL_PATH, model_name), trust_remote_code=True, cache_dir=None, use_fast=use_fast)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True) 
    return tokenizer
