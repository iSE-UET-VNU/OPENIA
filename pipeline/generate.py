import argparse
import glob
import json
import os
import copy
import time
import gc
import pickle
import pandas as pd
import torch
import tqdm
import pickle
from transformers import StoppingCriteria, StoppingCriteriaList
from sentence_transformers import SentenceTransformer
from torchmetrics.text.bert import BERTScore

import _settings
import dataeval.w_humaneval as human_eval
import dataeval.w_mbpp as mbpp
from dataeval.w_humaneval import cleanup_code as human_eval_cleanup_code
import models
import utils
from func.metric import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='llama-13b-hf')
parser.add_argument('--dataset', type=str, default='human_eval')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--tensor_parallel_size', type=int, default=1)
parser.add_argument('--fraction_of_data_to_use', type=float, default=1.0)
parser.add_argument('--num_generations_per_prompt', type=int, default=10)
parser.add_argument('--max_new_tokens', type=int, default=500)
parser.add_argument('--temperature', type=float, default=0.5)
parser.add_argument('--decoding_method', type=str, default='greedy')
parser.add_argument('--top_p', type=float, default=0.99)
parser.add_argument('--top_k', type=int, default=10)
parser.add_argument('--seed', type=int, default=2023)
parser.add_argument('--nprocess', type=int, default=None)
parser.add_argument('--project_ind', type=int, default=0)
parser.add_argument("--layers", default=-1, nargs='*', type=int,
                        help="List of layers of the LM to save embeddings from indexed negatively from the end")
parser.add_argument("--language", default="python", type=str,)
parser.add_argument("--load_in_8bit", action="store_true", help="Whether to load the model in 8bit mode")
#-1: Last Layer, -2: Middle Layer, Others: Specific Layer
args = parser.parse_args()
print(args.model.replace('/', '_'))
ml_time = int(time.time() * 1000)
layer_name = '_'.join(str(x) for x in args.layers)
OUTPUT_DIR = os.path.join(_settings.GENERATION_FOLDER, f'{args.model.replace("/", "_")}_{args.dataset}_{args.language}_{layer_name}')
os.makedirs(OUTPUT_DIR, exist_ok=True)
logInfo = open(os.path.join(OUTPUT_DIR, "logInfo.txt"), mode="w",encoding="utf-8")

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords_str, tokenizer):
        StoppingCriteria.__init__(self)
        self.current_context = []
        self.tokenizer = tokenizer
        self.keywords_str = keywords_str
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        self.current_context.append(input_ids[0][-1].item())
        current_context = self.tokenizer.decode(self.current_context)
        for word in self.keywords_str:
            if word in current_context:
                return True
        return False

# _UNUSED_TOKENIZER = models.load_tokenizer()
def get_dataset_fn(data_name):
    if data_name == 'human_eval':
        return human_eval.get_dataset
    if data_name == 'mbpp':
        return mbpp.get_dataset

def get_clean_up_code_fn(data_name):
    if data_name == 'human_eval':
        return human_eval_cleanup_code

def get_num_tokens(generation, tokenizer, language_type='python', stop_words=[], ground_truth=None):
    if args.dataset == 'human_eval':
        return humaneval_get_num_tokens(generation, tokenizer, language_type, stop_words)
    if args.dataset == 'mbpp':
        return mbpp_get_num_tokens(generation, tokenizer)
    if args.dataset == 'ds1000':
        return ds1000_get_num_tokens(generation, tokenizer)
    if args.dataset == 'repo_eval':
        return repo_eval_get_num_tokens(generation, ground_truth, tokenizer)


# def get_generation_config(tokenizer, data_name):
#     if data_name == 'human_eval':
#         generation_config = human_eval._generate_config(tokenizer)
#     if data_name == 'mbpp':
#         generation_config = mbpp._generate_config(tokenizer)
#     return generation_config

def get_stop_words(data_name):
    if data_name == 'human_eval':
        return ["\ndef", "\nclass", "\nif", "\n#", "\nprint"]
    if data_name == 'mbpp':
        return ["[DONE]"]


@torch.no_grad()
def get_generations(model_name:str, args, seed=1, old_sequences=None, max_num_gen_once=args.num_generations_per_prompt,cache_dir='output'):
    device = args.device
    model, tokenizer = models.load_model_and_tokenizer(model_name, args.device, args.load_in_8bit)
    # for name, param in model.named_parameters():
    #     print(f"Layer: {name} | Shape: {param.shape} | Parameters: {param.numel()}")
    # SenSimModel = SentenceTransformer('./data/weights/nli-roberta-large')
    # bertscore = BERTScore(model_name_or_path="./data/weights/bert-base/", device="cuda")
    # llm = LLM(model_name, args.tensor_parallel_size)
    # sampling_params = SamplingParams(
    #     n=args.num_generations_per_prompt,
    #     temperature=args.temperature, 
    #     top_k=args.top_k, 
    #     top_p=args.top_p,
    #     max_tokens=args.max_new_tokens,
    # )
    # tokenizer = llm.tokenizer
    
    utils.seed_everything(seed)
    dataset = get_dataset_fn(args.dataset)(tokenizer, language=args.language)
    cleanup_code = get_clean_up_code_fn(args.dataset)
    if hasattr(dataset[0],'stopwords'):
        stop_words = dataset[0]['stopwords']
    else:
        stop_words = []
    # print(stop_words)
    
    # stop_criteria = KeywordsStoppingCriteria(stop_words, tokenizer)
    if args.fraction_of_data_to_use < 1.0:
        dataset = dataset.train_test_split(test_size=(1 - args.fraction_of_data_to_use), seed=seed)['train']
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    print('len dataset', len(dataloader))
    if old_sequences is None:
        old_sequences = []
    old_sequences = {_['task_id']: _ for _ in old_sequences}
    # print('Checking task id: ', dataset[0]['task_id'])
    # print(dataset[0])
    sequences = []
    time_start=time.time()
    for batch_idx, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        # print(batch.keys())
        task_id_path = str(batch['task_id'][0]).replace('/','_').replace('[','_').replace(']','_')
        out_dir_task_id = os.path.join(cache_dir,f"{task_id_path}.pkl")
        if batch['task_id'][0] in old_sequences:
            sequences.append(old_sequences[batch['task_id'][0]])
            continue
        if os.path.exists(out_dir_task_id):
            continue # generated
        # print(f"Batch {batch_idx} | Task ID: {batch['task_id']}")
        # print(batch)
        if os.path.exists(out_dir_task_id):
            continue # generated
        input_ids = batch['input_ids'].to(device)
        # print(f"input_ids: {input_ids}")
        print(f"input_ids shape: {input_ids.shape}")
        # print(f"attention_mask: {batch['attention_mask']}")
        # print(f"attention_mask shape: {batch['attention_mask'].shape}")
        if input_ids.shape[-1] >1000 or input_ids.shape[-1] < 9:
            continue
        input_length = input_ids.shape[1]
        if input_ids.shape[-1] >1000 or input_ids.shape[-1] < 9:
            continue
        torch.cuda.empty_cache()
        
        generations = []
        generations_decoded = []
        num_gens = args.num_generations_per_prompt
        while num_gens > 0:
            dict_outputs =  model.generate(input_ids, attention_mask=batch['attention_mask'].to(device),
                            num_beams=1, max_new_tokens=args.max_new_tokens, num_return_sequences=min(max_num_gen_once, num_gens),
                            do_sample=True, top_p=args.top_p, top_k=args.top_k,
                            temperature=args.temperature, 
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.eos_token_id,
                            # stopping_criteria=StoppingCriteriaList([stop_criteria]),
                            output_hidden_states = True, return_dict_in_generate=True, output_scores=True
                            )

            generation = dict_outputs.sequences[:, input_length:].cpu()
            print(f"Generation shape: {generation.shape}")
            
            num_tokens = get_num_tokens(generation, tokenizer, language_type=args.language, stop_words=stop_words)
            for gen, num_token in zip(generation, num_tokens):
                generations.append(gen[:num_token])
            for gen_ids in generations:
                generations_decoded.append(tokenizer.decode(gen_ids, skip_special_tokens=True))
            # print(generations_decoded)
            hidden_states = dict_outputs.hidden_states
            del dict_outputs
            gc.collect()
            torch.cuda.empty_cache()
            layers = args.layers
            layer_embeddings = {}
            for layer in layers:
                if layer == -2:
                    layer_embeddings[layer] = getMiddleLayerEmbeddingEachToken(hidden_states, num_tokens)
                else:
                    layer_embeddings[layer] = getLayerEmbeddingEachToken(hidden_states, num_tokens, layer)
            del hidden_states
            gc.collect()
            torch.cuda.empty_cache()
            num_gens -= len(generation)
            

        # # remember the data
        # print(batch['task_id'][0])
        curr_seq = dict(
            prompt=tokenizer.decode(input_ids.cpu()[0], skip_special_tokens=True),
            id=batch['task_id'][0],
            problem=batch['original_prompt'][0],
        )
        curr_seq.update(
            dict(
                layer_embeddings = layer_embeddings,
                generations=generations_decoded,
                generations_ids=generations,
                num_tokens=num_tokens
            )
        )
        
        print("Prompt:", tokenizer.decode(input_ids.cpu()[0], skip_special_tokens=True))
        print("Problem:", batch['original_prompt'][0])
        print("AnswerGT:", batch['canonical_solution'][0])
        print("MostLikelyAns:", tokenizer.decode(curr_seq['generations_ids'][0], skip_special_tokens=True))
        
        print("Prompt:", tokenizer.decode(input_ids.cpu()[0], skip_special_tokens=True), file=logInfo)
        print("Problem:", batch['original_prompt'][0], file=logInfo)
        print("AnswerGT:", batch['canonical_solution'][0], file=logInfo)
        print("MostLikelyAns:", tokenizer.decode(curr_seq['generations_ids'][0], skip_special_tokens=True), file=logInfo)
        print("\n","\n","\n", file=logInfo)
        sequences.append(curr_seq)
        
        # pickle.dump(curr_seq,open(out_dir_task_id,'wb'))
        
        torch.cuda.empty_cache()
    return sequences


def find_sublist(gen_tensor_ids, stop_word_ids):
    gen_tensor_ids = gen_tensor_ids.to('cuda')
    stop_word_tensor_ids = torch.tensor(stop_word_ids).to('cuda')
    len_gen = gen_tensor_ids.size(0)
    len_stop_word = stop_word_tensor_ids.size(0)
    if len_stop_word > len_gen:
        first_index = -1
    else:
        windows = gen_tensor_ids.unfold(0, len_stop_word, 1)
        matches = (windows == stop_word_tensor_ids).all(dim=1)
        first_index = torch.where(matches)[0][0].item() if matches.any() else -1
    
    return first_index

def repo_eval_get_num_tokens(generation, ground_truth, tokenizer, language_type='python', stop_words=[]):
    # stop_words = stop_words + ["</code>", "# SOLUTION END", "\nEND SOLUTION", ]
    generation_decoded = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generation]
    
    tokenizer_stop_words = [tokenizer.encode(_)[1:] for _ in stop_words] + [[tokenizer.eos_token_id]]
    num_tokens = []
    for ids in generation:
        min_stop_idx = len(ids)
        for stop_word in tokenizer_stop_words:
            stop_index = find_sublist(ids, stop_word)
            if 0 <= stop_index < min_stop_idx:
                min_stop_idx = stop_index
        num_tokens.append(min_stop_idx)
    return num_tokens

def ds1000_get_num_tokens(generation, tokenizer, language_type='python', stop_words=[]):
    stop_words = stop_words + ["</code>", "# SOLUTION END", "\nEND SOLUTION", ]
    tokenizer_stop_words = [tokenizer.encode(_)[1:] for _ in stop_words] + [[tokenizer.eos_token_id]]
    num_tokens = []
    for ids in generation:
        min_stop_idx = len(ids)
        for stop_word in tokenizer_stop_words:
            stop_index = find_sublist(ids, stop_word)
            if 0 <= stop_index < min_stop_idx:
                min_stop_idx = stop_index
        num_tokens.append(min_stop_idx)
    return num_tokens

def mbpp_get_num_tokens(generation, tokenizer, language_type='python', stop_words=["[DONE]"]):
    stop_words = ["[DONE]"]
    tokenizer_stop_words = [tokenizer.encode(_)[1:] for _ in stop_words] + [[tokenizer.eos_token_id]]
    num_tokens = []
    for ids in generation:
        min_stop_idx = len(ids)
        for stop_word in tokenizer_stop_words:
            stop_index = find_sublist(ids, stop_word)
            if 0 <= stop_index < min_stop_idx:
                min_stop_idx = stop_index
        num_tokens.append(min_stop_idx)
    return num_tokens

def humaneval_get_num_tokens(generation, tokenizer, language_type='python', stop_words=[]):
    if language_type == 'python':
        stop_words = ["\ndef", "\nclass", "\nif", "\n#", "\nprint"]
    elif language_type == 'ts':
        stop_words = stop_words + ["\nexport", "\nimport", "\nexport default", "\nimport default", "\nconsole.log"]
    tokenizer_stop_words = [tokenizer.encode(_)[1:] for _ in stop_words] + [[tokenizer.eos_token_id]]
    num_tokens = []
    for ids in generation:
        min_stop_idx = len(ids)
        for stop_word in tokenizer_stop_words:
            stop_index = find_sublist(ids, stop_word)
            if 0 <= stop_index < min_stop_idx:
                min_stop_idx = stop_index
        num_tokens.append(min_stop_idx)
    return num_tokens

def cleanup_code(
    code: str,
    language_type: str = None,
    dataset: str = None,
    issft: bool = False,
    stop_words = []
):
    """
    Cleans up the generated code.
    """

    if language_type.lower() == "python":
        if issft:
            code = _clean_python_code_for_sft(code)
        stop_words = ["\ndef", "\nclass", "\nif", "\n#", "\nprint"]
        code = _truncate_code_at_stopwords(code, stop_words)
    elif language_type.lower() == "ts":
        code = _truncate_code_at_stopwords(code, stop_words + ["\nexport", "\nimport", "\nexport default", "\nimport default", "\nconsole.log"])
    else:
        code = _truncate_code_at_stopwords(code, stop_words)

    return code

def _clean_python_code_for_sft(code):
    code = code.replace("\r", "")
    if "```python" in code:
        code_start_idx = code.index("```python")
        code = code[code_start_idx:].replace("```python", "").strip()
        end_idx = code.find("```") if "```" in code else len(code)
        code = code[:end_idx].strip()

    return code

def _truncate_code_at_stopwords(code, stop_words):
    min_stop_idx = len(code)
    for stop_word in stop_words:
        stop_index = code.find(stop_word)
        if 0 <= stop_index < min_stop_idx:
            min_stop_idx = stop_index
    return code[:min_stop_idx]

def main(overwrite=False, continue_from=None, parallel:int=None):
    if continue_from:
        fname = os.path.basename(continue_from)
        args.__dict__ = utils.jload(continue_from.replace(fname, 'args'+fname.replace("_partial.pkl", ".json")))
        old_sequences = pd.read_pickle(continue_from)
        cache_dir = os.path.dirname(continue_from)
        run_id = int(os.path.basename(continue_from).replace("_partial.pkl", ""))
        model_name = args.model
    else:
        old_sequences = []
        model_name = args.model
        # if '/' in model_name:
        #     model_name = model_name.replace('/', '_')
        # cache_dir = os.path.join(_settings.GENERATION_FOLDER, f'{model_name}_{args.dataset}_{args.project_ind}')
        # os.makedirs(cache_dir, exist_ok=True)
        cache_dir = OUTPUT_DIR
        old_results = glob.glob(os.path.join(OUTPUT_DIR, '*.pkl'))
        old_results = [_ for _ in old_results if '_partial' not in _]
        # if len(old_results) > 0 and not overwrite:
        #     print(f'Found {len(old_results)} generations in {cache_dir}.')
        #     return
        run_id = len(old_results)
        with open(os.path.join(OUTPUT_DIR, f'args{run_id}.json'), 'w') as f:
            json.dump(args.__dict__, f)
    temp_dir = os.path.join(cache_dir,'temp')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    print(f'Generating {args.num_generations_per_prompt} generations per prompt for {model_name} on {args.dataset}...')
    print(f"Saving to {os.path.join(cache_dir, f'{run_id}.pkl')}")
    temp_dir = os.path.join(cache_dir,'temp')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    sequences = get_generations(model_name, args, seed=args.seed, old_sequences=old_sequences,cache_dir=temp_dir)
    print(f'Writing {len(sequences)} generations to {cache_dir}...')
    
    pd.to_pickle(sequences, os.path.join(cache_dir, f'{run_id}.pkl'))
    return

if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    task_runner = main(parallel=args.nprocess)
