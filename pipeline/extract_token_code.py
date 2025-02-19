import os
import pickle
import argparse
import tqdm
import torch
import pandas as pd

import models
import dataeval.w_humaneval as human_eval
import dataeval.w_mbpp as mbpp
import dataeval.w_deveval as dev_eval
from dataeval.w_humaneval import extract_generation_code as human_eval_egc
from dataeval.w_mbpp import extract_generation_code as mbpp_eval_egc
from dataeval.w_deveval import extract_generation_code as deveval_eval_egc

from func.metric import *


parser = argparse.ArgumentParser()
parser.add_argument('--generate_dir', type=str, default='generate')
# parser.add_argument('--output_dir', type=str, default='output')
parser.add_argument('--dataset', type=str, default='human_eval')
parser.add_argument('--model_name', type=str, default='opt-13b')
parser.add_argument("--language", default="python", type=str,)
parser.add_argument("--layers", default=[-1], type=int, nargs='+')
parser.add_argument("--type", default="LFCLF", type=str)
args = parser.parse_args()
   

def get_dataset_fn(data_name):
    if data_name == 'human_eval':
        return human_eval.get_dataset
    if data_name == 'mbpp':
        return mbpp.get_dataset
    if data_name == 'dev_eval':
        return dev_eval.get_dataset
    raise ValueError(f"Unknown dataset {data_name}")

def extract_generation_code_fun(data_name):
    if data_name == 'human_eval':
        return human_eval_egc
    if data_name == 'mbpp':
        return mbpp_eval_egc
    if data_name == 'dev_eval':
        return deveval_eval_egc

def process_lfclf():
    tokenizer = models.load_tokenizer(args.model_name)
    if 'chat' or 'instruct' in args.model_name.lower():
        instruction = True
    else:
        instruction = False
    dataset = get_dataset_fn(args.dataset)(tokenizer, language=args.language, instruction=instruction)
    dataset_egc = extract_generation_code_fun(args.dataset)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    output_dir = args.generate_dir.replace('temp', 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    clean_generations_range_all = {}
    
    for example in tqdm.tqdm(dataset, total=len(dataset)):
        has_error = False
        task_id_path =  str(example['task_id']).replace('/','_').replace('[','_').replace(']','_')
        if args.dataset == 'mbpp' or args.dataset == 'ds1000':
            task_id_path = f'tensor({task_id_path})'
        task_generation_seqs_path = f'generation_sequences_output_{task_id_path}.pkl'
        task_generation_seqs_path = os.path.join(args.generate_dir, task_generation_seqs_path)
        # print(task_generation_seqs_path)
        if not os.path.exists(task_generation_seqs_path):
            print(f'File {task_id_path} not found. Skipping...')
            continue
        
        # print(f'Found {task_id_path}. Processing...')
        
        with open(task_generation_seqs_path, 'rb') as f:
            task_generation_seqs = pickle.load(f)
        
        clean_generations_range = []
        for generated_ids in task_generation_seqs['generations_ids']:
            gen = tokenizer.decode(generated_ids, skip_special_tokens=True)
            clean_generation_decoded = dataset_egc(example, gen, args.language)
            start_ind, end_ind = getCleanGenerationRange(generated_ids.tolist(), clean_generation_decoded, tokenizer)
            if start_ind is None or end_ind is None:
                has_error = True
                # print("gen:", gen)
                # print("clean_generation_decoded:", clean_generation_decoded)
                print(f'Cannot find clean generation range for {task_id_path}')
                clean_generations_range.append((getGenerationRange(generated_ids.tolist(), tokenizer), has_error))
            else:
                clean_generations_range.append((start_ind, end_ind, has_error))

        clean_generations_range_all[task_id_path] = clean_generations_range
    # print(clean_generations_range_all)
    for layer in args.layers:
        print(f'Processing layer {layer}')
        results = pd.DataFrame(columns=[
            "task_id", 
            "completion_id", 
            "num_tokens", 
            "generation", 
            "first_token_embedding", 
            "last_token_embedding",
            "first_token_code_embedding",
            "last_token_code_embedding",
            "has_error",
            "first_code_index",
            "last_code_index",
            "first_index",
            "last_index",
            "generated_ids"
        ])
        found_sample = 0
        for example in tqdm.tqdm(dataset, total=len(dataset)):
            task_id_path =  str(example['task_id']).replace('/','_').replace('[','_').replace(']','_')
            if args.dataset == 'mbpp' or args.dataset == 'ds1000':
                task_id_path = f'tensor({task_id_path})'
            task_generation_seqs_path = f'generation_sequences_output_{task_id_path}.pkl'
            task_generation_seqs_path = os.path.join(args.generate_dir, task_generation_seqs_path)
            # if task_id_path == 'arctic.hooks.register_get_auth_hook':
            #     break
            if not os.path.exists(task_generation_seqs_path):
                # print(f'File {task_id_path} not found. Skipping...')
                continue
            with open(task_generation_seqs_path, 'rb') as f:
                task_generation_seqs = pickle.load(f)
            found_sample += 1
            clean_generations_range = clean_generations_range_all[task_id_path]
            task_embedding_path = f'all_token_embedding_{task_id_path}_{layer}.pkl'
            task_embedding_path = os.path.join(args.generate_dir, task_embedding_path)
            if not os.path.exists(task_embedding_path):
                print(f'File {task_id_path} {layer} not found. Skipping...')
                continue
            
            with open(task_embedding_path, 'rb') as f:
                task_embedding = pickle.load(f)
            
            task_last_token_embedding = []
            for j in range(len(task_generation_seqs['generations'])):
                task_id = example['task_id']
                completion_id = str(task_id) + '_' + str(j)
                # num_tokens = task_generation_seqs['num_tokens'][j]
                generation = task_generation_seqs["generations"][j]
                generated_ids = task_generation_seqs["generations_ids"][j]
                start_code_ind, end_code_ind, has_error = clean_generations_range[j]
                start_ind, end_ind = getGenerationRange(generated_ids.tolist(), tokenizer)
                num_tokens = end_ind - start_ind
                layer_embedding = task_embedding['layer_embeddings'][j]

                extracted_code = tokenizer.decode(generated_ids.tolist()[start_code_ind:end_code_ind], skip_special_tokens=True)

                start_ind = max(0, start_ind - 1)
                end_ind = end_ind - 1
                start_code_ind = max(0, start_code_ind)
                end_code_ind = end_code_ind - 1
                
                first_token_embedding = layer_embedding[start_ind].tolist()
                last_token_embedding = layer_embedding[end_ind - 1].tolist()
                first_token_code_embedding = layer_embedding[start_code_ind].tolist()
                last_token_code_embedding = layer_embedding[end_code_ind - 1].tolist()

                results = results._append({
                    "task_id": task_id, 
                    "completion_id": completion_id,
                    "num_tokens": num_tokens,
                    "generation": generation, 
                    "first_token_embedding": first_token_embedding, 
                    "last_token_embedding": last_token_embedding,
                    "first_token_code_embedding": first_token_code_embedding,
                    "last_token_code_embedding": last_token_code_embedding,
                    "has_error": has_error,
                    "extracted_code": extracted_code,
                    "first_code_index": start_code_ind,
                    "last_code_index" : end_code_ind - 1,
                    "first_index": start_ind,
                    "last_index": end_ind - 1,
                    "generated_ids": generated_ids.tolist()
                }, 
                ignore_index=True)

        print(f'Found {found_sample} / {len(dataset)}')
        model_name = args.model_name.replace('/', '_')
        results.to_parquet(os.path.join(output_dir, f'LFCLF_embedding_{args.dataset}_{model_name}_{layer}.parquet'))
            
    return

def process_last_line():
    # import tree_sitter_python as tspython
    from tree_sitter import Language, Parser
    code_parser = Parser()
    PY_LANGUAGE = Language('/home/trang-n/WCODELLM_MULTILANGUAGE/build/my-languages.so', 'python')
    # PY_LANGUAGE = Language(tspython.language())
    # code_parser = Parser(PY_LANGUAGE)
    code_parser.set_language(PY_LANGUAGE)
    tokenizer = models.load_tokenizer(args.model_name)
    if 'chat' or 'instruct' in args.model_name.lower():
        instruction = True
    else:
        instruction = False
    dataset = get_dataset_fn(args.dataset)(tokenizer, language=args.language, instruction=instruction)
    dataset_egc = extract_generation_code_fun(args.dataset)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    output_dir = args.generate_dir.replace('temp', 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    last_line_token_ids_list_all = {}
    
    for example in tqdm.tqdm(dataset, total=len(dataset)):
        has_error = False
        task_id_path =  str(example['task_id']).replace('/','_').replace('[','_').replace(']','_')
        if args.dataset == 'mbpp' or args.dataset == 'ds1000':
            task_id_path = f'tensor({task_id_path})'
        if args.dataset == 'dev_eval':
            function_name = example['task_id'].split('.')[-1]
        elif args.dataset == 'human_eval':
            function_name = get_function_name(example["original_prompt"].strip(), args.language)
        else:
            # raise ValueError(f"Not support dataset {args.dataset} yet.")
            function_name = None
            # function_name = get_function_name(, args.language)
        task_generation_seqs_path = f'generation_sequences_output_{task_id_path}.pkl'
        task_generation_seqs_path = os.path.join(args.generate_dir, task_generation_seqs_path)
        # print(task_generation_seqs_path)
        if not os.path.exists(task_generation_seqs_path):
            print(f'File {task_id_path} not found. Skipping...')
            continue
        
        # print(f'Found {task_id_path}. Processing...')
        
        with open(task_generation_seqs_path, 'rb') as f:
            task_generation_seqs = pickle.load(f)
        
        last_line_token_ids_list = []
        for generated_ids in task_generation_seqs['generations_ids']:
            gen = tokenizer.decode(generated_ids, skip_special_tokens=True)
            clean_generation_decoded = dataset_egc(example, gen, args.language)
            if function_name is None:
                function_name = get_function_name(clean_generation_decoded, args.language)
            last_line_token_ids = getLineGenerationTokens(generated_ids.tolist(), clean_generation_decoded, tokenizer, code_parser, function_name)
            last_line_token_ids_list.append(last_line_token_ids)

        last_line_token_ids_list_all[task_id_path] = last_line_token_ids_list
    
    for layer in args.layers:
        print(f'Processing layer {layer}')
        results = pd.DataFrame(columns=[
            "task_id", 
            "completion_id",
            "generation", 
            "generated_ids",
            "last_line_token_embeddings"
        ])
        found_sample = 0
        for example in tqdm.tqdm(dataset, total=len(dataset)):
            task_id_path =  str(example['task_id']).replace('/','_').replace('[','_').replace(']','_')
            if args.dataset == 'mbpp' or args.dataset == 'ds1000':
                task_id_path = f'tensor({task_id_path})'
            task_generation_seqs_path = f'generation_sequences_output_{task_id_path}.pkl'
            task_generation_seqs_path = os.path.join(args.generate_dir, task_generation_seqs_path)
            if not os.path.exists(task_generation_seqs_path):
                continue
            with open(task_generation_seqs_path, 'rb') as f:
                task_generation_seqs = pickle.load(f)
            found_sample += 1
            last_line_token_ids_list = last_line_token_ids_list_all[task_id_path]
            task_embedding_path = f'all_token_embedding_{task_id_path}_{layer}.pkl'
            task_embedding_path = os.path.join(args.generate_dir, task_embedding_path)
            if not os.path.exists(task_embedding_path):
                print(f'File {task_id_path} {layer} not found. Skipping...')
                continue
            
            with open(task_embedding_path, 'rb') as f:
                task_embedding = pickle.load(f)
            
            for j in range(len(task_generation_seqs['generations'])):
                task_id = example['task_id']
                completion_id = str(task_id) + '_' + str(j)
                generation = task_generation_seqs["generations"][j]
                generated_ids = task_generation_seqs["generations_ids"][j]
                last_line_token_ids = last_line_token_ids_list[j]
                layer_embedding = task_embedding['layer_embeddings'][j]
                last_line_token_embeddings = []
                for id in last_line_token_ids:
                    # chosen_id = max(0, id - 1)
                    chosen_id = min(len(layer_embedding) - 1, id + 1)
                    last_line_token_embeddings.append(layer_embedding[chosen_id].tolist())
                results = results._append({
                    "task_id": task_id, 
                    "completion_id": completion_id,
                    "generation": generation,
                    "generated_ids": generated_ids.tolist(),
                    "last_line_token_embeddings": last_line_token_embeddings,
                    "last_line_token_ids": last_line_token_ids
                }, 
                ignore_index=True)
        
        print(f'Found {found_sample} / {len(dataset)}')
        model_name = args.model_name.replace('/', '_')
        results.to_parquet(os.path.join(output_dir, f'last_2_token_line_embedding_{args.dataset}_{model_name}_{layer}.parquet'))
    
    return

def process_min_prob_token():

    tokenizer = models.load_tokenizer(args.model_name)
    if 'chat' or 'instruct' in args.model_name.lower():
        instruction = True
    else:
        instruction = False
    dataset = get_dataset_fn(args.dataset)(tokenizer, language=args.language, instruction=instruction)
    dataset_egc = extract_generation_code_fun(args.dataset)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    output_dir = args.generate_dir.replace('temp', 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    clean_generations_range_all = {}
    
    for example in tqdm.tqdm(dataset, total=len(dataset)):
        has_error = False
        task_id_path =  str(example['task_id']).replace('/','_').replace('[','_').replace(']','_')
        if args.dataset == 'mbpp' or args.dataset == 'ds1000':
            task_id_path = f'tensor({task_id_path})'
        # if args.dataset == 'dev_eval':
        #     function_name = example['task_id'].split('.')[-1]
        # elif args.dataset == 'human_eval':
        #     function_name = get_function_name(example["original_prompt"].strip(), args.language)
        # else:
        #     # raise ValueError(f"Not support dataset {args.dataset} yet.")
        #     function_name = None
        #     # function_name = get_function_name(, args.language)
        task_generation_seqs_path = f'generation_sequences_output_{task_id_path}.pkl'
        task_generation_seqs_path = os.path.join(args.generate_dir, task_generation_seqs_path)
        # print(task_generation_seqs_path)
        if not os.path.exists(task_generation_seqs_path):
            print(f'File {task_id_path} not found. Skipping...')
            continue
        
        # print(f'Found {task_id_path}. Processing...')
        
        with open(task_generation_seqs_path, 'rb') as f:
            task_generation_seqs = pickle.load(f)
        
        clean_generations_range = []
        for generated_ids, prob_scores in zip(task_generation_seqs['generations_ids'], task_generation_seqs['softmax_scores']):
            gen = tokenizer.decode(generated_ids, skip_special_tokens=True)
            clean_generation_decoded = dataset_egc(example, gen, args.language)
            start_ind, end_ind = getCleanGenerationRange(generated_ids.tolist(), clean_generation_decoded, tokenizer)
            if start_ind is None or end_ind is None:
                has_error = True
                # print("gen:", gen)
                # print("clean_generation_decoded:", clean_generation_decoded)
                print(f'Cannot find clean generation range for {task_id_path}')
                start_ind, end_ind = getGenerationRange(generated_ids.tolist(), tokenizer)
            
            # clean_generations_range.append((start_ind, end_ind, has_error))
            clean_generations_prob_scores = torch.tensor(prob_scores[start_ind:end_ind])
            prob_score , min_prob_ind = torch.min(clean_generations_prob_scores, dim=0)
            clean_generations_range.append((start_ind, end_ind, int(min_prob_ind.item()), prob_score.item()))
            
            

        clean_generations_range_all[task_id_path] = clean_generations_range
    
    for layer in args.layers:
        print(f'Processing layer {layer}')
        results = pd.DataFrame(columns=[
            "task_id", 
            "completion_id",
            "generation", 
            "generated_ids",
            "min_prob_token_embeddings",
            "min_prob_token_ind",
            "min_prob_token_score",
            "seq_prob_scores",
        ])
        found_sample = 0
        for example in tqdm.tqdm(dataset, total=len(dataset)):
            task_id_path =  str(example['task_id']).replace('/','_').replace('[','_').replace(']','_')
            if args.dataset == 'mbpp' or args.dataset == 'ds1000':
                task_id_path = f'tensor({task_id_path})'
            task_generation_seqs_path = f'generation_sequences_output_{task_id_path}.pkl'
            task_generation_seqs_path = os.path.join(args.generate_dir, task_generation_seqs_path)
            if not os.path.exists(task_generation_seqs_path):
                continue
            with open(task_generation_seqs_path, 'rb') as f:
                task_generation_seqs = pickle.load(f)
            found_sample += 1
            clean_generations_range = clean_generations_range_all[task_id_path]
            task_embedding_path = f'all_token_embedding_{task_id_path}_{layer}.pkl'
            task_embedding_path = os.path.join(args.generate_dir, task_embedding_path)
            if not os.path.exists(task_embedding_path):
                print(f'File {task_id_path} {layer} not found. Skipping...')
                continue
            
            with open(task_embedding_path, 'rb') as f:
                task_embedding = pickle.load(f)
            
            for j in range(len(task_generation_seqs['generations'])):
                task_id = example['task_id']
                completion_id = str(task_id) + '_' + str(j)
                generation = task_generation_seqs["generations"][j]
                generated_ids = task_generation_seqs["generations_ids"][j]
                seq_prob_scores = torch.tensor(task_generation_seqs['softmax_scores'][j]).numpy().tolist()
                start_code_ind, end_code_ind, min_prob_ind, min_prob_score = clean_generations_range[j]
                layer_embedding = task_embedding['layer_embeddings'][j]
                min_prob_ind = start_code_ind + min_prob_ind
                min_prob_line_token_embeddings = layer_embedding[max(0, min_prob_ind - 1)].tolist()
                # print(min_prob_score.float())
                results = results._append({
                    "task_id": task_id, 
                    "completion_id": completion_id,
                    "generation": generation,
                    "generated_ids": generated_ids.tolist(),
                    "min_prob_token_embeddings": min_prob_line_token_embeddings,
                    "min_prob_token_ind": min_prob_ind,
                    "min_prob_token_score": min_prob_score,
                    "seq_prob_scores": seq_prob_scores,
                }, 
                ignore_index=True)
        
        print(f'Found {found_sample} / {len(dataset)}')
        model_name = args.model_name.replace('/', '_')
        results.to_parquet(os.path.join(output_dir, f'min_prob_token_embedding_{args.dataset}_{model_name}_{layer}.parquet'))
    
    return

def process_min_prob_token_line():
    tokenizer = models.load_tokenizer(args.model_name)
    if 'chat' or 'instruct' in args.model_name.lower():
        instruction = True
    else:
        instruction = False
    dataset = get_dataset_fn(args.dataset)(tokenizer, language=args.language, instruction=instruction)
    dataset_egc = extract_generation_code_fun(args.dataset)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    output_dir = args.generate_dir.replace('temp', 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    clean_generations_range_all = {}
    
    for example in tqdm.tqdm(dataset, total=len(dataset)):
        has_error = False
        task_id_path =  str(example['task_id']).replace('/','_').replace('[','_').replace(']','_')
        if args.dataset == 'mbpp' or args.dataset == 'ds1000':
            task_id_path = f'tensor({task_id_path})'
        # if args.dataset == 'dev_eval':
        #     function_name = example['task_id'].split('.')[-1]
        # elif args.dataset == 'human_eval':
        #     function_name = get_function_name(example["original_prompt"].strip(), args.language)
        # else:
        #     # raise ValueError(f"Not support dataset {args.dataset} yet.")
        #     function_name = None
        #     # function_name = get_function_name(, args.language)
        task_generation_seqs_path = f'generation_sequences_output_{task_id_path}.pkl'
        task_generation_seqs_path = os.path.join(args.generate_dir, task_generation_seqs_path)
        # print(task_generation_seqs_path)
        if not os.path.exists(task_generation_seqs_path):
            print(f'File {task_id_path} not found. Skipping...')
            continue
        
        # print(f'Found {task_id_path}. Processing...')
        
        with open(task_generation_seqs_path, 'rb') as f:
            task_generation_seqs = pickle.load(f)
        
        clean_generations_range = []
        for generated_ids, prob_scores in zip(task_generation_seqs['generations_ids'], task_generation_seqs['softmax_scores']):
            gen = tokenizer.decode(generated_ids, skip_special_tokens=True)
            clean_generation_decoded = dataset_egc(example, gen, args.language)
            start_ind, end_ind = getCleanGenerationRange(generated_ids.tolist(), clean_generation_decoded, tokenizer)
            if start_ind is None or end_ind is None:
                has_error = True
                # print("gen:", gen)
                # print("clean_generation_decoded:", clean_generation_decoded)
                print(f'Cannot find clean generation range for {task_id_path}')
                start_ind, end_ind = getGenerationRange(generated_ids.tolist(), tokenizer)
            last_line_token_ids = getLineGenerationTokens(generated_ids.tolist(), clean_generation_decoded, tokenizer, parser=None, function_name=None)
            range_line_list = [start_ind]
            range_line_list.extend(last_line_token_ids)
            # clean_generations_range.append((start_ind, end_ind, has_error))
            min_prob_line_list = []
            for i in range(len(range_line_list) - 1):    
                clean_generations_prob_scores = torch.tensor(prob_scores[range_line_list[i]:range_line_list[i+1] + 1])
                prob_score , min_prob_ind = torch.min(clean_generations_prob_scores, dim=0)
                min_prob_line_list.append((int(range_line_list[i] + min_prob_ind.item()), prob_score.item()))
            
            clean_generations_range.append(min_prob_line_list)
            
            

        clean_generations_range_all[task_id_path] = clean_generations_range
    
    for layer in args.layers:
        print(f'Processing layer {layer}')
        results = pd.DataFrame(columns=[
            "task_id", 
            "completion_id",
            "generation", 
            "generated_ids",
            "min_prob_token_line_embeddings",
            "min_prob_token_line_ids",
            "min_prob_token_line_scores",
            "seq_prob_scores",
        ])
        found_sample = 0
        for example in tqdm.tqdm(dataset, total=len(dataset)):
            task_id_path =  str(example['task_id']).replace('/','_').replace('[','_').replace(']','_')
            if args.dataset == 'mbpp' or args.dataset == 'ds1000':
                task_id_path = f'tensor({task_id_path})'
            task_generation_seqs_path = f'generation_sequences_output_{task_id_path}.pkl'
            task_generation_seqs_path = os.path.join(args.generate_dir, task_generation_seqs_path)
            if not os.path.exists(task_generation_seqs_path):
                continue
            with open(task_generation_seqs_path, 'rb') as f:
                task_generation_seqs = pickle.load(f)
            found_sample += 1
            clean_generations_range = clean_generations_range_all[task_id_path]
            task_embedding_path = f'all_token_embedding_{task_id_path}_{layer}.pkl'
            task_embedding_path = os.path.join(args.generate_dir, task_embedding_path)
            if not os.path.exists(task_embedding_path):
                print(f'File {task_id_path} {layer} not found. Skipping...')
                continue
            
            with open(task_embedding_path, 'rb') as f:
                task_embedding = pickle.load(f)
            
            for j in range(len(task_generation_seqs['generations'])):
                task_id = example['task_id']
                completion_id = str(task_id) + '_' + str(j)
                generation = task_generation_seqs["generations"][j]
                generated_ids = task_generation_seqs["generations_ids"][j]
                seq_prob_scores = torch.tensor(task_generation_seqs['softmax_scores'][j]).numpy().tolist()
                min_prob_line_list = clean_generations_range[j]
                layer_embedding = task_embedding['layer_embeddings'][j]
                # min_prob_ind = start_code_ind + min_prob_ind
                min_prob_line_token_embeddings = []
                min_prob_token_line_scores = []
                min_prob_token_line_ids = []
                for min_prob_ind, min_prob_score in min_prob_line_list:
                    min_prob_line_token_embeddings.append(layer_embedding[max(0, min_prob_ind - 1)].tolist())
                    min_prob_token_line_scores.append(min_prob_score)   
                    min_prob_token_line_ids.append(min_prob_ind)
                
                results = results._append({
                    "task_id": task_id, 
                    "completion_id": completion_id,
                    "generation": generation,
                    "generated_ids": generated_ids.tolist(),
                    "min_prob_token_line_embeddings": min_prob_line_token_embeddings,
                    "min_prob_token_line_ids": min_prob_token_line_ids,
                    "min_prob_token_line_scores": min_prob_token_line_scores,
                    "seq_prob_scores": seq_prob_scores,
                }, 
                ignore_index=True)
        
        print(f'Found {found_sample} / {len(dataset)}')
        model_name = args.model_name.replace('/', '_')
        results.to_parquet(os.path.join(output_dir, f'min_prob_token_line_embedding_{args.dataset}_{model_name}_{layer}.parquet'))
    
    return

if __name__ == '__main__':
    if args.type == 'LFCLF':
        process_lfclf()
    elif args.type == 'last_line':
        process_last_line()
    elif args.type == 'min_prob_token':
        process_min_prob_token()
    elif args.type == 'min_prob_token_line':
        process_min_prob_token_line()
    else:
        raise ValueError(f"Unknown type {args.type}")