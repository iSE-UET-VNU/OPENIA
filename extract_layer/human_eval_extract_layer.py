import pandas as pd
from tqdm import tqdm
import random

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs

from transformers import AutoTokenizer
import json
import os
data_root = "/home/trang-n/WCODELLM_MULTILANGUAGE/layer_extracted"
continue_from = '/home/trang-n/WCODELLM_MULTILANGUAGE/output/1730577930925_deepseek-ai_deepseek-coder-1.3b-base_human_eval_go_-1/0.pkl'
model_name = 'deepseek-ai/deepseek-coder-6.7b-base'
sequences = pd.read_pickle(continue_from)
print(f'Loaded {len(sequences)} indices')
batch_size = 2
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

results = pd.DataFrame(columns=["task_id", "completion_id", "num_tokens", "generation", "last_token_middle_layer", "last_token_last_layer"])

print(sequences[0]['layer_embeddings'].keys())

stop_words = ["\nfunc", "struct", "\n// "]

for idx in tqdm(range(0, len(sequences), batch_size)):
    # batch_lines = []
    for sequence in sequences[idx:idx + batch_size]:
        for j in range(len(sequence['generations_ids'])):
            task_id = sequence['id']
            completion_id = str(task_id) + '_' + str(j)
            numtoken = sequence['num_tokens'][j]
            generation = sequence["generations"][j]
            # print(generation.split(stop_word))
            for stop_word in stop_words:
                generation = generation.split(stop_word)[0]
            numtoken = len(tokenizer.tokenize(generation))
            # print(generation)

            try:
                # last_token_middle_layer = sequence['layer_embeddings'][-2][j][numtoken-2]
                last_token_middle_layer = 0
                last_token_last_layer = sequence['layer_embeddings'][-1][j][numtoken-2]
            
                results = results._append({
                    "task_id": task_id, 
                    "completion_id": completion_id,
                    "num_tokens": numtoken,
                    "generation": generation, 
                    "last_token_middle_layer": last_token_middle_layer,
                    "last_token_last_layer": last_token_last_layer
                }, 
                ignore_index=True)
            except Exception as e:
                print(e)
                print(task_id)
                # print(len(sequence['layer_embeddings'][-2][j]))
                print(numtoken)
                print(generation)
                print("---------------------")

# results.to_pickle(os.path.join(data_root, 'mbpp_data_random_token_classify_dataset.pkl'))
results.to_parquet(os.path.join(data_root, 'human_eval_go_deepseek_6.7b_last_token_-1.parquet'))
print("Done!")
if __name__ == '__main__':
    print("Done!")