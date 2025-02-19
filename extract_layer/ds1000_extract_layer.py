import pandas as pd
from tqdm import tqdm
import random

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs

from transformers import AutoTokenizer
import json
import os
data_root = "/home/trang-n/WCODELLM_MULTILANGUAGE/benchmark/DS_1000/data"
continue_from = '/home/trang-n/WCODELLM_MULTILANGUAGE/output/1730930288024_deepseek-ai_deepseek-coder-6.7b-base_ds1000_python_-1_-2/0.pkl'

sequences = pd.read_pickle(continue_from)
print(f'Loaded {len(sequences)} indices')
batch_size = 2

results = pd.DataFrame(columns=["task_id", "completion_id", "num_tokens", "generation", "last_token_middle_layer", "last_token_last_layer"])

print(sequences[0]['layer_embeddings'].keys())

for idx in tqdm(range(0, len(sequences), batch_size)):
    # batch_lines = []
    for sequence in sequences[idx:idx + batch_size]:
        for j in range(len(sequence['generations_ids'])):
            task_id = sequence['id'].numpy().tolist()
            completion_id = str(task_id) + '_' + str(j)
            numtoken = sequence['num_tokens'][j]
            generation = sequence["generations"][j]
            try:
                last_token_middle_layer = sequence['layer_embeddings'][-2][j][-2]
                last_token_last_layer = sequence['layer_embeddings'][-1][j][-2]
            
                results = results._append({
                    "task_id": task_id, 
                    "completion_id": completion_id,
                    "num_tokens": numtoken,
                    "generation": generation, 
                    "last_token_middle_layer": last_token_middle_layer,
                    "last_token_last_layer": last_token_last_layer
                }, 
                ignore_index=True)
            except:
                print(task_id)
                print(len(sequence['layer_embeddings'][-2][j]))
                print(numtoken)
                print(generation)
                print("---------------------")

# results.to_pickle(os.path.join(data_root, 'mbpp_data_random_token_classify_dataset.pkl'))
results.to_parquet(os.path.join(data_root, 'ds1000_deepseek_6.7b_last_token_-1_-2_kalkds.parquet'))
print("Done!")
if __name__ == '__main__':
    print("Done!")