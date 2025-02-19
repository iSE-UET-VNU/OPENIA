import pandas as pd
from tqdm import tqdm
import random

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs

from transformers import AutoTokenizer
import json
import os
data_root = "/home/trang-n/WCODELLM_MULTILANGUAGE/layer_extracted"
continue_from = '/home/trang-n/WCODELLM_MULTILANGUAGE/output/1731339433291_deepseek-ai_deepseek-coder-6.7b-base_human_eval_rs_-1_-2/0.pkl'

# sequences = pd.read_pickle(continue_from)
# print(f'Loaded {len(sequences)} indices')
batch_size = 1

results = pd.DataFrame(columns=["task_id", "completion_id", "num_tokens", "generation", "last_token_middle_layer", "last_token_last_layer"])

# print(sequences[0]['layer_embeddings'].keys())

for idx in tqdm(range(11, 511, batch_size)):
    # batch_lines = []
    continue_from = f'/home/trang-n/WCODELLM_MULTILANGUAGE/output/1731360350565_deepseek-ai_deepseek-coder-6.7b-base_mbpp_python_-1_-2/temp/tensor({idx}).pkl'
    try: 
        sequence = pd.read_pickle(continue_from)
        for j in range(len(sequence['generations_ids'])):
            task_id = int(sequence['id'].numpy().astype(int))
            # print(task_id)
            completion_id = str(task_id) + '_' + str(j)
            numtoken = sequence['num_tokens'][j]
            generation = sequence["generations"][j]
            last_token_middle_layer = sequence['layer_embeddings'][-2][j][-2]
            # last_token_middle_layer = 0
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
            # print(len(results))
    except Exception as e:
        print(e)
# results.to_pickle(os.path.join(data_root, 'mbpp_data_random_token_classify_dataset.pkl'))
results.to_parquet(os.path.join(data_root, 'mbpp_deepseek_6.7b_last_token_-1_-2.parquet'))
print("Done!")
if __name__ == '__main__':
    print("Done!")