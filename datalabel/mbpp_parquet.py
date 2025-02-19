import pandas as pd
from tqdm import tqdm

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs

from transformers import AutoTokenizer
import json
import os
from benchmark.MBPP.human_eval.evaluation import evaluate_functional_correctness_each_sample

data_root = "benchmark/MBPP/data"
continue_from = '/drive2/tuandung/WCODELLM/mbpp_result/mbpp_deepseek_6.7b_last_token_-1_-2.parquet'
kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters=True)]
accelerator = Accelerator(mixed_precision="bf16", kwargs_handlers=kwargs_handlers)
model_name = 'deepseek-ai/deepseek-coder-6.7b-base'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True) 

sequences = pd.read_parquet(continue_from).to_dict(orient='records')
print(f'Loaded {len(sequences)} indices')
batch_size = 10
language = 'python'
log_dir = 'tmp/'

test_run_results = []
totalnum = len(sequences)
totalpass = 0
currentnum = 0
for idx in tqdm(range(0, len(sequences), batch_size)):
    log_file = os.path.join(log_dir,
                                    f'{model_name.replace("/", "_")}_{idx}_shot_log_{language}.json')
    tmpfile = open(log_file, "w")
    batch_lines = []
    for sequence in sequences[idx:idx + batch_size]:
        # indices = sequence['generations_ids']
        # suffixprediction = tokenizer.decode(indices, skip_special_tokens=True)
        suffixprediction = sequence['generation']
        # print(sequence['id'])
        # task_id = sequence['id'].numpy().tolist()
        task_id = sequence['task_id']
        completion_id = sequence['completion_id']
        res = {"task_id": task_id, "generation": suffixprediction, "completion_id": completion_id}
        batch_lines.append(res)
        tmpfile.write(json.dumps(res) + "\n")
        tmpfile.flush()
        currentnum += 1
    tmpfile.close()
    timeout = 10
    runlang = language
    results = evaluate_functional_correctness_each_sample(input_file=log_file, problem_file=os.path.join(data_root, f"mbpp_test.jsonl"), tmp_dir=log_dir, timeout=timeout, language=runlang, n_workers=8)
            # tmpfile.write(json.dumps(res) + "\n")
            # tmpfile.flush()
            # totoalnum += 1
    for line in batch_lines:
        test_run_results.append({
            "task_id": line['task_id'],
            "completion_id": line['completion_id'],
            "label": results[line['completion_id']][0][1]['passed']
        })
        if results[line['completion_id']][0][1]['passed']:
            totalpass += 1
    print(f"Total pass: {totalpass}, Current num: {currentnum}")
results = pd.DataFrame(test_run_results)
print(totalpass)
print(totalnum)
results.to_pickle(continue_from.replace(".parquet", "_label.pkl"))