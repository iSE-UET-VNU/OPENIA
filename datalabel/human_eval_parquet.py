import pandas as pd
from tqdm import tqdm
import datasets
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs

from transformers import AutoTokenizer
import json
import os
from benchmark.HumanEval.human_eval.evaluation import evaluate_functional_correctness_each_sample, evaluate_functional_correctness

import dataeval.w_humaneval as human_eval
data_root = "benchmark/HumanEval/data"
continue_from = '/drive2/tuandung/WCODELLM/human_eval_result/deepseek-6.7b/rs-1-2/human_eval_rs_deepseek_6.7b_last_token_-1_-2.parquet'
kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters=True)]
accelerator = Accelerator(mixed_precision="bf16", kwargs_handlers=kwargs_handlers)
model_name = 'deepseek-ai/deepseek-coder-6.7b-base'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True) 

# sequences = pd.read_pickle(continue_from)
sequences = pd.read_parquet(continue_from).to_dict(orient='records')

print(f'Loaded {len(sequences)} indices')
batch_size = 10
language = 'rs'
log_dir = 'tmp/humaneval'
human_eval_ds = datasets.load_from_disk(human_eval._save_dataset(language, sft=False))
test_run_results = []
totalnum = len(sequences)
# totalnum = 164 * 10
totalpass = 0
currentnum = 0
total_samples = []
for idx in tqdm(range(0, len(sequences), batch_size)):
    log_file = os.path.join(log_dir,
                                    f'{model_name.replace("/", "_")}_{idx}_shot_log_{language}.json')
    tmpfile = open(log_file, "w")
    # with open(log_file, "r") as f:
    #     batch_lines = [json.loads(line) for line in f.readlines()]
    batch_lines = []
    timeout = 10
    runlang = language
    for sequence in sequences[idx:idx + batch_size]:
        # indices = sequence['generations_ids']
        # suffixprediction = tokenizer.decode(indices, skip_special_tokens=True)
        suffixprediction = sequence['generation']
        task_id = sequence['task_id']
        completion_id = sequence['completion_id']
        for human_eval_sample in human_eval_ds:
            if human_eval_sample['task_id'] == task_id:
                prompt = human_eval_sample['prompt']
                break
        # print(completion_id)
        res = {
            "task_id": task_id, 
            "generation": suffixprediction, 
            "prompt": prompt, 
            "completion_id": completion_id
        }
        
        # res = {"task_id": task_id, "generation": suffixprediction, "completion_id": completion_id}
        batch_lines.append(res)
        tmpfile.write(json.dumps(res) + "\n")
        tmpfile.flush()
        currentnum += 1
    results = evaluate_functional_correctness_each_sample(input_file=log_file, problem_file=os.path.join(data_root, f"humaneval-{language}.jsonl"), tmp_dir=log_dir, timeout=timeout, language=runlang, n_workers=1)

    
    # print("Prompt", batch_lines[0]['prompt'])
    for line in batch_lines:
        test_run_results.append({
            "task_id": line['task_id'],
            "completion_id": line['completion_id'],
            "label": results[line['completion_id']][0][1]['passed']
        })
        if results[line['completion_id']][0][1]['passed']:
            totalpass += 1
    print(f"Total pass: {totalpass}, Current num: {currentnum}")
    currentnum += len(batch_lines)
    tmpfile.close()
    for line in batch_lines:
        total_samples.append(line)
# print(len(total_samples))
# with open(os.path.join(log_dir, f"humaneval-{language}.jsonl"), "w") as f:
#     for line in total_samples:
#         f.write(json.dumps(line) + "\n")
# timeout = 10
# runlang = language
# logfilepath = os.path.join(log_dir, f"humaneval-{language}.jsonl")
# rs = evaluate_functional_correctness(input_file=logfilepath, problem_file=os.path.join(data_root, f"humaneval-{language}.jsonl"), tmp_dir=log_dir, timeout=timeout, language=runlang)
# print(rs)
    
results = pd.DataFrame(test_run_results)
print(totalpass)
print(totalnum)
results.to_pickle(continue_from.replace(".parquet", "_label.pkl"))
