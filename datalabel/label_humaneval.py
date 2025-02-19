import pandas as pd
from tqdm import tqdm

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs

from transformers import AutoTokenizer
import json
import os
from benchmark.HumanEval.human_eval.evaluation import evaluate_functional_correctness_each_sample, evaluate_functional_correctness


data_root = "/home/trang-n/WCODELLM_MULTILANGUAGE/benchmark/HumanEval/data"
continue_from = '/home/trang-n/WCODELLM_MULTILANGUAGE/output/1731338674896_deepseek-ai_deepseek-coder-6.7b-base_human_eval_java_-1_-2/0.pkl'
kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters=True)]
accelerator = Accelerator(mixed_precision="bf16", kwargs_handlers=kwargs_handlers)
model_name = 'deepseek-ai/deepseek-coder-6.7b-base'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True) 

# sequences = pd.read_pickle(continue_from)
sequences = pd.read_parquet(continue_from).to_dict(orient='records')

print(f'Loaded {len(sequences)} indices')
batch_size = 8
language = 'java'
log_dir = 'tmp/humaneval'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

test_run_results = []
totalnum = len(sequences) * len(sequences[0]['generations_ids'])
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
        for j in range(len(sequence['generations_ids'])):
            indices = sequence['generations_ids'][j]
            suffixprediction = tokenizer.decode(indices, skip_special_tokens=True)
            task_id = sequence['id']
            completion_id = str(task_id) + '_' + str(j)
            # print(completion_id)
            res = {
                "task_id": task_id, 
                "generation": suffixprediction, 
                "prompt": sequence['problem'], 
                "completion_id": completion_id
            }
            
            # res = {"task_id": task_id, "generation": suffixprediction, "completion_id": completion_id}
            batch_lines.append(res)
            tmpfile.write(json.dumps(res) + "\n")
            tmpfile.flush()
            currentnum += 1
    results = evaluate_functional_correctness_each_sample(input_file=log_file, problem_file=os.path.join(data_root, f"humaneval-{language}.jsonl"), tmp_dir=log_dir, timeout=timeout, language=runlang, n_workers=8)

    
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
results.to_pickle(continue_from.replace(".pkl", "_label.pkl"))
