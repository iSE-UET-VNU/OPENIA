import re

languge_settings = {
    'python': {
        'full_name': 'Python',
        'indent': 4,
    },
    'cpp': {
        'full_name': 'cpp',
        'indent': 0,
        'main': "int main()",
    },
    'java': {
        'full_name': 'Java',
        'indent': 4,
        'main': "public static void main",
    },
    'cs': {
        'full_name': "csharp",
        'indent': 0,
        'main': "public static void Main",
    },
    'php': {
        'full_name': "PHP",
        'indent': 0,
    },
    'ts': {
        'full_name': "TypeScript",
        'indent': 0,
    },
    'js': {
        'full_name': "JavaScript",
        'indent': 0
    },
    'sh': {
        'full_name': "Bash",
        'indent': 0
    }
}

def get_function_name(question: str, lang: str):
    func_lines = [x for x in question.strip().split('\n') if x.strip()]

    if lang.lower() == 'python':
        func_idx = [i for i in range(len(func_lines)) if func_lines[i].startswith("def ")][-1]
        func_name = func_lines[func_idx].split('(')[0].strip()
        func_prefix = "\n".join(func_lines[:func_idx])
        return func_name, func_prefix
    
    func_name = func_lines[-1].split('{')[0].strip()
    func_prefix = "\n".join(func_lines[:-1])
    return func_name, func_prefix

def extract_generation_code(gpt_completion):
    # gpt_completion = example['gpt_completion']
    generation = gpt_completion
    try:
        code_block: str = re.findall(f'```python\n(.*?)```', gpt_completion, re.DOTALL | re.IGNORECASE)[0]
        generation = code_block
    except Exception as ex:
        print("Failed to extract codeblock:\n{}".format(gpt_completion))

    # example['generation'] = generation
    return generation

import pandas as pd
from tqdm import tqdm

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs

from transformers import AutoTokenizer
import json
import os
from benchmark.MBPP.human_eval.evaluation import evaluate_functional_correctness_each_sample

data_root = "/drive2/tuandung/WCODELLM/benchmark/MBPP/data"
continue_from = '/drive2/tuandung/WCODELLM/LFCLF_embedding_mbpp_deepseek-ai_deepseek-coder-1.3b-instruct_24_label.parquet'
kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters=True)]
accelerator = Accelerator(mixed_precision="bf16", kwargs_handlers=kwargs_handlers)
model_name = 'deepseek-ai/deepseek-coder-1.3b-instruct'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# sequences = pd.read_pickle(continue_from)
sequences = pd.read_parquet(continue_from).to_dict(orient='records')
print(f'Loaded {len(sequences)} indices')
batch_size = 8
language = 'python'
log_dir = 'tmp/mbpp'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

test_run_results = []
totalnum = len(sequences)
# totalnum = 164 * 10
totalpass = 0
currentnum = 0
total_samples = []

extract_generation_list = []

for idx in tqdm(range(0, len(sequences), batch_size)):
    log_file = os.path.join(log_dir,
                                    f'{model_name.replace("/", "_")}_{idx}_shot_log_{language}.json')
    tmpfile = open(log_file, "w")
    # with open(log_file, "r") as f:
    #     batch_lines = [json.loads(line) for line in f.readlines()]
    batch_lines = []
    timeout = 3.0
    runlang = language
    for sequence in sequences[idx:idx + batch_size]:
        # suffixprediction = tokenizer.decode(i, skip_special_tokens=True)
        suffixprediction = extract_generation_code(sequence['generation'])
        task_id = sequence['task_id']
        completion_id = sequence['completion_id']
        # print(completion_id)
        extract_generation_list.append(suffixprediction)
    #     res = {
    #         "task_id": task_id, 
    #         "generation": suffixprediction, 
    #         "prompt": '', 
    #         "completion_id": completion_id
    #     }
        
    #     # res = {"task_id": task_id, "generation": suffixprediction, "completion_id": completion_id}
    #     batch_lines.append(res)
    #     tmpfile.write(json.dumps(res) + "\n")
    #     tmpfile.flush()
    #     currentnum += 1
    # results = evaluate_functional_correctness_each_sample(input_file=log_file, problem_file=os.path.join(data_root, f"mbpp_test.jsonl"), tmp_dir=log_dir, language=runlang, n_workers=8, is_mbpp=True)
    # # results = evaluate_functional_correctness_each_sample(input_file=log_file, problem_file=os.path.join(data_root, f"humaneval-{language}.jsonl"), tmp_dir=log_dir, timeout=timeout, language=runlang, n_workers=8)
    
    
    # print("Prompt", batch_lines[0]['prompt'])
    # for line in batch_lines:
    #     test_run_results.append(results[line['completion_id']][0][1]['passed'])
    #     if results[line['completion_id']][0][1]['passed']:
    #         totalpass += 1
    # print(f"Total pass: {totalpass}, Current num: {currentnum}")
    # # currentnum += len(batch_lines)
    # tmpfile.close()
    # for line in batch_lines:
    #     total_samples.append(line)
    
results = pd.DataFrame(sequences)
results['cleaned_code'] = extract_generation_list

print(totalpass)
print(totalnum)
results.to_parquet(continue_from.replace(".parquet", "_cleaned_code.parquet"))
