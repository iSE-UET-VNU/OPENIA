from datasets import load_dataset
import pandas as pd
import datasets
from datasets import Dataset
import os
import textwrap
import _settings
import json

BENCHMARK_ROOT= os.path.join(_settings.BENCHMARK_FOLDER, "DevEval", "data")
STOP_WORDS = []
wrong_code = "    pass\n"
TEMPLATE = """\
Please complete the {function_name} function based on the contexts above the function.

The contexts above the function are:
```Python
{contexts_above}
```

The code to be completed is:
```Python
{input_code}
```

Completed code:
"""

def load_jsonl(fname):
    with open(fname, 'r', encoding='utf8') as f:
        lines = []
        for line in f:
            lines.append(json.loads(line))
        return lines

def _save_dataset(tokenizer,  max_seq_len, max_gen_len , instruction=False):
    save_path = f"{BENCHMARK_ROOT}/{tokenizer.name_or_path}_{max_seq_len}_{max_gen_len}_{instruction}"
    # if not os.path.exists(save_path):
    data_path = os.path.join(BENCHMARK_ROOT, "completion_dataset.jsonl")
    lines = load_jsonl(data_path)

    dataset = {}
    dataset["prompt"] = []
    dataset["task_id"] = []
    dataset["original_prompt"] = []
    dataset["canonical_solution"] = []
    dataset["stopwords"] = []
    if instruction:
        for idx, sample in enumerate(lines):
            input_ids = tokenizer.encode(sample['input_code'])
            max_context_length = max_seq_len - len(input_ids) - max_gen_len
            context_above_ids = tokenizer.encode(sample['contexts_above'])
            context_above = sample['contexts_above']
            if len(context_above_ids) > max_context_length:
                context_above_ids = context_above_ids[-max_context_length:]
                context_above = tokenizer.decode(context_above_ids)
            prompt = TEMPLATE.format(
                function_name=sample['function_name'],
                contexts_above=context_above,
                input_code=sample['input_code']
            )
            dataset["prompt"].append(prompt)
            dataset["task_id"].append(sample["namespace"])
            dataset["original_prompt"].append(sample["input_code"])
            dataset["canonical_solution"].append(sample["ground_truth"])
            dataset["stopwords"].append(STOP_WORDS)
    else:
        for idx, sample in enumerate(lines):
            input_ids = tokenizer.encode(sample['input_code'])
            max_context_length = max_seq_len - len(input_ids) - max_gen_len
            context_above_ids = tokenizer.encode(sample['contexts_above'])
            context_above = ''
            if len(context_above_ids) > max_context_length:
                context_above_ids = context_above_ids[-max_context_length:]
                context_above = tokenizer.decode(context_above_ids)
            prompt = context_above + "\n" + sample['input_code']
            dataset["prompt"].append(prompt)
            dataset["task_id"].append(sample["namespace"])
            dataset["original_prompt"].append(sample["input_code"])
            dataset["canonical_solution"].append(sample["ground_truth"])
            dataset["stopwords"].append(STOP_WORDS)
        
    data_df = pd.DataFrame.from_dict(dataset)
    dataset = Dataset.from_pandas(data_df)
    
    dataset.save_to_disk(save_path)
    return save_path

# _save_dataset(sft=False)
def get_dataset(tokenizer, language='python', sft=False, instruction=False, max_seq_len=2048, max_gen_len=500):
    dataset = datasets.load_from_disk(_save_dataset(tokenizer, max_seq_len, max_gen_len, instruction))

    def encode_humaneval(example):
        prompt = example['prompt']
        if instruction:
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": f'{prompt}'}], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, truncation=False, padding=False)
        return inputs

    dataset = dataset.map(encode_humaneval, batched=False, load_from_cache_file=False)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'], output_all_columns=True)

    return dataset

def extract_code_from_response(completion: str):
    """
    Extract code from a completion. The code is in markdown format.
    :param completion: String.
    :return: Code in the completion.
    """
    # if args.model_type == 'glm':
    #     completion_lines = completion.split("\\n")
    #     code_lines = completion.split("\\n")
    # else:
    completion_lines = completion.split("\n")
    code_lines = completion.split("\n")
    code_sol, code_eol = None, None
    for i, line in enumerate(completion_lines):
        if line.strip().startswith("```"):
            # print(code_sol, i)
            if code_sol is None:
                code_sol = i+1
            else:
                code_eol = i
                break
    # print(code_sol)
    if code_sol is None: # No markdown code block
        if code_eol is None:
            code_sol = 0
            code_eol = len(completion_lines)
        else:
            code_sol = 0
    elif code_eol is None: # No end of markdown block
        code_eol = len(completion_lines)
    code_lines = completion_lines[code_sol:code_eol]
    # print(code_sol, code_eol)
    code = "\n".join(code_lines)
    # if args.model_type == 'glm':
    #     code = code.replace('\\\"', '"')
    return code

def count_indent(code):
    if type(code) == str: # a single statement
        return len(code) - len(textwrap.dedent(code))
    elif type(code) == list: # a list of statements, i.e., a function body
        for line in code:
            if line.strip() != '':
                return len(line) - len(textwrap.dedent(line))

def extract_generation_code(example, output, lang_code: str, verbose: bool=False):
    task_id = example['task_id']
    # output = example.get('output', example.get("gpt_completion"))
    function_name = task_id.split('.')[-1]
    
    try:
        # print(output)
        # code_block: str = re.findall(f'```python\n(.*?)```', output, re.DOTALL | re.IGNORECASE)[0]
        # code_block: str = re.findall(r'```(?:Python)?\n(.*?)```', output, re.DOTALL | re.IGNORECASE)[0]
        # print(code_block)
        code = extract_code_from_response(output)
        # if not f'def {function_name}(' in code:
        #     completion_lines = code.strip('\n').split("\n")
        #     body_indent = count_indent(completion_lines[0])
        #     new_complation = []
        #     for line in completion_lines:
        #         if count_indent(line) >= body_indent or line.strip() == "":
        #             new_complation.append(line)
        #         else:
        #             break
        #     new_complation = "\n".join(new_complation)
        #     # results.append(new_complation)
        # else:
        #     function_body = find_function_body(code, function_name)
        #     if function_body is None:
        #         results.append(wrong_code)
        #     else:
        #         results.append(function_body)
        # generation = code_block
        generation = code
        # print(f"Function Prefix: {func_prefix}")
        # example['generation'] = generation

    except Exception as ex:
        print("Failed to extract code block with error `{}`:\n>>> Task: {}\n>>> Output:\n{}".format(
            ex, task_id, output
        ))
        generation = output
    # generation = generation.split('</code>')[0]
    # generation = generation.split('\nEND SOLUTION')[0]
    # generation = generation.replace('<code>', '')
    # print(f'Output: {output}')
    # print(f'Generation: {generation}')
    # print(generation)
    return generation