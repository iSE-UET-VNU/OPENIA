import os
import re
import datasets
import pandas as pd
from datasets import Dataset
from benchmark.HumanEval.utils.dataset import HumanEvalDataset

import _settings
BENCHMARK_ROOT= os.path.join(_settings.BENCHMARK_FOLDER, "HumanEval", "data")
# LANGUAGE="python"

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

def build_deepseekcoder_instruction(languge: str, question: str):
    return '''
Please continue to complete the function. You are not allowed to modify the given code and do the completion only. Please return all completed function in a codeblock. Here is the given code to do completion:
```{}
{}
```
'''.strip().format(languge.lower(), question.strip())

def _save_dataset(language, sft=False, instruction=False):
    save_path = f"{BENCHMARK_ROOT}/{language}" if not sft else f"{BENCHMARK_ROOT}/{language}_sft"
    save_path = f"{save_path}_instruction" if instruction else save_path
    # if not os.path.exists(save_path):
    data = HumanEvalDataset(root=BENCHMARK_ROOT, language=language, issft=sft)
    dataset = {}
    dataset["prompt"] = []
    dataset["task_id"] = []
    dataset["original_prompt"] = []
    dataset["canonical_solution"] = []
    dataset["stopwords"] = []
    
    if instruction:
        # prompt = build_deepseekcoder_instruction(languge_settings[language]['full_name'], sample['prompt'])
        for sample in data:
            prompt = build_deepseekcoder_instruction(languge_settings[language]['full_name'], sample['prompt'])
            dataset["prompt"].append(prompt)
            dataset["task_id"].append(sample["task_id"])
            dataset["original_prompt"].append(sample["original_prompt"])
            dataset["canonical_solution"].append(sample["canonical_solution"])
            dataset["stopwords"].append(sample["stopwords"])
    else:
        for sample in data:
            dataset["prompt"].append(sample["prompt"])
            dataset["task_id"].append(sample["task_id"])
            dataset["original_prompt"].append(sample["original_prompt"])
            dataset["canonical_solution"].append(sample["canonical_solution"])
            dataset["stopwords"].append(sample["stopwords"])
    
    data_df = pd.DataFrame.from_dict(dataset)
    dataset = Dataset.from_pandas(data_df)
    
    dataset.save_to_disk(save_path)
    return save_path

# _save_dataset(sft=False)

def get_dataset(tokenizer, language, sft=False, instruction=False):
    dataset = datasets.load_from_disk(_save_dataset(language, sft, instruction))

    def encode_humaneval(example):
        prompt = example['prompt']
        if instruction:
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": f'{prompt}'}], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, truncation=False, padding=False)
        return inputs

    dataset = dataset.map(encode_humaneval, batched=False, load_from_cache_file=False)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'], output_all_columns=True)

    return dataset

def _generate_config(tokenizer):
    return dict()

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

def extract_generation_code(example, output, lang_code: str, verbose: bool=False):
    task_id = example['task_id']
    # output = example.get('output', example.get("gpt_completion"))
    question = example["original_prompt"].strip()
    setting = languge_settings[lang_code]
    lang = setting['full_name']
    indent = setting['indent']
    
    try:
        # print(output)
        # code_block: str = re.findall(f'```{lang.lower()}\n(.*?)```', output, re.DOTALL | re.IGNORECASE)[0]
        code_block: str = re.findall(f'```(?:{lang.lower()})?\n(.*?)```', output, re.DOTALL | re.IGNORECASE)[0]
        
        # print(code_block)

        # Remove main
        if setting.get('main', None) and setting['main'] in code_block:
            main_start = code_block.index(setting['main'])
            code_block = code_block[:main_start]
        
        func_name, func_prefix = get_function_name(question, lang)
        # print(func_name, func_prefix)

        try:
            start = code_block.lower().index(func_name.lower())
            indent = 0
            while start - indent >= 0 and code_block[start - indent-1] == ' ':
                indent += 1
            
            # try:
            #     end = code_block.rindex('\n' + ' '*indent + '}')
            # except:
            end = len(code_block)
        except:
            start = 0
            # try:
            #     end = code_block.rindex('\n' + ' '*indent + '}')
            # except:
            end = len(code_block)

        # print(len(code_block), start, end)
        body = code_block[start:end]

        # if lang_code.lower() in ['php', 'ts', 'js']:
        #     body += '\n' + ' '*indent + '}'

        # generation = func_prefix + '\n' + body + '\n'
        generation = body
        # print(f"Function Prefix: {func_prefix}")
        # example['generation'] = generation

    except Exception as ex:
        print("Failed to extract code block with error `{}`:\n>>> Task: {}\n>>> Output:\n{}".format(
            ex, task_id, output
        ))
        # example['generation'] = example['prompt'] + '\n' + output
        generation = output

    # print(f'Generation: {generation}')
    return generation
