import os
import re
import datasets
import pandas as pd
from datasets import Dataset
import torch

from benchmark.HumanEval.utils.dataset import HumanEvalDataset
from benchmark.MBPP.utils.dataset import MBPPDataset

from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
import json
import _settings
BENCHMARK_ROOT= os.path.join(_settings.BENCHMARK_FOLDER, "MBPP", "data")
DATASET_ROOT = os.path.join(_settings.DATA_FOLDER, "MBPP")

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords_str, tokenizer):
        StoppingCriteria.__init__(self)
        self.current_context = []
        self.tokenizer = tokenizer
        self.keywords_str = keywords_str
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        self.current_context.append(input_ids[0][-1].item())
        current_context = self.tokenizer.decode(self.current_context)
        for word in self.keywords_str:
            if word in current_context:
                return True
        return False

def read_test_examples(data_path: str):
    def format_test_example(q, tests, code: str=None):
        prompt = ">>> Problem:\n{}\n>>> Test Cases:\n{}\n".format(q.strip(), "\n".join(tests))
        if code:
            code = code.replace("\r", "").replace("\t", "    ")
            prompt += "\n>>> Code:\n```python\n{}\n```".format(code)
        return prompt

    examples = [json.loads(x) for x in open(data_path)]
    print("Read all {} examples from {} over!".format(len(examples), data_path))

    # test_cases
    examples_str = []
    for i in range(1, 4):
        ex = examples[i]
        q, test, code = ex['text'], ex['test_list'], ex['code']
        ex_prompt = format_test_example(q, test, code)
        example_prompt = '- Example {}:\n{}'.format(i, ex_prompt)
        examples_str += [example_prompt]

    for i in range(10, 510):
        ex = examples[i]
        q, test, code = ex['text'], ex['test_list'], ex['code']
        
        prompt = format_test_example(q, test, code=None)

        prompt_with_shots = '''
Please refer the given examples and generate a python function for my problem.\n
Examples are listed as follows:
{}

Here is my problem:
{}
'''.strip().format('\n\n'.join(examples_str), prompt)
        yield {
            'task_id': ex['task_id'],
            'prompt': prompt_with_shots,
            'original_prompt': ex['text'],
            'canonical_solution': ex['code']
        }

def _save_dataset(tokenizer, max_seq_len, max_gen_len, instruction=False):
    save_path = f"{DATASET_ROOT}/{tokenizer.name_or_path}_{max_seq_len}_{max_gen_len}_{instruction}"
    
    # if not os.path.exists(save_path):
    mpbb_data = MBPPDataset(root=BENCHMARK_ROOT)
    dataset = {}
    dataset["prompt"] = []
    dataset["task_id"] = []
    dataset["canonical_solution"] = []
    dataset["prompt_length"] = []
    dataset["original_prompt"] = []
    
    if instruction:
        problem_file = os.path.join(BENCHMARK_ROOT, f"mbpp.jsonl")
        examples = list(read_test_examples(problem_file))
        for ex in examples:
            dataset["prompt"].append(ex["prompt"])
            dataset["task_id"].append(ex["task_id"])
            dataset["canonical_solution"].append(ex["canonical_solution"])
            dataset["prompt_length"].append(len(ex["prompt"]))
            dataset["original_prompt"].append(ex["original_prompt"])
    else:
        for j in range(len(mpbb_data)):
            data = mpbb_data[j]
            prompt = mpbb_data.prompt
            prompt1 = data["prompt"]
            tests = "\n".join(data["test"])
            # test_list.append(data["test"])
            prompt_curr = f"You are an expert Python programmer, and here is your task: {prompt1} Your code should pass these tests:\n\n{tests}\n[BEGIN]"
            fprompt = ""
            for i in range(len(prompt) - 1, -1, -1):
                finalprompt = prompt[i] + prompt_curr
                curr_seq_len = len(tokenizer.encode(finalprompt))
                if curr_seq_len >= max_seq_len - max_gen_len:
                    continue
                else:
                    fprompt = finalprompt
                    break
            if fprompt == "":
                fprompt = prompt_curr
                encodelist = tokenizer.encode(fprompt)
                while True:
                    try:
                        fprompt = tokenizer.decode(encodelist[:max_seq_len - max_gen_len])
                        break
                    except:
                        encodelist.pop(-1)
            dataset["prompt"].append(fprompt)
            dataset["canonical_solution"].append(data['code'])
            dataset["prompt_length"].append(len(fprompt))
            dataset["task_id"].append(data["task_id"])
            dataset["original_prompt"].append(data["prompt"])
        
    data_df = pd.DataFrame.from_dict(dataset)
    dataset = Dataset.from_pandas(data_df)
    
    dataset.save_to_disk(save_path)
    return save_path

# _save_dataset(sft=False)

def get_dataset(tokenizer, language='python', instruction=False, max_seq_len=2048, max_gen_len=512):
    dataset = datasets.load_from_disk(_save_dataset(tokenizer, max_seq_len, max_gen_len, instruction))
    
    def encode_mbpp(example):
        prompt = example['prompt']
        if instruction:
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": f'{prompt}'}], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, truncation=False, padding=False)
        return inputs
        # return tokenized_prompt

    dataset = dataset.map(encode_mbpp, batched=False, load_from_cache_file=False)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'], output_all_columns=True)

    return dataset

def _generate_config(tokenizer):
    stop_criteria = KeywordsStoppingCriteria(["[DONE]"], tokenizer)
    return dict(stopping_criteria=StoppingCriteriaList([stop_criteria]))

def _stop_word_list(language):
    return "[DONE]"

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


def extract_generation_code(example, output, lang_code: str, verbose: bool=False):
    task_id = example['task_id']
    # output = example.get('output', example.get("gpt_completion"))
    
    try:
        # print(output)
        # code_block: str = re.findall(f'```python\n(.*?)```', output, re.DOTALL | re.IGNORECASE)[0]
        code_block: str = re.findall(r'```(?:python)?\n(.*?)```', output, re.DOTALL | re.IGNORECASE)[0]
        # print(code_block)
        generation = code_block
        # print(f"Function Prefix: {func_prefix}")
        # example['generation'] = generation

    except Exception as ex:
        print("Failed to extract code block with error `{}`:\n>>> Task: {}\n>>> Output:\n{}".format(
            ex, task_id, output
        ))
        # example['generation'] = example['prompt'] + '\n' + output
        generation = output

    # print(f'Generation: {generation}')
    # print(generation)
    return generation