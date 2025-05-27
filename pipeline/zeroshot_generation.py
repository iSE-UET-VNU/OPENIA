import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import pandas as pd

from sklearn.metrics import (
    classification_report,
    recall_score,
    precision_score,
    f1_score,
    accuracy_score,
)
import dataeval.w_mbpp as mbpp
import dataeval.w_repoeval as repo_eval
import argparse
import dataeval.w_humaneval as human_eval
import dataeval.w_mbpp as mbpp
import dataeval.w_ds1000 as ds1000
import dataeval.w_repoeval as repo_eval
import random
from torch.utils.data import DataLoader


random.seed(42)


def get_dataset_fn(data_name):
    if data_name == "human_eval":
        return human_eval.get_dataset
    if data_name == "mbpp":
        return mbpp.get_dataset
    if data_name == "ds1000":
        return ds1000.get_dataset
    if data_name == "repo_eval":
        return repo_eval.get_dataset


def build_prompt(Query):
    return f"Are you capable of providing an accurate response to the query given below? Respond only to this question with ’yes’ or ’no’ and do not address the content of the query itself. The query in block [Query] and [/Query] and your respone after 'Answer'. \n[Query]\n{Query}\n[/Query] \n\nAre you capable of providing an accurate response to the query given above without more information? Respond only to this question with yes or no. \nAnswer: "


def build_prompt_with_output(Query, Respone):
    return f"The query in block [Query] and [/Query] and your respone in block [Respone] and [/Respone]. \n[Query]\n{Query}\n[/Query] \n[Respone]\n{Respone}\n[/Respone]\n\nIs your respone is accurate to query? Answer only to this question with yes or no. \nAnswer: "


def get_model_tokenize(model_name):
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name, resume_download=True, trust_remote_code=True
    # )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.bfloat16
    )
    # model.to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # Need to set the padding token to the eos token for generation
    if tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    tokenizer.padding_side = "left"
    return model, tokenizer


def main(args):

    # Generate unique numbers within the range

    df = pd.read_parquet(
        args.source_file
        # "/drive2/tuandung/WCODELLM/LFCLF_embedding_ds1000_deepseek-ai_deepseek-coder-1.3b-instruct_24_label_cleaned_code.parquet"
    )

    task_id_list = df["task_id"].unique().tolist()

    sample_size = int(1 * len(task_id_list))
    # Generate unique numbers within the range
    train_task_ids = random.sample(task_id_list, sample_size)

    df_test = df[df["task_id"].isin(train_task_ids)]
    # df_test = df
    df_test_dict = df_test.to_dict(orient="records")

    # problem_file = "/drive2/tuandung/WCODELLM/benchmark/DS_1000/data/ds1000.jsonl"
    examples_tmp = [json.loads(x) for x in open(args.problem_file) if x.strip()]
    examples = dict()
    for ex in examples_tmp:
        if hasattr(ex, "name"):
            examples[ex["name"]] = ex
        else:
            examples[ex["task_id"]] = ex
    # print(examples[0])
    # Apply padding on the left since we are doing generation

    for test_dict in df_test_dict:
        test_dict["prompt"] = examples[test_dict["task_id"]]["prompt"]

    baseline_prompts = []
    true_labels = [test_dict["label"] for test_dict in df_test_dict]
    for idx, test_dict in enumerate(df_test_dict):
        # print(test_dict)
        prompt = test_dict["prompt"]
        respone = test_dict["cleaned_code"]
        if args.type == "output":
            baseline_prompts.append(
                {
                    "prompt": build_prompt_with_output(prompt, respone),
                    "label": true_labels[idx],
                }
            )
        else:
            baseline_prompts.append(
                {
                    "prompt": build_prompt(prompt),
                    "label": true_labels[idx],
                }
            )
    with open(args.prompt_file, "w") as f:
        json.dump(baseline_prompts, f)


def eval_prompt(args):
    # done_prompt
    model, tokenizer = get_model_tokenize(args.model)
    baseline_prompts = []
    with open(args.prompt_file) as f:
        baseline_prompts = json.load(f)

    model_name = args.model.replace("/", "-")
    generation_config = {"do_sample": False, "max_new_tokens": 32, "num_beams": 1}

    generated_texts = []

    batch_size = 4

    # Convert prompts into a DataLoader for batching
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": item["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for item in baseline_prompts
    ]
    # print(prompts[0])
    data_loader = DataLoader(prompts, batch_size=batch_size, shuffle=False)

    for batch in tqdm(data_loader, desc="Generating in batches"):
        inputs = tokenizer(batch, return_tensors="pt", padding="longest").to("cuda")
        # print(inputs)
        generation_config["pad_token_id"] = tokenizer.eos_token_id
        generated_ids = model.generate(**inputs, **generation_config)

        output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        for i_, source in enumerate(batch):
            res = output[i_]
            generated_texts.append(res)

    true_labels = [test_dict["label"] for test_dict in baseline_prompts]

    predictions = []
    for text in generated_texts:
        if "yes" in text or "Yes" in text:
            predictions.append(1)
        else:
            predictions.append(0)

    print(sum(predictions))
    accuracy = accuracy_score(true_labels, predictions)
    print(classification_report(true_labels, predictions))
    tmp_r = recall_score(true_labels, predictions, average="weighted")
    tmp_p = precision_score(true_labels, predictions, average="weighted")
    tmp_f = f1_score(true_labels, predictions, average="weighted")
    tmp_a = accuracy_score(true_labels, predictions)
    print(f"{tmp_a}\t{tmp_p}\t{tmp_r}\t{tmp_f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/deepseek-coder-1.3b-instruct",
        help="which results to run",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="where to resume inference",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="gen_prompt",
        help="gen_prompt|evaluate",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="wo_output",
        help="wo_output|output",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        help="prompt_file",
    )  # prompt_file
    parser.add_argument(
        "--problem_file",
        type=str,
        help="problem_file",
    )
    parser.add_argument(
        "--source_file",
        type=str,
        help="source_file.paquet",
    )
    args = parser.parse_args()
    if args.mode == "gen_prompt":
        main(args)
    else:
        eval_prompt(args)
