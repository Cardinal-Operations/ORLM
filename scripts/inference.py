import argparse
import json
import os
import re
import sys

from vllm import LLM, SamplingParams
from tqdm import tqdm

TEMPLATE_q2mc_en = r"""
Below is an operations research question. Build a mathematical model and corresponding python code using `coptpy` that appropriately addresses the question.

# Question:
{Question}

# Response:
"""

ONE_QUESTION = r"""
A lab has 1000 units of medicinal ingredients to make two pills, a large pill and a small pill. A large pill requires 3 units of medicinal ingredients and 2 units of filler. A small pill requires 2 units of medicinal ingredients and 1 unit of filler. The lab has to make at least 100 large pills. However, since small pills are more popular at least 60% of the total number of pills must be small. How many of each should be made to minimize the total number of filler material needed?
"""

def main(args):
    assert isinstance(args.topk, int)
    assert args.decoding_method in ["greedy", "sampling"]
    assert os.path.exists(args.model_name_or_path), "We only support local model path!"

    # Load data
    prompt = TEMPLATE_q2mc_en.replace("{Question}", ONE_QUESTION.strip()).strip()
    sample = [{"prompt": prompt}]

    # Init model
    model = LLM(model=args.model_name_or_path, tensor_parallel_size=args.tensor_parallel_size)
    print("init model done.")
    stop_tokens = ["</s>"]
    if args.decoding_method == "greedy":
        sampling_params = SamplingParams(n=args.topk, temperature=0, top_p=1, max_tokens=model.llm_engine.model_config.max_model_len, stop=stop_tokens)
    elif args.decoding_method == "sampling":
        sampling_params = SamplingParams(n=args.topk, temperature=0.7, top_p=0.95, max_tokens=model.llm_engine.model_config.max_model_len, stop=stop_tokens)
    else:
        raise
    print(f"init sampling params done: {sampling_params}")

    # generate
    prompts = [example["prompt"] for example in sample]
    generations = model.generate(prompts, sampling_params)
    prompt_to_output = {
        g.prompt: g.outputs[0].text for g in generations
    }
    outputs = [prompt_to_output[prompt] if prompt in prompt_to_output else "" for prompt in prompts]

    for prompt, completion in zip(prompts, outputs):
        print("-" * 20 + "prompt" + "-" * 20)
        print(prompt)
        print("-" * 20 + "completion" + "-" * 20)
        print(completion)
        print("-" * 80)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default=None)  # model path
    parser.add_argument("--tensor_parallel_size", type=int, default=8)  # num_gpus
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--decoding_method", type=str, default="greedy")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)