
import os
import torch
import numpy as np
from dataclasses import dataclass
from typing import Union, List
from functools import partial

from datasets import load_dataset
from torch.utils.data import Dataset

import logging
logger = logging.getLogger(__name__)

def encode_with_prompt_completion_format(example, tokenizer, max_seq_length):
    '''
    Here we assume each example has 'prompt' and 'completion' fields.
    We concatenate prompt and completion and tokenize them together because otherwise prompt will be padded/trancated 
    and it doesn't make sense to follow directly with the completion.
    '''
    # if prompt doesn't end with space and completion doesn't start with space, add space
    if not example['prompt'].endswith((' ', '\n', '\t')) and not example['completion'].startswith((' ', '\n', '\t')):
        example_text = example['prompt'] + ' ' + example['completion']
    else:
        example_text = example['prompt'] + example['completion']
    example_text = example_text + tokenizer.eos_token
    tokenized_example = tokenizer(example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()
    tokenized_prompt = tokenizer(example['prompt'], return_tensors='pt', max_length=max_seq_length, truncation=True)
    # mask the prompt part for avoiding loss
    labels[:, :tokenized_prompt.input_ids.shape[1]] = -100
    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }


def encode_with_messages_format(example, tokenizer, max_seq_length):
    '''
    Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    '''
    messages = example['messages']
    if len(messages) == 0:
        raise ValueError('messages field is empty.')
    
    def _concat_messages(messages):
        message_text = ""
        for message in messages:
            if message["role"] == "system":
                message_text += "<|system|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "user":
                message_text += "<|user|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "assistant":
                message_text += "<|assistant|>\n" + message["content"].strip() + tokenizer.eos_token + "\n"
            else:
                raise ValueError("Invalid role: {}".format(message["role"]))
        return message_text
        
    example_text = _concat_messages(messages).strip()
    tokenized_example = tokenizer(example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    _concat_messages(messages[:message_idx]), return_tensors='pt', max_length=max_seq_length, truncation=True
                ).input_ids.shape[1]
            if message_idx < len(messages) - 1 and messages[message_idx+1]["role"] == "assistant":
                # here we also ignore the role of the assistant
                messages_so_far = _concat_messages(messages[:message_idx+1]) + "<|assistant|>\n"
            else:
                messages_so_far = _concat_messages(messages[:message_idx+1])
            message_end_idx = tokenizer(
                messages_so_far,
                return_tensors='pt', 
                max_length=max_seq_length, 
                truncation=True
            ).input_ids.shape[1]
            labels[:, message_start_idx:message_end_idx] = -100
            
            if message_end_idx >= max_seq_length:
                break

    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }

class CustomDataset(Dataset):
    def __init__(
            self,
            training_args, 
            data_args, 
            model_args, 
            tokenizer, 
    ):
        self.training_args = training_args
        self.data_args = data_args
        self.model_args = model_args
        self.tokenizer = tokenizer

        if os.path.exists(self.data_args.train_dataset_name_or_path):
            if os.path.isdir(self.data_args.train_dataset_name_or_path):
                dir_ents = os.listdir(self.data_args.train_dataset_name_or_path)
                data_files = [os.path.join(self.data_args.train_dataset_name_or_path, de) for de in dir_ents]
            else:
                data_files = [self.data_args.train_dataset_name_or_path]
            one_file = data_files[0]
            if one_file.endswith("json") or one_file.endswith("jsonl"):
                data_type = "json"
            elif one_file.endswith("csv"):
                data_type = "csv"
            else:
                raise Exception("Unsupported data type.")
            raw_datasets = load_dataset(
                data_type,
                data_files=data_files, 
                cache_dir=self.model_args.cache_dir
            )
        else:
            raw_datasets = load_dataset(
                self.data_args.train_dataset_name_or_path,
                cache_dir=self.model_args.cache_dir
            )

        if self.data_args.preprocessing_num_workers > 0:
            # Preprocessing the datasets.
            if "prompt" in raw_datasets["train"].column_names and "completion" in raw_datasets["train"].column_names:
                encode_function = partial(
                    encode_with_prompt_completion_format,
                    tokenizer=self.tokenizer,
                    max_seq_length=self.data_args.max_seq_length,
                )
            elif "messages" in raw_datasets["train"].column_names:
                encode_function = partial(
                    encode_with_messages_format,
                    tokenizer=self.tokenizer,
                    max_seq_length=self.data_args.max_seq_length,
                )
            else:
                raise ValueError("You need to have either 'prompt'&'completion' or 'messages' in your column names.")

            with self.training_args.main_process_first(desc="Processing instruction data"):
                lm_datasets = raw_datasets.map(
                    encode_function,
                    batched=False,
                    num_proc=self.data_args.preprocessing_num_workers,
                    load_from_cache_file=not self.data_args.overwrite_cache,
                    desc="Tokenizing and reformatting instruction data",
                )
                lm_datasets.set_format(type="pt")
        else:
            lm_datasets = raw_datasets

        self.train_dataset = lm_datasets["train"]
        if self.data_args.max_train_samples is not None:
            max_samples = min(len(self.train_dataset), self.data_args.max_train_samples)
            self.train_dataset = self.train_dataset.select(range(max_samples))

    def __len__(self):
        return len(self.train_dataset)

    def __getitem__(self, item):
        example = self.train_dataset[item]
        if self.data_args.preprocessing_num_workers > 0:
            return example
        else:
            if "prompt" in example and "completion" in example:
                example = encode_with_prompt_completion_format(example, self.tokenizer, self.data_args.max_seq_length)
            elif "messages" in example:
                example = encode_with_messages_format(example, self.tokenizer, self.data_args.max_seq_length)
            else:
                raise Exception("You need to have either 'prompt'&'completion' or 'messages' in your column names")
            return example

