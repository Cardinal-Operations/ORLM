
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import os
from dataclasses import dataclass, field
from typing import Optional, List, Union
from transformers import TrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

    use_lora: Optional[bool] = field(
        default=False,
    )
    lora_rank: Optional[int] = field(
        default=16,
    )
    lora_alpha: Optional[int] = field(
        default=16,
    )
    lora_dropout: Optional[float] = field(
        default=0.05,
    )
    lora_target_modules: Optional[str] = field(
        default="[q_proj,k_proj,v_proj,o_proj]",
    )

    def __post_init__(self):
        self.lora_target_modules = self.lora_target_modules.replace(" ", "").lstrip("[").rstrip("]").split(",")

@dataclass
class DataArguments:
    train_dataset_name_or_path: str = field(
        default=None, metadata={"help": "huggingface dataset name or local data path"}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": ("The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,")
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=0,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

