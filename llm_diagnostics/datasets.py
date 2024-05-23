import os
import sys
import torch
import logging
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from .utils import simplify_a_an_

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(os.path.basename(__file__))


class ClozeDataset(Dataset):
    def __init__(
        self,
        filename: str,
        tokenizer: AutoTokenizer,
        context_col: str = "context",
        target_col: str = "expected",
        masked: str = False,  # not using Config.is_decoder cuz not accurate
        max_length: int = 50,
        context_prefix: str = "",
        target_prefix: str = "",  # set to " " for mamba models
        simplify_a_an: str = None,
    ):
        # Tokenizer setup
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.warning("Setting tokenizer's pad_token to eos_token")
        tokenizer.padding_side = "left"

        # Load dataset
        if filename.endswith("tsv"):
            self.dataset = pd.read_csv(filename, sep="\t")
        else:
            self.dataset = pd.read_csv(filename)

        contexts = self.dataset[context_col].tolist()
        targets = self.dataset[target_col].tolist()

        contexts = simplify_a_an_(contexts, targets, simplify_a_an)

        # add mask token for mlms
        if masked:
            contexts = [c + f" {tokenizer.mask_token}." for c in contexts]

        contexts = [context_prefix + c.strip() for c in contexts]
        targets = [target_prefix + t.strip() for t in targets]
        self.contexts = contexts
        self.targets = targets

        contexts_tokenized = tokenizer(
            self.contexts,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        tokenizer.padding_side = "right"
        targets_tokenized = tokenizer(
            self.targets,
            max_length=2,  # assuming single token target, might change later
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        tokenizer.padding_side = "left"
        self.input_ids = contexts_tokenized["input_ids"]
        self.attention_mask = contexts_tokenized["attention_mask"]
        self.target_ids = targets_tokenized["input_ids"][
            :, 0
        ]  # get first token of target (since padding side was right)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "target_ids": self.target_ids[idx],
        }


def collate_fn(batch):
    batch_dict = {}
    for key in batch[0].keys():
        batch_dict[key] = []

    for single_dict in batch:
        for key, value in single_dict.items():
            batch_dict[key].append(value)

    for key, value in batch_dict.items():
        batch_dict[key] = torch.stack(value, dim=0)

    return batch_dict
