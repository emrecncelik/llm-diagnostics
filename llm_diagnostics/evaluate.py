import os
import sys
import torch
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from .datasets import collate_fn
from torch.utils.data import DataLoader

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(os.path.basename(__file__))


class Evaluator:
    pass


def evaluate_accuracy(
    model,
    eval_dataset,
    device="cpu",
    batch_size=16,
    topk=[1, 3, 5, 10, 20],
    output_predictions=True,
    progress_bar=True,
):
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    if not next(model.parameters()).is_cuda:
        model.to(device)
    model.eval()

    all_targets, all_preds = [], []
    for batch in tqdm(
        eval_dataloader, desc="Running evaluation", disable=not progress_bar
    ):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target_ids = batch["target_ids"]
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        topk_preds = (
            # get top k predictions
            torch.topk(
                outputs.logits[:, -1, :], max(topk)
            ).indices.to(  # get last token from each element in batch
                "cpu"
            )  # get vocab indices, move to cpu
        )

        all_targets.extend(target_ids.numpy().tolist())
        all_preds.extend(topk_preds.numpy().tolist())

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)

    # Calculate top k accuracy
    accuracies = {}
    for k in topk:
        hits = (all_targets == all_preds[:, :k].T).any(axis=0)
        accuracy = np.sum(hits) / len(all_targets)
        accuracies[f"top{k}"] = accuracy

    if not output_predictions:
        return accuracies
    else:
        return accuracies, all_targets, all_preds


def format_results(
    dataset,
    tokenizer,
    target_ids,
    pred_ids,
):
    target_tokens, pred_tokens = tokenizer.convert_ids_to_tokens(target_ids), [
        tokenizer.convert_ids_to_tokens(ids) for ids in pred_ids
    ]
    results = pd.DataFrame(
        {
            "target_tokens": target_tokens,
            "pred_tokens": pred_tokens,
        }
    )

    correct = []
    for _, row in results.iterrows():
        correct.append(row["target_tokens"] in row["pred_tokens"])

    results["correct"] = correct
    results["target"] = dataset.targets
    results["context"] = dataset.contexts
    return results


def report_results():
    pass
