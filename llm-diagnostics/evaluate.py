import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader


def evaluate_accuracy(
    model,
    eval_dataset,
    device="cpu",
    batch_size=16,
    topk=1,
    output_predictions=True,
    progress_bar=True,
):
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuflle=False,
    )

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
                outputs.logits[:, -1, :], topk
            )  # get last token from each element in batch
            .indices.to("cpu")  # get vocab indices, move to cpu
            .numpy()
        )

        all_targets.extend(target_ids.numpy())
        all_preds.extend(topk_preds)

    # Calculate top k accuracy
    hits = (all_targets == all_preds[:, :topk].T).any(axis=0)
    accuracy = np.sum(hits) / len(all_targets)

    if not output_predictions:
        return accuracy
    else:
        return accuracy, all_targets, all_preds
