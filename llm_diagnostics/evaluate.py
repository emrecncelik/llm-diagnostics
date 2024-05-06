import os
import sys
import torch
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from .datasets import collate_fn
from torch.utils.data import DataLoader
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from llm_diagnostics.datasets import ClozeDataset
from llm_diagnostics.config import DATASETS

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(os.path.basename(__file__))


class LLMDiagnosticsEvaluator:
    def __init__(
        self,
        experiment_name: str,
        data_dir: str = "datasets",
        output_dir: str = "outputs",
    ):
        self.model = None
        self.tokenizer = None
        self.dataset = None

        self.experiment_name = experiment_name
        self.data_dir = data_dir
        self.output_dir = os.path.join(output_dir, experiment_name)

    def load_model(
        self,
        model_name: str,
        quantization: bool,
    ):
        if quantization:
            logger.info("Quantization is set to True, configuring quantization.")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        logger.info(f"Loading language model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config if quantization else None,
        )

        logger.info("Loading tokenizer.")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def load_dataset(self, dataset, is_affirmative: bool, simplify_a_an: str):
        filename = os.path.join(self.data_dir, DATASETS[dataset]["filename"])

        logger.info(f"Processing dataset: {dataset} @ {filename}")

        if dataset.startswith("neg"):
            logger.info(
                f"Dataset is a negation dataset. is_affirmative: {is_affirmative}"
            )
            is_negative = int(not is_affirmative)
            self.dataset = ClozeDataset(
                filename=filename,
                tokenizer=self.tokenizer,
                context_col=DATASETS[dataset]["context_col"][is_negative],
                target_col=DATASETS[dataset]["target_col"][is_negative],
                simplify_a_an=simplify_a_an,
            )
        else:
            raise NotImplementedError("Only negation datasets are supported for now.")

        return self.dataset

    def _run_inference_no_generate(self, input_ids, attention_mask, topk):
        with torch.no_grad():
            outputs = self.model(
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

        return topk_preds

    def _run_inference_generate(self, input_ids, attention_mask, topk):
        raise NotImplementedError(
            "Evaluation using generate() method not supported yet."
        )

    def run_inference(
        self,
        topk: list[int],
        batch_size: int,
        use_generate: bool,
        progress_bar: bool,
        device: str,
    ):
        eval_dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        # Check if model is on cuda before moving it
        # Quantization loads automatically to cuda
        # so we need to check if the model is already on cuda
        # -> otherwise error
        if not next(self.model.parameters()).is_cuda:
            self.model.to(device)

        # Store every target and prediction here,
        # since dataset sizes are small not much of a problem.
        targets, preds = [], []

        if not use_generate:
            logging.info("Getting predictions without generate() method.")
            logging.info("Only outputs the first token of the prediction.")

        for batch in tqdm(
            eval_dataloader, desc="Running inference...", disable=not progress_bar
        ):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_ids = batch["target_ids"]

            if not use_generate:
                topk_preds = self._run_inference_no_generate(
                    input_ids, attention_mask, topk
                )
            else:
                topk_preds = self._run_inference_generate(
                    input_ids, attention_mask, topk
                )

            targets.extend(target_ids.numpy().tolist())
            preds.extend(topk_preds.numpy().tolist())

        return np.array(targets), np.array(preds)

    def compute_accuracy(self, targets, preds, topk: list[int]):
        accuracies = {}
        for k in topk:
            hits = (targets == preds[:, :k].T).any(
                axis=0
            )  # thanks to sklearn, TODO: needs to change for multiple targets though
            accuracy = np.sum(hits) / len(targets)
            accuracies[f"top{k}"] = accuracy
        return accuracies

    def evaluate_sensitivity(self):
        raise NotImplementedError("Sensitivity evaluation not supported yet.")

    def format_results(self, targets, preds):
        return format_results(self.dataset, self.tokenizer, targets, preds)

    def save_results(self):
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
    """
    Evaluate the accuracy of a model on a given dataset.

    Args:
        model (torch.nn.Module): The model to evaluate.
        eval_dataset (llm_diagnostics.datasets.NegationDataset): The dataset to evaluate the model on.
        device (str, optional): The device to use for evaluation (default: "cpu").
        batch_size (int, optional): The batch size for evaluation (default: 16).
        topk (list[int], optional): The list of top-k values to calculate accuracy for, calculates until the max k. (default: [1, 3, 5, 10, 20]).
        output_predictions (bool, optional): Whether to output the predictions along with accuracies (default: True).
        progress_bar (bool, optional): Whether to display a progress bar during evaluation (default: True).

    Returns:
        dict or tuple: If `output_predictions` is False, returns a dictionary of accuracies for each top-k value.
                      If `output_predictions` is True, returns a tuple containing the accuracies, all target labels,
                      and all predicted labels.
    """

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
    """
    Formats the evaluation results into a DataFrame. Takes the output
    of evaluate_accuracy and converts it into a DataFrame with the
    following columns: target_tokens, pred_tokens, correct, target, context.

    Args:
        dataset (Dataset): The dataset used for evaluation.
        tokenizer (Tokenizer): The tokenizer used to convert token IDs to tokens.
        target_ids (list): The list of target token IDs.
        pred_ids (list): The list of predicted token IDs.

    Returns:
        DataFrame: A DataFrame containing the formatted evaluation results.
            The DataFrame has the following columns:
            - target_tokens: The target tokens.
            - pred_tokens: The predicted tokens.
            - correct: A boolean indicating whether the prediction is correct.
            - target: The target values from the dataset.
            - context: The context values from the dataset.
    """
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
