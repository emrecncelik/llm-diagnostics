import os
import sys
import torch
import logging
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BitsAndBytesConfig
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from .datasets import ClozeDataset, collate_fn
from .utils import format_results, identify_task_type
from .config import DATASETS

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(os.path.basename(__file__))


class LLMDiagnosticsEvaluator:
    # TODO: Add support for multiple targets (role)
    # TODO: Implement sensitivity evaluation
    # TODO: Implements save results
    # TODO: Implement inference with generate() method
    # TODO: Show types for function arguments and return values
    def __init__(
        self,
        experiment_name: str,
        data_dir: str = "datasets",
        output_dir: str = "outputs",
    ):
        self.model: PreTrainedModel = None
        self.tokenizer: PreTrainedTokenizer = None
        self.datasets: list[ClozeDataset] = []

        self.model_name: str = None
        self.task_type: str = None
        self.experiment_name: str = experiment_name
        self.data_dir: str = data_dir
        self.output_dir: str = os.path.join(output_dir, experiment_name)

    def load_model(
        self,
        model_name: str,
        quantization: bool,
        token: str,
        cache_dir: str,
    ):
        self.model_name = model_name
        self.task_type = identify_task_type(model_name)
        logger.info(f"Identified task type from model name: {self.task_type}")

        if quantization:
            logger.info("Quantization is set to True, configuring quantization.")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        logger.info(f"Loading language model: {model_name}")
        if "maskedlm" == self.task_type:
            self.model = AutoModelForMaskedLM.from_pretrained(
                model_name,
                quantization_config=bnb_config if quantization else None,
                token=token,
                cache_dir=cache_dir,
                trust_remote_code=True,
            )
        elif "causallm" == identify_task_type(model_name):
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config if quantization else None,
                token=token,
                cache_dir=cache_dir,
                trust_remote_code=True,
            )
        else:
            raise ValueError(f"Model type not found for model: {model_name}")

        logger.info("Loading tokenizer.")
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=token,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )

    def load_dataset(self, dataset: str, **dataset_kwargs) -> list[ClozeDataset]:
        filename = os.path.join(self.data_dir, DATASETS[dataset]["filename"])
        logger.info(f"Creating dataset instances for: {dataset} @ {filename}")
        for i in [0, 1]:
            logger.info(f"Loading dataset for {DATASETS[dataset]['context_col'][i]}")
            self.datasets.append(
                ClozeDataset(
                    filename=filename,
                    tokenizer=self.tokenizer,
                    context_col=DATASETS[dataset]["context_col"][i],
                    target_col=DATASETS[dataset]["target_col"][i],
                    **dataset_kwargs,
                )
            )
        return self.datasets

    def _run_inference(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, topk: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        if self.task_type == "maskedlm":
            mask_indices = (input_ids == self.tokenizer.mask_token_id).nonzero()[:, -1]
            logits = outputs.logits.gather(
                1, mask_indices.view(-1, 1, 1).expand(-1, 1, outputs.logits.size()[-1])
            ).squeeze(1)
        elif self.task_type == "causallm":
            logits = outputs.logits[:, -1, :]

        topk_preds = (
            # get top k predictions
            torch.topk(
                logits,
                max(topk),  # get last token from each element in batch
            ).indices  # get vocab indices, move to cpu
        )

        return topk_preds.detach().cpu(), logits.detach().cpu()

    def run_inference(
        self,
        topk: list[int],
        batch_size: int,
        progress_bar: bool,
        device: str,
        negative_or_reversed: bool = False,
    ):
        eval_dataloader = DataLoader(
            self.datasets[int(negative_or_reversed)],
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
        targets, preds, logits = [], [], []

        for batch in tqdm(
            eval_dataloader, desc="Running inference...", disable=not progress_bar
        ):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_ids = batch["target_ids"]

            topk_preds, batch_logits = self._run_inference(
                input_ids, attention_mask, topk
            )

            targets.extend(target_ids.numpy().tolist())
            preds.extend(topk_preds.numpy().tolist())
            logits.extend(batch_logits.numpy().tolist())

        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()

        return np.array(targets), np.array(preds), probs

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

    def format_results(self, targets, preds, negative_or_reversed):
        return format_results(
            self.datasets[int(negative_or_reversed)], self.tokenizer, targets, preds
        )


class Metric:
    @staticmethod
    def topk_accuracy(targets, preds, topk: list[int]):
        accuracies = {}
        for k in topk:
            hits = (targets == preds[:, :k].T).any(
                axis=0
            )  # thanks to sklearn, TODO: needs to change for multiple targets though
            accuracy = np.sum(hits) / len(targets)
            accuracies[f"top{k}"] = accuracy
        return accuracies

    def _get_probabilities_for_indices(probs, targets):
        return np.take_along_axis(probs, targets[:, np.newaxis], axis=1).squeeze()

    @staticmethod
    def sensitivity_negation_ettinger(
        targets_aff: np.ndarray,
        targets_neg: np.ndarray,
        probs_aff: np.ndarray,
        probs_neg: np.ndarray,
        reverse: bool = False,
    ) -> float:
        """
        Calculate the sensitivity to negation using the metric definition
        from Ettinger (2019). Definition follows as,
        "Measuring the proportion of items in which the model
        assigns higher probabilities to true completions than to false ones."

        Args:
            targets_aff (np.ndarray): Array of target indices for affirmative form with shape (N,).
            targets_neg (np.ndarray): Array of target indices for negative form with shape (N,).
            probs_aff (np.ndarray): Array of predicted probabilities for affirmative form with shape (N, Vocab).
            probs_neg (np.ndarray): Array of predicted probabilities for negative form with shape (N, Vocab).
            reverse (bool, optional): If True, calculate sensitivity in reverse direction.
                                      Defaults to False.

        Returns:
            float: Sensitivity to negation using the Ettinger method.
        """
        aff = Metric._get_probabilities_for_indices(probs_aff, targets_aff)
        neg = Metric._get_probabilities_for_indices(probs_neg, targets_neg)
        return (
            (aff > neg).sum() / len(targets_aff)
            if not reverse
            else (neg > aff).sum() / len(targets_aff)
        )

    @staticmethod
    def sensitivity_negation_shivagunde(
        preds_aff: np.ndarray, preds_neg: np.ndarray
    ) -> float:
        """
        Calculate the sensitivity to negation the metric definition
        from Shivagunde et al. (2023). Definition follows as,
        "Percentage of sentence pairs for which the top-1 prediction changed"

        Args:
            preds_aff (numpy.ndarray): Predictions for the affirmative form.
            preds_neg (numpy.ndarray): Predictions for the negation form.

        Returns:
            float: Shivagunde sensitivity to negation.
        """
        return (preds_aff[:, 0] != preds_neg[:, 0]).sum() / len(preds_aff)

    @staticmethod
    def sensitivity_role_ettinger(targets, targets_reversed, logits, logits_reversed):
        pass
