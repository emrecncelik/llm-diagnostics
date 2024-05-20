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
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=token,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )

    def load_dataset(self, dataset, simplify_a_an: str):
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
                    simplify_a_an=simplify_a_an,
                    target_prefix=(
                        " " if "mamba" in self.model_name else ""
                    ),  # for mamba models, gotta find a better solution
                )
            )
        return self.datasets

    def _run_inference_no_generate(self, input_ids, attention_mask, topk):
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        logits = outputs.logits[:, -1, :]
        topk_preds = (
            # get top k predictions
            torch.topk(
                logits,
                max(topk),  # get last token from each element in batch
            ).indices  # get vocab indices, move to cpu
        )

        return topk_preds.detach().cpu(), logits.detach().cpu()

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
                topk_preds, batch_logits = self._run_inference_no_generate(
                    input_ids, attention_mask, topk
                )
            else:
                topk_preds, batch_logits = self._run_inference_generate(
                    input_ids, attention_mask, topk
                )

            targets.extend(target_ids.numpy().tolist())
            preds.extend(topk_preds.numpy().tolist())
            logits.extend(batch_logits.numpy().tolist())

        return np.array(targets), np.array(preds), np.array(logits)

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

    def save_results(self):
        pass
