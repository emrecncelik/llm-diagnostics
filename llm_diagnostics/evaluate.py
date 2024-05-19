import os
import sys
import torch
import logging
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from .datasets import ClozeDataset, collate_fn
from .utils import format_results
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
        token: str,
        cache_dir: str,
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
            token=token,
            cache_dir=cache_dir,
        )

        logger.info("Loading tokenizer.")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, token=token, cache_dir=cache_dir
        )

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
