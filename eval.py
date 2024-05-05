import os
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from llm_diagnostics.evaluate import evaluate_accuracy, format_results, report_results
from llm_diagnostics.datasets import ClozeDataset
from llm_diagnostics.config import DATASETS


def get_args():
    """Set hyperparameters"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name", required=True, help="Identifier of HuggingFace model"
    )
    parser.add_argument("--quantization", action="store_true", help="Quantize model")
    parser.add_argument(
        "--data_dir", default="datasets", help="Directory containing the dataset files"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=list(DATASETS.keys()),
        help="Identifier of dataset to evaluate",
    )
    parser.add_argument(
        "--is_affirmative",
        action="store_true",
        help="Evaluate affirmative or negative context",
    )
    parser.add_argument(
        "--simplify_a_an",
        default=None,
        type=str,
        help="Simplify (a|an) in context to this value",
    )
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="Batch size for evaluation",
    )
    parser.add_argument("--progress_bar", action="store_true", help="Show progress bar")
    parser.add_argument(
        "--output_predictions", action="store_true", help="Output predictions"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    ######################################
    ############# LOAD MODEL #############
    ######################################
    model_name = args.model_name
    if args.quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config if args.quantization else None
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    ########################################
    ############# LOAD DATASET #############
    ########################################
    filename = os.path.join(args.data_dir, DATASETS[args.dataset]["filename"])

    if args.dataset.startswith("neg"):
        is_negative = int(not args.is_affirmative)
        dataset = ClozeDataset(
            filename=filename,
            tokenizer=tokenizer,
            context_col=DATASETS[args.dataset]["context_col"][is_negative],
            target_col=DATASETS[args.dataset]["target_col"][is_negative],
            simplify_a_an=args.simplify_a_an,
        )

    ########################################
    ############### EVALUATE ###############
    ########################################
    accuracy, target_ids, pred_ids = evaluate_accuracy(
        model,
        dataset,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    results = format_results(dataset, tokenizer, target_ids, pred_ids)
    print(f"Accuracy: {accuracy}")
    print(results.head())
    results.to_csv(f"results_{args.dataset}.csv", index=False)
