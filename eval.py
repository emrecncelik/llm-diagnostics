import torch
import warnings
import argparse
from llm_diagnostics.evaluate import LLMDiagnosticsEvaluator, format_results
from llm_diagnostics.config import DATASETS

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


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
        "--simplify_a_an",
        default=None,
        type=str,
        help="Simplify (a|an) in context to this value (simplified depending on target if set to 'adaptive')",
    )
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="Batch size for evaluation",
    )
    parser.add_argument("--progress_bar", action="store_true", help="Show progress bar")
    parser.add_argument(
        "--hf_cache_dir",
        default=None,
        type=str,
        help="HuggingFace cache directory to download/load models from.",
    )
    parser.add_argument(
        "--hf_token",
        default=None,
        type=str,
        help="HuggingFace token to access gated models",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    evaluator = LLMDiagnosticsEvaluator(
        "llama2_test", args.data_dir, output_dir="outputs"
    )
    evaluator.load_model(
        args.model_name,
        quantization=args.quantization,
        token=args.hf_token,
        cache_dir=args.hf_cache_dir,
    )
    evaluator.load_dataset(
        args.dataset,
        simplify_a_an=args.simplify_a_an,
        masked=True if evaluator.task_type == "maskedlm" else False,
        target_prefix=" " if "mamba" in args.model_name else "",
    )
    targets, preds, logits = evaluator.run_inference(
        topk=[1, 3, 5, 10, 20],
        batch_size=args.batch_size,
        use_generate=False,  # use_generate
        progress_bar=args.progress_bar,
        device="cuda" if torch.cuda.is_available() else "cpu",
        negative_or_reversed=False,
    )
    topk_accuracies = evaluator.compute_accuracy(
        targets=targets,
        preds=preds,
        topk=[1, 3, 5, 10, 20],
    )

    results = evaluator.format_results(targets, preds, False)
    print(f"Accuracy: {topk_accuracies}")
    print(results.head())
    print(logits)
    results.to_csv(f"results_{args.dataset}.csv", index=False)
