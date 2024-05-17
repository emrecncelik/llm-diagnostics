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
    parser.add_argument(
        "--access_token",
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
        args.model_name, quantization=args.quantization, token=args.access_token
    )
    evaluator.load_dataset(
        args.dataset,
        is_affirmative=args.is_affirmative,
        simplify_a_an=args.simplify_a_an,
    )
    targets, preds = evaluator.run_inference(
        topk=[1, 3, 5, 10, 20],
        batch_size=args.batch_size,
        use_generate=False,  # use_generate
        progress_bar=args.progress_bar,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    topk_accuracies = evaluator.compute_accuracy(
        targets=targets,
        preds=preds,
        topk=[1, 3, 5, 10, 20],
    )

    results = format_results(evaluator.dataset, evaluator.tokenizer, targets, preds)
    print(f"Accuracy: {topk_accuracies}")
    print(results.head())
    results.to_csv(f"results_{args.dataset}.csv", index=False)
