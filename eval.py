import os
import torch
import warnings
import argparse
from llm_diagnostics.evaluation import LLMDiagnosticsEvaluator, Metric
from llm_diagnostics.config import DATASETS
from llm_diagnostics.utils import format_results

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
        "--output_dir",
        default="outputs",
        type=str,
        help="Directory to store results.",
    )
    parser.add_argument(
        "--batch_size",
        default=4,
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

    evaluator = LLMDiagnosticsEvaluator(data_dir="datasets")
    evaluator.load_model(
        model_name=args.model_name,
        quantization=args.quantization if "bert" not in args.model_name else False,
        token=args.hf_token,
        cache_dir=args.hf_cache_dir,
    )
    model_results = {args.model_name: {}}
    for dataset in DATASETS.keys():
        model_results[args.model_name][dataset] = {}
        datasets = evaluator.load_dataset(
            dataset=dataset,
            simplify_a_an=DATASETS[dataset]["simplify_a_an"],
            masked=True if evaluator.task_type == "maskedlm" else False,
            target_prefix=" " if "mamba" in args.model_name else "",
        )

        targets, preds, probs = evaluator.run_inference(
            topk=[1, 3, 5, 10, 20],
            batch_size=args.batch_size,
            progress_bar=args.progress_bar,
            device="cuda" if torch.cuda.is_available() else "cpu",
            negative_or_reversed=False,
        )
        targets_hat, preds_hat, probs_hat = evaluator.run_inference(
            topk=[1, 3, 5, 10, 20],
            batch_size=args.batch_size,
            progress_bar=args.progress_bar,
            device="cuda" if torch.cuda.is_available() else "cpu",
            negative_or_reversed=True,
        )

        accuracy = Metric.topk_accuracy(targets, preds, [1, 3, 5, 10, 20])
        accuracy_hat = Metric.topk_accuracy(targets_hat, preds_hat, [1, 3, 5, 10, 20])
        sensitivity = Metric.sensitivity_ettinger(
            targets, targets_hat, probs, probs_hat, False, 0, True
        )
        sensitivity_hat = Metric.sensitivity_ettinger(
            targets, targets_hat, probs, probs_hat, True, 0, True
        )

        sensitivity_th = Metric.sensitivity_ettinger(
            targets, targets_hat, probs, probs_hat, False, 0.01, True
        )
        sensitivity_hat_th = Metric.sensitivity_ettinger(
            targets, targets_hat, probs, probs_hat, True, 0.01, True
        )
        negation_sensitivity = None
        if "neg" in dataset:
            negation_sensitivity = Metric.sensitivity_negation_shivagunde(
                preds, preds_hat
            )

        for k in [1, 3, 5, 10, 20]:
            model_results[args.model_name][dataset][f"accuracy_top{k}"] = accuracy[
                f"top{k}"
            ]
            model_results[args.model_name][dataset][f"accuracy_top{k}_hat"] = (
                accuracy_hat[f"top{k}"]
            )

        model_results[args.model_name][dataset]["sensitivity"] = sensitivity
        model_results[args.model_name][dataset]["sensitivity_hat"] = sensitivity_hat
        model_results[args.model_name][dataset]["sensitivity_th"] = sensitivity_th
        model_results[args.model_name][dataset][
            "sensitivity_hat_th"
        ] = sensitivity_hat_th
        model_results[args.model_name][dataset][
            "shivagunde_sensitivity"
        ] = negation_sensitivity

        result_string = ""
        result_string += f"Model: {args.model_name}\n"
        result_string += f"Dataset: {dataset}\n"
        result_string += f"Accuracy: {accuracy}\n"
        result_string += f"Accuracy (negative_or_reversed): {accuracy_hat}\n"
        result_string += f"Sensitivity: {sensitivity}\n"
        result_string += f"Sensitivity (negative_or_reversed): {sensitivity_hat}\n"
        result_string += f"Sensitivity (threshold=0.01): {sensitivity_th}\n"
        result_string += f"Sensitivity (negative_or_reversed, threshold=0.01): {sensitivity_hat_th}\n"
        if "neg" in dataset:
            result_string += f"Negation Sensitivity (Shivagunde et al., 2023): {negation_sensitivity}\n"

        outdir = os.path.join(
            args.output_dir, f"results/{args.model_name.replace('/', '_')}"
        )
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        with open(os.path.join(outdir, f"results_{dataset}.txt"), "w") as f:
            f.write(result_string)

        print(result_string)

        prediction_results = format_results(
            dataset=evaluator.datasets[0],
            tokenizer=evaluator.tokenizer,
            target_ids=targets,
            pred_ids=preds,
        )

        prediction_results_hat = format_results(
            dataset=evaluator.datasets[1],
            tokenizer=evaluator.tokenizer,
            target_ids=targets_hat,
            pred_ids=preds_hat,
        )

        print(prediction_results.head())
        print(prediction_results_hat.head())

        prediction_results.to_csv(
            os.path.join(outdir, f"predictions_{dataset}.csv"),
            index=False,
        )
        prediction_results_hat.to_csv(
            os.path.join(outdir, f"predictions_{dataset}_negative_or_reversed.csv"),
            index=False,
        )

        import json

        with open(os.path.join(outdir, f"metrics.json"), "w") as file:
            json.dump(model_results, file)

        evaluator.datasets = []
