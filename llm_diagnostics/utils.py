import re
import os
import sys
import logging
import pandas as pd

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(os.path.basename(__file__))


def simplify_a_an(contexts, targets, method: str):
    """
    Simplifies the determiner (a|an) in the given contexts based on the specified method.

    Args:
        contexts (list): A list of strings representing the contexts.
        targets (list): A list of strings representing the targets.
        method (str): The method to use for simplification. Possible values are:
            - "adaptive": Simplify (a|an) depending on the target.
            - Any other string: Simplify (a|an) to the specified string.

    Returns:
        list: A list of strings with the simplified determiners.

    Raises:
        None
    """

    if method:
        if method == "adaptive":  # if adaptive, simplify (a|an) depending on target
            logger.info(
                "Adaptive simplification of (a|an) in context, depending on target."
            )
            determinants = []  # store determinants for each target
            for t in targets:
                if re.match("[aeiou]", t):
                    determinants.append("an")  # if target starts with a vowel
                else:
                    determinants.append("a")  # if target starts with a consonant

            # update contexts
            logger.info(f">> # of 'a': {determinants.count('a')}")
            logger.info(f">> # of 'an': {determinants.count('an')}")
            contexts = [c.replace("(a|an)", d) for c, d in zip(contexts, determinants)]
        else:
            logger.info(f"Simplification of (a|an) to {method}.")
            contexts = [c.replace("(a|an)", method) for c in contexts]

    return contexts


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
