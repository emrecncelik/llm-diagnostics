import io
import os
import requests
import argparse
import pandas as pd

DATA_DIR = "datasets"
DOWNLOAD = True

parser = argparse.ArgumentParser(description="Download datasets script")
parser.add_argument(
    "--data_dir",
    type=str,
    default=DATA_DIR,
    help="The directory to save the downloaded datasets",
)
parser.add_argument(
    "--download",
    action="store_true",
    help="Flag to indicate whether to download the datasets",
)


ORIGINAL = [
    "https://raw.githubusercontent.com/aetting/lm-diagnostics/master/datasets/ROLE-88/ROLE-88.tsv",
    "https://raw.githubusercontent.com/aetting/lm-diagnostics/master/datasets/NEG-136/NEG-136-NAT.tsv",
    "https://raw.githubusercontent.com/aetting/lm-diagnostics/master/datasets/NEG-136/NEG-136-SIMP.tsv",
    "https://raw.githubusercontent.com/aetting/lm-diagnostics/master/datasets/CPRAG-102/CPRAG-102.tsv",
    "https://raw.githubusercontent.com/aetting/lm-diagnostics/master/datasets/CPRAG-102/CPRAG-explanations.tsv",
]

EXTENDED = [
    "https://huggingface.co/datasets/text-machine-lab/NEG-1500-SIMP-TEMP/raw/main/NEG-1500-SIMP-TEMP.txt",
    "https://huggingface.co/datasets/text-machine-lab/NEG-1500-SIMP-GEN/raw/main/NEG-1500-SIMP-GEN.txt",
    "https://huggingface.co/datasets/text-machine-lab/ROLE-1500/raw/main/ROLE-1500.txt",
]


def download_dataset(
    dataset_url: str, format: str = "tsv", data_dir: str = DATA_DIR
) -> None:
    """
    Download a dataset from a given URL and save it to the specified directory.

    Args:
        dataset_url (str): The URL of the dataset to download.
        format (str, optional): The format of the dataset. Defaults to "tsv".
        data_dir (str, optional): The directory to save the downloaded dataset. Defaults to DATA_DIR.

    Returns:
        None
    """
    dataset_name = dataset_url.split("/")[-1]
    print(f"Downloading {dataset_name}...")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    dataset_file = requests.get(dataset_url).content

    if format == "tsv":
        dataset = pd.read_csv(io.StringIO(dataset_file.decode("utf-8")), sep="\t")
        dataset.to_csv(f"{data_dir}/{dataset_name}", sep="\t", index=False)
    elif format == "txt":
        dataset = pd.read_csv(io.StringIO(dataset_file.decode("utf-8")), header=None)
        dataset.to_csv(f"{data_dir}/{dataset_name}", header=None, index=False)
    print(f"Downloaded {dataset_name} to {data_dir}/{dataset_name}")


def format_negation(file_dir: str) -> None:
    """
    Formats the negation dataset to original data format
    by splitting the context into affirmative and negative parts,
    replacing the second-to-last word with "(a|an)", and extracting
    the last word as the target.

    Args:
        file_dir (str): The file directory of the dataset.

    Returns:
        None
    """
    dataset = pd.read_csv(file_dir, header=None)
    positive = dataset[0][[i for i in range(0, 1500, 2)]]
    negative = dataset[0][[i for i in range(1, 1500, 2)]]

    dataset["context_aff"] = positive.reset_index(drop=True)
    dataset["context_neg"] = negative.reset_index(drop=True)
    dataset = dataset.drop(columns=[0, 1])
    dataset = dataset.dropna()
    dataset["target_aff"] = ""
    dataset["target_neg"] = ""

    for i, row in dataset.iterrows():
        aff_split = row["context_aff"].split()
        neg_split = row["context_neg"].split()
        aff_last_word = aff_split[-1]
        neg_last_word = neg_split[-1]
        aff_split[-2] = "(a|an)"
        neg_split[-2] = "(a|an)"
        context_aff = " ".join(aff_split[:-1])
        context_neg = " ".join(neg_split[:-1])

        dataset.loc[i, "context_aff"] = context_aff
        dataset.loc[i, "context_neg"] = context_neg
        dataset.loc[i, "target_aff"] = aff_last_word
        dataset.loc[i, "target_neg"] = neg_last_word

    dataset.to_csv(file_dir, index=False)


def format_role(file_dir) -> None:
    """
    Formats the role dataset by splitting the first column into 'context' and 'expected' columns.

    Args:
        file_dir (str): The file directory of the dataset.

    Returns:
        None
    """
    dataset = pd.read_csv(file_dir, header=None)
    dataset["context"] = ""
    dataset["expected"] = ""
    for i, row in dataset.iterrows():
        context_split = row[0].split()
        context = " ".join(context_split[:-1])
        expected = context_split[-1]

        dataset.loc[i, "context"] = context
        dataset.loc[i, "expected"] = expected
    dataset = dataset.drop(columns=[0, 1])
    dataset = dataset.dropna()
    dataset.to_csv(file_dir, index=False)


if __name__ == "__main__":
    args = parser.parse_args()

    if args.download:
        original_dir = os.path.join(DATA_DIR, "original")
        extended_dir = os.path.join(DATA_DIR, "extended")
        for dataset in ORIGINAL:
            if not os.path.exists(original_dir):
                os.makedirs(original_dir)
            download_dataset(dataset, format="tsv", data_dir=original_dir)
        for dataset in EXTENDED:
            if not os.path.exists(extended_dir):
                os.makedirs(extended_dir)
            download_dataset(dataset, format="txt", data_dir=extended_dir)

    for dataset in os.listdir(os.path.join(DATA_DIR, "extended")):
        if dataset.startswith("NEG"):
            print(f"Formatting {dataset}...")
            format_negation(os.path.join(DATA_DIR, f"extended/{dataset}"))

    for dataset in os.listdir(os.path.join(DATA_DIR, "extended")):
        if dataset.startswith("ROLE"):
            print(f"Formatting {dataset}...")
            format_role(os.path.join(DATA_DIR, f"extended/{dataset}"))
