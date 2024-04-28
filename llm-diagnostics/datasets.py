import pandas as pd
from torch.utils.data import Dataset, DataLoader


class NegationDataset(Dataset):
    def __init__(
        self,
        filename,
        prompt_template="""
            The goal is to complete the given sentence with one English word.
            DO NOT generate punctuation or new line characters or underlines.
            Sentence: {context}
            Answer:
            """,
    ):

        if filename.endswith("tsv"):
            self.dataset = pd.read_csv(filename, sep="\t")
        else:
            self.dataset = pd.read_csv(filename)
        self.contexts = (
            self.dataset["context_aff"].tolist() + self.dataset["context_neg"].tolist()
        )
        self.targets = (
            self.dataset["target_aff"].tolist() + self.dataset["target_neg"].tolist()
        )
        self.prompt_template = prompt_template

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        return (
            self.prompt_template.format(context=self.contexts[idx]),
            self.targets[idx],
        )
