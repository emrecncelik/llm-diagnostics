MODELS = {
    "causallm": [
        "meta-llama/Llama-2-7b-hf",
        "meta-llama/Meta-Llama-3-8B",
        "mistralai/Mistral-7B-v0.1",
        "mistralai/Mixtral-8x7B-v0.1",
        "state-spaces/mamba-2.8b-hf",
        "state-spaces/mamba-790m-hf",
        "state-spaces/mamba-130m-hf",
        "Qwen/Qwen1.5-1.8B",
        "Qwen/Qwen1.5-7B",
    ],
    "maskedlm": [
        "bert-base-uncased",
        "bert-large-uncased",
        "distilbert-base-uncased",
        "albert/albert-base-v1",
        "albert/albert-large-v1",
    ],
}

RANDOM_MODELS = {
    "causallm": [
        "yujiepan/llama-2-tiny-random",
        "yujiepan/llama-3-tiny-random",
        "yujiepan/mistral-tiny-random",
        "yujiepan/mixtral-8xtiny-random",
        "yujiepan/mixtral-tiny-random",
        "yujiepan/mamba-tiny-random",
        "yujiepan/qwen-vl-tiny-random",
        "yujiepan/qwen1.5-tiny-random",
    ],
    "maskedlm": [
        "yujiepan/tiny-random-bert",
    ],
}


DATASETS = {
    "neg1500gen": {
        "filename": "extended/NEG-1500-SIMP-GEN.txt",
        "context_col": ["context_aff", "context_neg"],
        "target_col": ["target_aff", "target_neg"],
        "url": "https://huggingface.co/datasets/text-machine-lab/NEG-1500-SIMP-GEN/raw/main/NEG-1500-SIMP-GEN.txt",
        "type": "extended",
        "simplify_a_an": "adaptive",
    },
    "neg1500temp": {
        "filename": "extended/NEG-1500-SIMP-TEMP.txt",
        "context_col": ["context_aff", "context_neg"],
        "target_col": ["target_aff", "target_neg"],
        "url": "https://huggingface.co/datasets/text-machine-lab/NEG-1500-SIMP-TEMP/raw/main/NEG-1500-SIMP-TEMP.txt",
        "type": "extended",
        "simplify_a_an": "adaptive",
    },
    "role1500": {
        "filename": "extended/ROLE-1500.txt",
        "context_col": ["context", "context_r"],
        "target_col": ["expected", "expected_r"],
        "url": "https://huggingface.co/datasets/text-machine-lab/ROLE-1500/raw/main/ROLE-1500.txt",
        "type": "extended",
        "simplify_a_an": None,
    },
    "neg136nat": {
        "filename": "original/NEG-136-NAT.tsv",
        "context_col": ["context_aff", "context_neg"],
        "target_col": ["target_aff", "target_neg"],
        "url": "https://raw.githubusercontent.com/aetting/lm-diagnostics/master/datasets/NEG-136/NEG-136-NAT.tsv",
        "type": "original",
        "simplify_a_an": None,
    },
    "neg136simp": {
        "filename": "original/NEG-136-SIMP.tsv",
        "context_col": ["context_aff", "context_neg"],
        "target_col": ["target_aff", "target_neg"],
        "url": "https://raw.githubusercontent.com/aetting/lm-diagnostics/master/datasets/NEG-136/NEG-136-SIMP.tsv",
        "type": "original",
        "simplify_a_an": "adaptive",
    },
    "role88": {
        "filename": "original/ROLE-88.tsv",
        "context_col": ["context", "context_r"],
        "target_col": ["expected", "expected_r"],
        "url": "https://raw.githubusercontent.com/aetting/lm-diagnostics/master/datasets/ROLE-88/ROLE-88.tsv",
        "type": "original",
        "simplify_a_an": None,
    },
}
