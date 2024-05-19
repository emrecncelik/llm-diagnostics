RANDOM_MODELS = [
    "yujiepan/tiny-random-bert",
    "yujiepan/llama-2-tiny-random",
    "yujiepan/llama-3-tiny-random",
    "yujiepan/mistral-tiny-random",
    "yujiepan/mixtral-8xtiny-random",
    "yujiepan/mixtral-tiny-random",
    "yujiepan/mamba-tiny-random",
    "yujiepan/qwen-vl-tiny-random",
    "yujiepan/qwen1.5-tiny-random",
]

DATASETS = {
    "neg1500gen": {
        "filename": "extended/NEG-1500-SIMP-GEN.txt",
        "context_col": ["context_aff", "context_neg"],
        "target_col": ["target_aff", "target_neg"],
        "url": "https://huggingface.co/datasets/text-machine-lab/NEG-1500-SIMP-GEN/raw/main/NEG-1500-SIMP-GEN.txt",
        "type": "extended",
    },
    "neg1500temp": {
        "filename": "extended/NEG-1500-SIMP-TEMP.txt",
        "context_col": ["context_aff", "context_neg"],
        "target_col": ["target_aff", "target_neg"],
        "url": "https://huggingface.co/datasets/text-machine-lab/NEG-1500-SIMP-TEMP/raw/main/NEG-1500-SIMP-TEMP.txt",
        "type": "extended",
    },
    "role1500": {
        "filename": "extended/ROLE-1500.txt",
        "context_col": ["context", "context_r"],
        "target_col": ["expected", "expected_r"],
        "url": "https://huggingface.co/datasets/text-machine-lab/ROLE-1500/raw/main/ROLE-1500.txt",
        "type": "extended",
    },
    "neg136nat": {
        "filename": "original/NEG-136-NAT.tsv",
        "context_col": ["context_aff", "context_neg"],
        "target_col": ["target_aff", "target_neg"],
        "url": "https://raw.githubusercontent.com/aetting/lm-diagnostics/master/datasets/NEG-136/NEG-136-NAT.tsv",
        "type": "original",
    },
    "neg136simp": {
        "filename": "original/NEG-136-SIMP.tsv",
        "context_col": ["context_aff", "context_neg"],
        "target_col": ["target_aff", "target_neg"],
        "url": "https://raw.githubusercontent.com/aetting/lm-diagnostics/master/datasets/NEG-136/NEG-136-SIMP.tsv",
        "type": "original",
    },
    "role88": {
        "filename": "original/ROLE-88.tsv",
        "context_col": ["context", "context_r"],
        "target_col": ["expected", "expected_r"],
        "url": "https://raw.githubusercontent.com/aetting/lm-diagnostics/master/datasets/ROLE-88/ROLE-88.tsv",
        "type": "original",
    },
}
