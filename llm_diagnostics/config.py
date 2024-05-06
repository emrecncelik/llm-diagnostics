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
        "context_col": "context",
        "target_col": "expected",
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
        "context_col": "context",
        "target_col": "expected",
        "url": "https://raw.githubusercontent.com/aetting/lm-diagnostics/master/datasets/ROLE-88/ROLE-88.tsv",
        "type": "original",
    },
}
