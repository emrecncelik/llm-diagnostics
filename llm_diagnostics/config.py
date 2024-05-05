DATASETS = {
    "neg1500_gen": {
        "filepath": "extended/NEG-1500-SIMP-GEN.txt",
        "context_col": ["context_aff", "context_neg"],
        "target_col": ["target_aff", "target_neg"],
    },
    "neg1500_temp": {
        "filepath": "extended/NEG-1500-SIMP-TEMP.txt",
        "context_col": ["context_aff", "context_neg"],
        "target_col": ["target_aff", "target_neg"],
    },
    "role1500": {
        "filepath": "extended/ROLE-1500.txt",
        "context_col": "context",
        "target_col": "expected",
    },
    "neg136nat": {
        "filepath": "original/NEG-136-NAT.tsv",
        "context_col": ["context_aff", "context_neg"],
        "target_col": ["target_aff", "target_neg"],
    },
    "neg136simp": {
        "filepath": "original/NEG-136-SIMP.tsv",
        "context_col": ["context_aff", "context_neg"],
        "target_col": ["target_aff", "target_neg"],
    },
    "role88": {
        "filepath": "original/ROLE-88.tsv",
        "context_col": "context",
        "target_col": "expected",
    },
}
