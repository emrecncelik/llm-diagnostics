
RANDOM_MODELS=(
    # "yujiepan/tiny-random-bert"
    "yujiepan/llama-2-tiny-random"
    "yujiepan/llama-3-tiny-random"
    "yujiepan/mistral-tiny-random"
    "yujiepan/mixtral-8xtiny-random"
    "yujiepan/mixtral-tiny-random"
    "yujiepan/mamba-tiny-random"
    "yujiepan/qwen-vl-tiny-random"
    "yujiepan/qwen1.5-tiny-random"
)

for model_name in "${RANDOM_MODELS[@]}"
do
    python eval.py --model_name "$model_name" \
                   --data_dir datasets \
                   --dataset neg136simp \
                   --is_affirmative \
                   --progress_bar \
                   --simplify_a_an "a"
done