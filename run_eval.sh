python eval.py --model_name "meta-llama/Llama-2-7b-hf" \
               --quantization \
               --data_dir datasets \
               --dataset neg136simp \
               --is_affirmative \
               --progress_bar \
               --simplify_a_an "a"

# python eval.py --model_name "akreal/tiny-random-LlamaForCausalLM" \
#                --quantization \
#                --data_dir datasets \
#                --dataset neg136simp \
#                --is_affirmative \
#                --topk 5 \
#                --progress_bar