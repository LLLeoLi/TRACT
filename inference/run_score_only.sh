source ~/.bashrc
conda init 
conda activate raft

output_dir="evaluation_result_score_only_no_reference/"
# Feedback Bench

for model_name in "/home/dcml0714/raft_llm_as_a_judge/models/mistral-7b/lora/lora_score_only_cot_raft/" "/home/dcml0714/raft_llm_as_a_judge/models/mistral-7b/lora/lora_score_only_sft/"; do
    for n_cot in 1; do
        # for feedback_bench_split in data/feedback_collection_test.json data/feedback_collection_ood_test.json; do
        # for feedback_bench_split in data/flask_eval.json; do
        for feedback_bench_split in  data/mt_bench_eval.json data/vicuna_eval.json data/feedback_collection_test.json data/feedback_collection_ood_test.json data/flask_eval.json; do
            echo $n_cot
            python3 inference.py \
                --output_dir $output_dir \
                --dataset $feedback_bench_split \
                --mode score_only \
                --prompt_dir ../finetuning_utils/prompts \
                --model_name_or_path $model_name \
                --tokenizer mistralai/Mistral-7B-Instruct-v0.2 \
                --num_samples -1 \
                --score_generation_mode decode \
                --tensor_parallel_size 1 \
                --n_cot $n_cot \
                --remove \
                --is_for
            python3 inference.py \
                --output_dir $output_dir \
                --dataset $feedback_bench_split \
                --mode score_only \
                --prompt_dir ../finetuning_utils/prompts \
                --model_name_or_path $model_name \
                --tokenizer mistralai/Mistral-7B-Instruct-v0.2 \
                --num_samples -1 \
                --score_generation_mode raft \
                --tensor_parallel_size 1 \
                --n_cot $n_cot \
                --is_for \
                --remove \
                --save
        done
    done

done


