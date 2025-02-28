source ~/.bashrc
conda init 
conda activate raft

# model_name="/home/dcml0714/raft_llm_as_a_judge/models/lora_cot_raft_lora_cot_raft_feedback_score_1/"
# model_name="prometheus-eval/prometheus-7b-v2.0"
output_dir="evaluation_result"
# Feedback Bench
# for model_name in "/home/dcml0714/raft_llm_as_a_judge/models/lora_cot_raft_lora_cot_raft_feedback_score_1/" "/home/dcml0714/raft_llm_as_a_judge/models/sft_lora_cot_raft_feedback_score_1/"; do
for random_seed in 42; do 
    # for model_name in "/home/dcml0714/raft_llm_as_a_judge/models/lora_cot_raft_lora_cot_raft_feedback_score_1/" "/home/dcml0714/raft_llm_as_a_judge/models/lora_cot_raft_2_epoch/" "/home/dcml0714/raft_llm_as_a_judge/models/lora_sft_lora_sft_cot/"; do
    for model_name in "/home/dcml0714/raft_llm_as_a_judge/models/mistral-7b/lora/cot_single_and_pair"; do
        for n_cot in 1; do
            # for feedback_bench_split in data/feedback_collection_test.json data/feedback_collection_ood_test.json; do
            # for feedback_bench_split in data/flask_eval.json; do
            # for feedback_bench_split in data/mt_bench_eval.json data/vicuna_eval.json data/feedback_collection_ood_test.json data/flask_eval.json; do
            for feedback_bench_split in "data/flask_eval.json"; do
                echo $model_name $feedback_bench_split $n_cot $random_seed
                # python3 inference.py \
                #     --output_dir $output_dir \
                #     --dataset $feedback_bench_split \
                #     --mode feedback_score \
                #     --prompt_dir ../finetuning_utils/prompts \
                #     --model_name_or_path $model_name \
                #     --tokenizer mistralai/Mistral-7B-Instruct-v0.2 \
                #     --num_samples -1 \
                #     --score_generation_mode decode \
                #     --tensor_parallel_size 1 \
                #     --seed $random_seed \
                #     --n_cot $n_cot \
                #     --is_for
               python3 inference.py \
                    --output_dir $output_dir \
                    --dataset $feedback_bench_split \
                    --mode feedback_score \
                    --prompt_dir ../finetuning_utils/prompts \
                    --model_name_or_path $model_name \
                    --tokenizer mistralai/Mistral-7B-Instruct-v0.2 \
                    --num_samples -1 \
                    --score_generation_mode raft \
                    --tensor_parallel_size 1 \
                    --n_cot $n_cot \
                    --seed $random_seed \
                    --is_for \
                    --save
            done
        done
    done
done


