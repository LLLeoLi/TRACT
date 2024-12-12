# RAFT for LLM-as-a-Judge

This repo contains the under-developping code for [RAFT](https://openreview.net/pdf?id=8euJaTveKw) for LLM-as-a-judge.

> ⚠️ **Warning:** This repo is under active development.


## Setup

First, clone this repo by 
```
git clone https://github.com/d223302/raft-for-llm-as-a-judge
```


Create a conda environment by
```
conda create -y -n raft python=3.11
conda activate raft
```

Next, install all the necessary packages. Since we will be using [LLaMA-Factory
](https://github.com/hiyouga/LLaMA-Factory/tree/main) for fine-tuning, please go to [LLaMA-Factory
](https://github.com/hiyouga/LLaMA-Factory/tree/main) and follow the [installation guides](https://github.com/hiyouga/LLaMA-Factory/tree/main?tab=readme-ov-file#getting-started).


Also, install `colorama` for printing colored words on the terminal by
```
pip install colorama
```

## Fine-tuning

### Prepare Fine-tuning dataset

We are using LLaMA-Factory for fine-tuning, and we will need to prepare the data to match the format for LLaMA-Factory.

We will use two datasets: [`prometheus-eval/Feedback-Collection`](https://huggingface.co/datasets/prometheus-eval/Feedback-Collection?row=0) for training and [`prometheus-eval/Feedback-Bench`](https://huggingface.co/datasets/prometheus-eval/Feedback-Bench) for evaluation.

Generate the dataset using the script `prepare_dataset.py`.

An example usage can be:

```
python3 prepare_dataset.py \
    --output_dir data \
    --dataset prometheus-eval/Feedback-Collection \
    --mode feedback_score \
    --prompt_dir prompts \
    --tokenizer mistralai/Mistral-7B-Instruct-v0.2 \
    --num_samples 20
```
Refer to the arguments to see how to use this script.

After running this script, you should see a `*.jsonl` in the output directory. Copy it to the data directory of Llama-Factory, which should located at `$llama-factory_root/data`.
You will also need to modify the `data_info.json` in `$llama-factory_root/data`. Refer to [here](https://github.com/hiyouga/LLaMA-Factory/tree/main/data) to see how to modify it. You can also find a .json snippet from the output of running `prepare_dataset.py`, which can be slightly modified to be copy-paste to the `data_info.json`

### Prepare the Llama-Factory

Since RAFT is not a standard fine-tuning that is supported by `transformers` and `llama-factory`, we will need to manually modify some codes in the two packages for our need.
 
#### Transformers

Please find where you install transformers and find the file `src/transformers/trainer_seq2seq.py`.
Add two lines of codes following [this PR](https://github.com/huggingface/transformers/pull/35136/files).
This is because the `Trainer` class supports the `compute_loss_func`, but this is not yet supported.

#### Llama-Factory

We need to slightly modify the codes related to SFT in Llama-Factory.

- **Step 1**: Copy `finetuning_utils/llama_factory_utils.py` to `$llama-factory_root/src/llamafactory/train/sft`.
In `finetuning_utils/llama_factory_utils.py`, we define the RAFT loss.

- **Step 2**: Replace the `$llama-factory_root/src/llamafactory/train/sft/workflow.py` with `finetuning_utils/sft_workflow.py`. In this replacement, we incoporate the RAFT loss function to the trainer and update the data collator to incoporate for the loss function change.

> ⚠️ **Warning:** Currently, the RAFT is hard-coded for Mistral-7B-Instruct-v0.2. If you want to use other models, you must change the `score_to_indices` and `indices_to_scores` in `finetuning_utils/llama_factory_utils.py`


### Start Fine-tuning

Fine-tuning uses Llama-Factory. You can follow the how fine-tuning is done in Llama-Factory. For example, you can copy the `finetuning_utils/finetune_raft.yaml` to `$llama-factory_root/examples/train_lora` and run fine-tuning by 
```
cd $llama-factory_root
llamafactory-cli train examples/finetune_raft.yaml
```

I haven't check the hyperparameters in the config, so they may not be optimal

## Inference

The inference codes are in `inference/`.
Run inference by 
```
python3 inference.py \
    --output_dir evaluation_result \
    --dataset prometheus-eval/Feedback-Bench \
    --mode feedback_score \
    --prompt_dir ../finetuning_utils/prompts \
    --model_name_or_path ../models/lm/feedback_score/ \
    --tokenizer mistralai/Mistral-7B-Instruct-v0.2 \
    --num_samples 30 \
    --score_generation_mode raft \
    --tensor_parallel_size 2
```
Use `python3 inference.py -h` to see how to use the arguments.
Some important arguments:
- `--mode`: This determines whether to generate CoT then the score (`feedback_score`) or directly generate the score (`score_only`)
- `--score_generation_mode`: Whether to generate the score using standard LM decoding/sampling (`decode`) or using MALI (`raft`).


The codes for RAFT during inference are defined by `raft_score_processor` located in `inference/inference_utils/generation_utils.py`


## TODO
- [x] Prepare the inference code
- [ ] The instruction prompt for score-only is slightly wrong. Need to re-generate the dataset for score-only and fine-tune using LM loss + score-only again.
- [ ] Double check the code to see if there are any errors in the dataset processing. The result I obtained by fine-tuning is much higher than the results in the original paper.
- [ ] The Feedback-Collection's CoT output target contains sentences like *"So the overall score is 5."* Conditioning on this and sample a score using MALI is odd. Need to remove those sentences from the training data and re-train the model
- [ ] Create a new training pipeline in Llama-factory instead of directly modifying the SFT-related codes.
- [x] RAFT + CoT? 
- [ ] The current loss is only averaged on the same device but not across device. This should be fixed in `transformers==4.47.0`. However, Llama-Factory currently only supports `transformers<=4.46.1`.
- [ ] The RAFT loss is currently hard-coded for Mistral-7b-Instruct. Make it adaptable to other models.