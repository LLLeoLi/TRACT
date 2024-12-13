from datasets import load_dataset
import json
from tqdm import tqdm
from colorama import Fore, Style
import logging
import os
import argparse
from typing import List, Dict, Literal, Tuple
import numpy as np
import time
from vllm import SamplingParams
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import torch
from string import Formatter

from inference_utils.generation_utils import load_vllm_model_and_tokenizer, get_output_text, raft_score_processor, get_logits_from_transformer_models
from inference_utils.evaluation_utils import calculate_correlations
from inference_utils.io_utils import save_results

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(args):
    # Verify the arguments
    if args.validation_ratio < 0 or args.validation_ratio >= 1:
        raise ValueError(f"Invalid validation ratio: {args.validation_ratio}")
    if args.validation_ratio != 0 and args.dataset == 'prometheus-eval/Feedback-Collection':
        raise ValueError(f"Validation is not supported for the dataset {args.dataset}")
    

    # Prepare the output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    logger.info(f"Output directory: {Fore.GREEN}{args.output_dir}{Style.RESET_ALL}")

    # Load and process the dataset
    dataset_load_start_time = time.time()
    dataset = load_dataset(args.dataset, split = 'train')
    dataset = dataset.shuffle(seed = args.seed)
    if args.num_samples > 0:
        dataset = dataset.select(range(args.num_samples))
    num_samples = len(dataset)
    end_time = time.time()
    logger.info(f"Dataset {Fore.GREEN}{args.dataset}{Style.RESET_ALL} ({Fore.GREEN}{num_samples}{Style.RESET_ALL} samples) loaded in {Fore.GREEN}{end_time - dataset_load_start_time:.2f} seconds{Style.RESET_ALL}")
                              
    
    if args.validation_ratio != 0:
        dataset.train_test_split(test_size = args.validation_ratio)
        dataset = dataset['test']

    # Prepare the output file name
    output_file_name = os.path.join(
        args.output_dir, 
        f"{args.dataset.split('/')[-1]}_{args.mode}.jsonl"
    )
    logger.info(f"Output file: {Fore.GREEN}{output_file_name}{Style.RESET_ALL}")

    # Prepare the template file
    if args.mode == "score_only":
        template_file = os.path.join(
            args.prompt_dir,
            "no_feedback.prompt"
        )
    elif args.mode == 'feedback_score':
        template_file = os.path.join(
            args.prompt_dir,
            "with_feedback.prompt"
        )
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

    with open(template_file, 'r', encoding = 'utf-8') as f:
        template = f.read()
    
    # Get the placeholders in the template
    formatter = Formatter()
    keys_to_fill = [key for _, key, _, _ in formatter.parse(template) if key is not None]
    logger.info(f"Loaded template file: {Fore.GREEN}{template_file}{Style.RESET_ALL}, containing the following placeholders: {Fore.YELLOW}{keys_to_fill}{Style.RESET_ALL}")

    # Remove the columns that are not needed
    all_columns = dataset.column_names
    columns_to_remove = [column for column in all_columns if column not in keys_to_fill + ['output']]
    dataset = dataset.remove_columns(columns_to_remove)

    # Collect the labels 
    labels = []
    for idx, instance in tqdm(enumerate(dataset), total = len(dataset), desc = "Collecting labels"):
        output = instance.pop('output')
        labels.append(float(output.split('[RESULT]')[-1].strip()))

    # Generate CoT if the mode is feedback_score
    if args.mode == 'feedback_score':
        model, tokenizer = load_vllm_model_and_tokenizer(
            model_name_or_path = args.model_name_or_path,
            tokenizer_name = args.tokenizer,
            seed = args.seed,
            tensor_parallel_size = 1,
            v100 = args.v100,
            max_model_len = args.max_model_len
        )
        formatted_inputs = []
        for idx, instance in tqdm(enumerate(dataset), total = len(dataset), desc = "Formatting CoT generation prompts"):
            output = instance.pop('output')
            full_input_str = template.format(**instance)
            lm_input = tokenizer.apply_chat_template(
                [
                    {
                        'role': 'user',
                        'content': full_input_str
                    }
                ],
                tokenize = False,
                add_generation_prompt = True,
            )
            formatted_inputs.append(lm_input)


        # Generate the CoT
        start_time = time.time()
        logger.info(f"Generating CoT for {Fore.GREEN}{len(dataset)}{Style.RESET_ALL} samples")
        logger.info(f"Example input:\n{Fore.YELLOW}{formatted_inputs[0]}{Style.RESET_ALL}")
        cot_outputs = model.generate(
            formatted_inputs,
            sampling_params = SamplingParams(
                temperature = args.temperature,
                top_p = args.top_p,
                repetition_penalty = args.repetition_penalty,
                max_tokens = args.max_cot_tokens,
                stop = ['[RESULT]', ' [RESULT]']
            ),
        )

        

        def cot_splitter(cot):
            return cot.split('So the overall')[0].strip() + " So the overall score is "

        # We should add ' [RESULT] ' to the end of the output. Note that there is a space after the [RESULT]. This may be model-specific.
        cot_outputs = get_output_text(
            cot_outputs,
            post_processor = lambda x: x.replace('[RESULT]', '').strip() + ' [RESULT] '
            # post_processor = cot_splitter,

        )
        end_time = time.time()
        logger.info(f"CoT generated in {Fore.GREEN}{end_time - start_time:.2f} seconds{Style.RESET_ALL}")
        del model
        del tokenizer
    
    else:
        cot_outputs = [""] * len(dataset)

    # Load the model and tokenizer
    if args.score_generation_mode == 'decode':
        model, tokenizer = load_vllm_model_and_tokenizer(
            model_name_or_path = args.model_name_or_path,
            tokenizer_name = args.tokenizer,
            seed = args.seed,
            tensor_parallel_size = args.tensor_parallel_size,
            v100 = args.v100,
            max_model_len = args.max_model_len,
        )
    elif args.score_generation_mode == 'raft':
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            device_map = 'auto',
            load_in_4bit = True,
        )
        model = model.eval()
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Prepare the inputs
    formatted_inputs = []
    for idx, (instance, cot) in tqdm(enumerate(zip(dataset, cot_outputs)), total = len(dataset), desc = "Formatting score generation prompts"):
        full_input_str = template.format(**instance)
        # TODO: Verify the two behaviors are identical:
        # 1. CoT = "", add_generation_prompt = False, continue_final_message = True
        # 2. No user prompt, add_generation_prompt = True, continue_final_message = False
        lm_input = tokenizer.apply_chat_template(
            [
                {
                    'role': 'user',
                    'content': full_input_str
                },
                {
                    'role': 'assistant',
                    'content': cot,
                }
            ],
            tokenize = False,
            add_generation_prompt = False,
            continue_final_message = True,
        )
        lm_input = lm_input.replace("</s>", "")
        formatted_inputs.append(lm_input)

    logger.info(f"Example of the input:\n{Fore.YELLOW}{formatted_inputs[0]}{Style.RESET_ALL}<-End of the input")

    # Generate one token using vllm

    start_time = time.time()
    logger.info(f"Generating scores for {Fore.GREEN}{len(dataset)}{Style.RESET_ALL} samples")

    if args.score_generation_mode == 'decode':
        raw_outputs = model.generate(
            formatted_inputs,
            sampling_params = SamplingParams(
                max_tokens = 5,
                temperature = args.temperature,
                top_p = args.top_p,
                repetition_penalty = args.repetition_penalty,
            ),
        )
        maybe_scores = get_output_text(raw_outputs)
        print(maybe_scores)

    elif args.score_generation_mode == 'raft':
        raw_outputs = get_logits_from_transformer_models(
            model = model,
            tokenizer = tokenizer,
            inputs = formatted_inputs
        )
        maybe_scores = raft_score_processor(
            raw_outputs,
            token_idx_to_score = {
                28740: 1,
                28750: 2,
                28770: 3,
                28781: 4,
                28782: 5,
            }
        )
        
    end_time = time.time()

    logger.info(f"Example score: {Fore.YELLOW}{maybe_scores[0]}{Style.RESET_ALL}")

    # Calculate the correlation between predicted and true scores
    all_correlations = calculate_correlations(
        groundtruth = labels,
        predictions = maybe_scores,
    )

    # Save the results
    save_results(
        output_dir = args.output_dir,
        all_correlations = all_correlations,
        args = vars(args)
    )

    if args.num_samples != -1:
        logger.warning(f"{Fore.RED}WARNING{Style.RESET_ALL}: Only {Fore.RED}{args.num_samples}{Style.RESET_ALL} samples were processed. This may mean that not all samples are processed. Double-check if this is the intended behavior.")



if __name__ == '__main__':
    '''
    Example usage:
    python3 inference.py \
        --output_dir evaluation_result \
        --dataset prometheus-eval/Feedback-Bench \
        --mode score_only \
        --prompt_dir ../finetuning_utils/prompts \
        --model_name_or_path ../models/lm/score_only/ \
        --tokenizer mistralai/Mistral-7B-Instruct-v0.2 \
        --num_samples 30 \
        --score_generation_mode decode \
        --tensor_parallel_size 1

    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type = str, required = True)
    parser.add_argument("--seed", type = int, default = 42)
    parser.add_argument("--num_samples", type = int, default = 2048, help = "The number of samples to process. Use -1 to process all samples.")
    parser.add_argument("--model_name_or_path", type=str, required = True)
    parser.add_argument("--tokenizer", type = str, default = None)
    parser.add_argument("--dataset", required = True, choices = ['prometheus-eval/Feedback-Bench', 'prometheus-eval/Feedback-Collection'], help = "The Feedback-Collection is the dataset used for training, while the Feedback-Bench is the dataset used for evaluation.")
    parser.add_argument("--mode", choices = ['score_only', 'feedback_score'], required = True)
    parser.add_argument("--score_generation_mode", choices = ['decode', 'raft'], required = True)
    parser.add_argument('--tensor_parallel_size', type = int, default = 2)
    parser.add_argument('--temperature', type = float, default = 1.0)
    parser.add_argument('--top_p', type = float, default = 0.9)
    parser.add_argument('--repetition_penalty', type = float, default = 1.03)
    parser.add_argument('--max_cot_tokens', type = int, default = 1024)
    parser.add_argument("--prompt_dir", type = str, default = 'prompts')
    parser.add_argument("--v100", action = 'store_true', help = "Use the V100 GPU")
    parser.add_argument("--max_model_len", type = int, default = 4096)
    parser.add_argument("--validation_ratio", type = float, default = 0.1, help = "The ratio of the dataset to be used for validation splitted from the training set.")
    args = parser.parse_args()
    main(args)
