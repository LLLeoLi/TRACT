from colorama import Fore, Style
import logging
from typing import List, Dict, Literal, Tuple
import time
from vllm import LLM, RequestOutput
from transformers import AutoTokenizer, AutoConfig
import torch
from tqdm import tqdm
import math
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_vllm_model_and_tokenizer(
    model_name_or_path: str,
    tokenizer_name: str,
    seed: int,
    tensor_parallel_size: int,
    v100: bool,
    max_model_len: int,
): 
    # Load model and tokenizer
    logger.info(f"Loading model: {Fore.GREEN}{model_name_or_path}{Style.RESET_ALL}")
    start_time = time.time()
    config = AutoConfig.from_pretrained(model_name_or_path)


    v100_parameters = {
        'dtype': torch.float16,
        'enable_chunked_prefill': False,
        'gpu_memory_utilization': 0.85,
        "max_model_len": config.max_sequence_length if 'max_sequence_length' in config else max_model_len,
    }

    a6000_parameters = {
        'quantization': 'fp8',
        'gpu_memory_utilization': 0.85,
    }
    model = LLM(
        model = model_name_or_path,
        seed = seed,
        tensor_parallel_size = tensor_parallel_size,
        **(v100_parameters if v100 else a6000_parameters),
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    logger.info(f"Model and tokenizer loaded in {time.time() - start_time:.2f} seconds")
    return model, tokenizer


def get_output_text(
    outputs,
    post_processor = None,
):
    output_texts = []
    for output in outputs:
        assert isinstance(output.outputs, list)
        if len(output.outputs) == 1:
            output_texts.append(
                output.outputs[0].text
            )
        else:
            output_texts.append(
                [out.text for out in output.outputs]
            )
    if post_processor is not None:
        output_texts = [post_processor(o) for o in output_texts]
    
    logger.info(f"{Fore.RED}Before post-processing{Style.RESET_ALL}: {outputs[0].outputs[0].text}")
    logger.info(f"{Fore.GREEN}After post-processing{Style.RESET_ALL}: {output_texts[0]}")

    return output_texts

def raft_score_processor(
    all_logits: torch.Tensor,
    token_idx_to_score: Dict[int, int],
) -> List[float]:
    logger.info(f"All logits have shape: {all_logits.shape}")
    raft_scores = []
    score_token_ids = [k for k in token_idx_to_score.keys()]
    score_grids = [v for _, v in token_idx_to_score.items()]
    score_grids = torch.tensor(score_grids).unsqueeze(0)
    
    all_probs = torch.nn.functional.softmax(all_logits, dim=-1)


    logger.info(f"Seq logprobs shape: {all_probs.shape}")
    score_token_probs = all_probs[..., score_token_ids]
    # Extract the logprobs for the target tokens
    logger.info(f"Score token probs shape: {score_token_probs}")
    logger.info(f"Score grids shape: {score_grids}")
    weighted_scores = (score_grids * score_token_probs).sum(dim=-1)
    logger.info(f"Weighted scores shape: {weighted_scores.shape}")
    raft_scores = weighted_scores.tolist()

    return raft_scores

def get_logits_from_transformer_models(
    model,
    tokenizer,
    inputs: List[str],
    batch_size: int = 4,
    device: Literal['cpu', 'cuda'] = 'cuda',
):
    # Add pad token if not already in the tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    total_batches = math.ceil(len(inputs) / batch_size)

    all_logits = []

    for i in tqdm(range(total_batches), desc="Getting logits", total=total_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        batch_inputs = tokenizer(
            inputs[start:end], 
            return_tensors='pt', 
            padding=True,
            padding_side = 'right',
            truncation=True,
        ).to(device)
        seq_len = batch_inputs['attention_mask'].sum(-1)
        with torch.no_grad():
            outputs = model(**batch_inputs)
            logits = outputs.logits.detach().cpu()
        for idx, (logit, seq_len) in enumerate(zip(logits, seq_len)):
            current_token_seq = batch_inputs['input_ids'][idx]
            #  logger.info(f"The input token ids are: {current_token_seq}")
            #  logger.info(f"The input tokens are: {tokenizer.convert_ids_to_tokens(current_token_seq)}")
            #  logger.info(f"The token we are extracting logits for is: {current_token_seq[seq_len - 1]}")
            #  logger.info(f"This token corresponds to {tokenizer.convert_ids_to_tokens([current_token_seq[seq_len - 1]])}")
            all_logits.append(logit[seq_len - 1])
    all_logits = torch.stack(all_logits)

    return all_logits
    