import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging
import numpy as np
import os
import argparse
from accelerate import Accelerator
import time
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import math
import gc
from datasets import load_dataset
import copy 

# --- Configuration and Constants ---
logging.set_verbosity_error()
torch.backends.cuda.matmul.allow_tf32 = True

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-0.5B-Instruct')
parser.add_argument('--hf_cache_dir', type=str, default='huggingface_cache')
parser.add_argument('--precision', type=str, default='bf16', choices=['fp32', 'fp16', 'bf16'])
parser.add_argument('--gpu_threads', type=int, default=1, help='Number of parallel threads per GPU')
parser.add_argument('--verbose', action='store_true', help='Print verbose logs')
args = parser.parse_args()


# --- Standardized ES Hyperparameters ---
NUM_ITERATIONS = 10        # Synced to 10
POPULATION_SIZE = 25       # Synced to 25
SIGMA = 0.05           # Synced to 0.05
ALPHA = 0.002           # Synced to 0.002
DATASET_SIZE = 60         # Synced to 60
initial_seed = 33         # Synced to 33


# --- Standardized Data Loading and Preprocessing for SST-2 ---
print("Loading SST-2 dataset...")
sst2 = load_dataset("glue", "sst2", split="train") 
dataset = sst2.select(range(DATASET_SIZE))


def preprocess_dataset(dataset):
    """Applies formatting and extracts the prompt and target key for SST-2."""
    processed_data = []
    for item in dataset:
        prompt = (
            f"Sentence: \"{item['sentence']}\" Is this positive or negative? "
            f"Answer with '0' for negative or '1' for positive.\nAnswer:"
        )
        processed_data.append(
            {
                "prompt": prompt,
                "target_key": str(item["label"]), # '0' or '1' as a string
            }
        )
    return processed_data


processed_dataset = preprocess_dataset(dataset)
INPUT_TEXTS = [item["prompt"] for item in processed_dataset]
TARGET_KEYS = [item["target_key"] for item in processed_dataset]
# -----------------------------------------------


def force_memory_cleanup():
    """Force aggressive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()


# --- Standardized Reward Function (Log-Prob) ---
def evaluate_model_log_prob(model, tokenizer, accelerator, input_texts, target_keys, **kwargs):
    """Calculates the dense reward based on the log-probability of the correct answer token."""

    answer_token_ids = {}
    for label in ["0", "1"]:  # Standardized labels for SST-2
        token_id = tokenizer.encode(label, add_special_tokens=False)
        if token_id and len(token_id) == 1:
            answer_token_ids[label] = token_id[0]
        else:
            answer_token_ids[label] = -1

    tokenized_inputs = tokenizer(
        input_texts, return_tensors="pt", padding=True, padding_side="left",
    )
    input_ids = tokenized_inputs["input_ids"].to(accelerator.device)
    attention_mask = tokenized_inputs["attention_mask"].to(accelerator.device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    rewards = []
    last_token_indices = attention_mask.sum(dim=1) - 1

    for i in range(len(input_texts)):
        target_key = target_keys[i]
        target_token_id = answer_token_ids.get(target_key, -1)

        if target_token_id == -1 or target_token_id not in range(logits.shape[-1]):
            rewards.append(-10.0)
            continue

        last_logit_vector = logits[i, last_token_indices[i], :]
        log_probs = torch.log_softmax(last_logit_vector, dim=-1)
        reward = log_probs[target_token_id].item()
        rewards.append(reward)

    del input_ids, attention_mask, outputs, tokenized_inputs, logits
    torch.cuda.empty_cache()

    return sum(rewards) / len(input_texts)
# -----------------------------------------------


def process_seed(seed_args):
    """Function to process a single seed, perturbing/restoring ALL weights."""
    seed_idx, seed, model, tokenizer, accelerator, thread_id, verbose = seed_args

    if verbose:
        print(f"Process {accelerator.process_index} Thread {thread_id} processing seed {seed_idx}")

    # --- Perturbation ---
    gen = torch.Generator(device=model.device)
    gen.manual_seed(int(seed))

    for name, param in model.named_parameters():
        # Iterate over ALL parameters (Full Fine-Tuning)
        noise = torch.randn(
            param.shape, generator=gen, device=param.device, dtype=param.dtype,
        )
        param.data.add_(SIGMA * noise)
        del noise

    if torch.cuda.is_available():
        torch.cuda.synchronize(accelerator.device)

    # Evaluation uses the standardized log-prob function and SST-2 data
    avg_reward = evaluate_model_log_prob(
        model, tokenizer, accelerator, INPUT_TEXTS, TARGET_KEYS,
        seed_idx=seed_idx, thread_id=thread_id, verbose=verbose,
    )

    # --- Restoration ---
    gen = torch.Generator(device=model.device)
    gen.manual_seed(int(seed))

    for name, param in model.named_parameters():
        # Restore ALL parameters
        noise = torch.randn(
            param.shape, generator=gen, device=param.device, dtype=param.dtype,
        )
        param.data.add_(-SIGMA * noise) # Subtract the noise
        del noise

    if torch.cuda.is_available():
        torch.cuda.synchronize(accelerator.device)

    force_memory_cleanup()
    return seed_idx, avg_reward


# --- Main Evolution Strategies Loop (Full Fine-Tuning) ---
def main_full_finetuning():
    accelerator = Accelerator()
    device = accelerator.device
    
    if accelerator.is_main_process:
        print(f"Full Fine-Tuning ES (Standardized) starting.")
        print(f"Population size: {POPULATION_SIZE}, Iterations: {NUM_ITERATIONS}")
        print(f"Sigma: {SIGMA}, Alpha: {ALPHA}")
    
    # --- 1. Model Loading (No LoRA) ---
    model_name = args.model_name
    hf_cache_dir = args.hf_cache_dir
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, cache_dir=hf_cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    dtype = (
        torch.float16
        if args.precision == "fp16"
        else (torch.bfloat16 if args.precision == "bf16" else torch.float32)
    )
    device_map = {"": accelerator.device} if torch.cuda.is_available() else "cpu"
    
    # Load Base Model directly
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, cache_dir=hf_cache_dir, device_map=device_map, torch_dtype=dtype,
    )
    # Ensure ALL parameters are trainable (default behavior without LoRA/PEFT)
    for param in base_model.parameters():
        param.requires_grad = True

    # --- Create copies of the model for parallel threading ---
    model_list = []
    initial_state = base_model.state_dict()

    for i in range(args.gpu_threads):
        with accelerator.local_main_process_first():
            new_model = copy.deepcopy(base_model).to(accelerator.device)
        new_model.load_state_dict(initial_state, strict=True)
        new_model.eval()
        model_list.append(new_model)
        
    master_model = model_list[0] 

    total_trainable_params = sum(p.numel() for p in master_model.parameters() if p.requires_grad)

    if accelerator.is_main_process:
        print(f"Total Trainable Parameters (Full Model): {total_trainable_params:,}")
        print(f"Model copies: {args.gpu_threads} (WARNING: High Memory Usage)")
        
    training_start_time = time.time()
    np.random.seed(initial_seed)
    
    # --- ES Training Loop ---
    for iteration in range(NUM_ITERATIONS):
        iter_start_time = time.time()
        force_memory_cleanup()
        
        # Generate and distribute seeds
        if accelerator.is_main_process:
            seeds = np.random.randint(0, 2**30, size=POPULATION_SIZE, dtype=np.int64).tolist()
            seeds_tensor = torch.tensor(seeds, device=device)
        else:
            seeds_tensor = torch.zeros(POPULATION_SIZE, dtype=torch.long, device=device)

        if accelerator.num_processes > 1:
            torch.distributed.broadcast(seeds_tensor, src=0)
        seeds = seeds_tensor.cpu().tolist()
        
        # Assign seeds to this process
        local_seeds = []
        for seed_idx, seed in enumerate(seeds):
            if seed_idx % accelerator.num_processes == accelerator.process_index:
                local_seeds.append((seed_idx, seed))
        
        # Process seeds in parallel batches using ThreadPoolExecutor
        local_rewards = []
        batch_size = max(1, min(args.gpu_threads, len(local_seeds)))

        for batch_start in range(0, len(local_seeds), batch_size):
            batch_end = min(batch_start + batch_size, len(local_seeds))
            batch_seeds = local_seeds[batch_start:batch_end]
            
            with ThreadPoolExecutor(max_workers=len(batch_seeds)) as executor:
                thread_args = []
                for thread_id, (seed_idx, seed) in enumerate(batch_seeds):
                    model_copy = model_list[thread_id]
                    thread_args.append(
                        (seed_idx, seed, model_copy, tokenizer, accelerator, thread_id, args.verbose,)
                    )

                results = list(executor.map(process_seed, thread_args))
                local_rewards.extend(results)

            force_memory_cleanup()

        # Collect rewards from all processes
        all_rewards = torch.zeros(POPULATION_SIZE, device=device)
        for seed_idx, reward in local_rewards:
            all_rewards[seed_idx] = reward

        if accelerator.num_processes > 1:
            torch.distributed.all_reduce(all_rewards, op=torch.distributed.ReduceOp.SUM)

        rewards_tensor_np = all_rewards.cpu().numpy()
        del all_rewards
        force_memory_cleanup()

        # Normalize rewards
        rewards_tensor = np.array(rewards_tensor_np, dtype=np.float32)
        rewards_normalized = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)

        # --- WEIGHT UPDATE (CRITICAL FP32 ACCUMULATION FIX) ---
        gen = torch.Generator(device=device)

        for name, param in master_model.named_parameters():
            if param.requires_grad:  # All model parameters
                
                # CRITICAL FIX: Use FP32 for accumulation
                update = torch.zeros_like(param, dtype=torch.float32)
                param_fp32 = param.data.detach().clone().float() # Copy current weights to FP32

                for seed_idx in range(POPULATION_SIZE):
                    r_norm = rewards_normalized[seed_idx]
                    seed = seeds[seed_idx]

                    gen.manual_seed(int(seed))
                    # Noise generated in FP32
                    noise = torch.randn(
                        param.shape, generator=gen, device=device, dtype=torch.float32,
                    )

                    noise.mul_(float(r_norm))
                    update.add_(noise)
                    del noise

                update.div_(POPULATION_SIZE)
                
                # Apply update to the FP32 copy
                param_fp32.add_(ALPHA * update)
                
                # Copy updated FP32 data back to the parameter (casting back to model's dtype)
                param.data.copy_(param_fp32.to(param.data.dtype))
                del param_fp32
                torch.cuda.empty_cache()

        # Synchronize weights across all parallel models
        for model_idx in range(len(model_list)):
            target_model = model_list[model_idx]
            for name, param in target_model.named_parameters():
                if param.requires_grad:
                    # Copy from master_model to ensure all threads start with the same weights next iter
                    param.data.copy_(master_model.get_parameter(name).data)

        if torch.cuda.is_available():
            torch.cuda.synchronize(device)

        force_memory_cleanup()

        iter_time = time.time() - iter_start_time
        mean_reward = rewards_tensor.mean().item()
        
        # --- ADDED: Calculate Min and Max Rewards ---
        min_reward = rewards_tensor.min().item()
        max_reward = rewards_tensor.max().item()

        if accelerator.is_main_process:
            print(
                f"Iteration {iteration + 1}/{NUM_ITERATIONS}, Time: {iter_time:.2f}s, "
                f"Mean Reward (Log-Prob): {mean_reward:.4f}, "
                f"Min: {min_reward:.4f}, Max: {max_reward:.4f}"
            )

    if accelerator.is_main_process:
        print(f"Training completed in {time.time() - training_start_time:.2f}s.")

if __name__ == "__main__":
    os.environ["PYTHONWARNINGS"] = "ignore"
    mp.set_start_method("spawn", force=True)
    main_full_finetuning()