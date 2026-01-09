import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging
import numpy as np
import copy
import os
import argparse
from accelerate import Accelerator
import time
import torch.multiprocessing as mp
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import math
import gc

logging.set_verbosity_error()
torch.backends.cuda.matmul.allow_tf32 = True

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-0.5B-Instruct')
parser.add_argument('--hf_cache_dir', type=str, default='huggingface_cache')
parser.add_argument('--precision', type=str, default='bf16')
parser.add_argument('--gpu_threads', type=int, default=1, help='Number of parallel threads per GPU')
parser.add_argument('--verbose', action='store_true', help='Print verbose logs')
args = parser.parse_args()


# Hyperparameters for ES
NUM_ITERATIONS = 20             # Number of ES iterations (generations)
POPULATION_SIZE = 25              # Population size (number of perturbations per iteration)
SIGMA = 0.001                     # Standard deviation for weight perturbations (noise scale)
ALPHA = 0.0005                    # Learning rate
max_new_tokens = 100              # Maximum number of tokens allowed to be generated
do_sample = False                 # Whether sampling is allowed in generating tokens, default to be not allowed (greedy decoding for ES)
initial_seed = 33                 # Initial random seed


# --- Dummy Dataset and Reward Function ---
# In practice, define a set of input reasoning tasks with desired targets.
dataset = [
    # --- Simple Arithmetic (Sequencing and Calculation) ---
    ("Solve: 12 + 7 =", "19"),
    ("Calculate: 18 - 9 =", "9"),
    ("What is 5 multiplied by 4?", "20"),
    ("What is 100 divided by 5?", "20"),
    ("If I have 15 apples and eat 6, how many are left?", "9"),
    ("Add 22 and 11.", "33"),
    ("Subtract 14 from 30.", "16"),
    ("Solve: 9 * 3 =", "27"),
    ("Calculate: 45 / 5 =", "9"),
    ("If a train has 50 seats and 12 are empty, how many are full?", "38"),
    ("Find the sum of 1, 2, 3, and 4.", "10"),
    ("Solve: 7 * 7 =", "49"),
    ("Calculate: 50 - 18 =", "32"),
    ("What is 1 + 1 + 1 + 1 + 1?", "5"),
    ("Find the remainder of 10 / 3.", "1"),
    ("If a day has 24 hours, how many hours are in half a day?", "12"),
    ("Solve: 15 + 15 + 5 =", "35"),
    ("Calculate: 72 / 9 =", "8"),
    ("What is 8 * 8?", "64"),
    ("If a box holds 12 eggs, and I have 3 boxes, how many eggs total?", "36"),

    # --- Simple Deduction/Logic (Rule Application) ---
    ("If all mammals are warm-blooded and cats are mammals, is a cat warm-blooded?", "Yes"),
    ("If the light is green, the car must go. The light is green. What happens?", "The car must go"),
    ("If A is before B, and B is before C, is A before C?", "Yes"),
    ("The object is either metal or wood. It is not metal. What is it?", "Wood"),
    ("Rule: Only squares are red. Is a red circle allowed?", "No"),
    ("Rule: All fruits are sweet. Is a savory apple possible?", "No"),
    ("If the sky is blue and roses are red, what is the sky color?", "Blue"),
    ("Is it true that all numbers ending in 0 are even?", "Yes"),
    ("If Monday follows Sunday, what follows Tuesday?", "Wednesday"),
    ("If the box contains a key and a lock, which item can open the lock?", "Key"),
    ("All dogs bark. My pet is a dog. What does my pet do?", "Barks"),
    ("If the statement 'The sun rises in the West' is false, then the sun rises in the...?", "East"),
    ("The category is 'Shapes'. Is 'Triangle' a valid member?", "Yes"),
    ("The category is 'Vegetables'. Is 'Banana' a valid member?", "No"),
    ("Deduce: If all cars have wheels, does a boat need wheels?", "No"),

    # --- Sequence and Pattern Reasoning ---
    ("Reverse the sequence: 1, 2, 3, 4, 5.", "5, 4, 3, 2, 1"),
    ("What comes next in the pattern: A, C, E, G, ?", "I"),
    ("Reverse the sequence: Cat, Dog, Bird.", "Bird, Dog, Cat"),
    ("Complete the pattern: Red, Blue, Red, Blue, Red, ?", "Blue"),
    ("Reverse the digits: 901.", "109"),
    ("Sequence: 10, 20, 40, 80, ?", "160"),
    ("What is the next number: 5, 10, 15, 20, ?", "25"),
    ("Reverse the words: Quick brown fox.", "Fox brown quick"),
    ("Complete the pattern: Circle, Square, Circle, Square, ?", "Circle"),
    ("Sequence: 1, 4, 9, 16, ?", "25"),
    ("Reverse the colors: Green, Yellow, Red.", "Red, Yellow, Green"),
    ("Complete the sequence: Z, Y, X, W, ?", "V"),
    ("If the alphabet starts A, B, C, what is the 5th letter?", "E"),
    ("Reverse the numbers: 777.", "777"),
    ("What comes next: January, February, March, ?", "April"),

    # --- Mixed Practice Samples (55 samples) ---
    ("Solve: 5 + 18 - 3 =", "20"),
    ("If a square has 4 sides and a triangle has 3, how many total sides?", "7"),
    ("The sequence is 2, 4, 6, 8, 10. What is the next number?", "12"),
    ("Is the number 17 prime or composite?", "Prime"),
    ("If today is Friday, what was yesterday?", "Thursday"),
    ("Reverse the order: Moon, Sun, Earth.", "Earth, Sun, Moon"),
    ("Calculate the difference: 40 and 15.", "25"),
    ("If P means 'Positive' and N means 'Negative', what is P + N?", "Neutral"),
    ("Solve: 10 * 10 - 1 =", "99"),
    ("Sequence: 1, 1, 2, 3, 5, 8, ?", "13"),
    ("If all squares are rectangles, is a square a rectangle?", "Yes"),
    ("Reverse the word: racecar.", "racecar"),
    ("What is 100 + 50 - 25?", "125"),
    ("If it is cold outside, do I need a coat?", "Yes"),
    ("Complete the sequence: Do, Re, Mi, Fa, ?", "Sol"),
    ("Calculate: (3 + 3) * 2 =", "12"),
    ("If the rule is 'contains blue', does 'sky and clouds' match?", "Yes"),
    ("Reverse the phrase: Hello world.", "World hello"),
    ("What is 7 * 6?", "42"),
    ("Sequence: 10, 9, 8, 7, ?", "6"),
    ("If the category is 'Planets', is 'Star' a member?", "No"),
    ("Solve: 25 - 5 * 2 =", "15"),
    ("What is the opposite of 'Up'?", "Down"),
    ("Reverse the numbers: 12345.", "54321"),
    ("If all pencils write, does a broken pencil write?", "No"),
    ("Calculate the product of 11 and 3.", "33"),
    ("Sequence: O, T, T, F, F, S, S, ?", "E"), # First letter of numbers 1-8
    ("If the temperature is 0 degrees, is it freezing?", "Yes"),
    ("Reverse: The quick red fox.", "Fox red quick The"),
    ("What is 9 + 9 + 9?", "27"),
    ("If the battery is low, do I need to charge it?", "Yes"),
    ("Sequence: Z, X, V, T, R, ?", "P"),
    ("Solve: 10 + 2 * 5 =", "20"),
    ("If something is liquid, can you hold it in your hand?", "No"),
    ("Reverse: Five four three two one.", "One two three four five"),
    ("Calculate 99 minus 1.", "98"),
    ("If rain means wet, what does sunshine mean?", "Dry"),
    ("What is the next item: Apple, Banana, Cherry, ?", "Grape"),
    ("Solve: 10 / 2 + 3 =", "8"),
    ("If all dogs have four legs, does a three-legged dog violate the rule?", "No"),
    ("Reverse the word: Python.", "nohtyP"),
    ("What is 12 + 12?", "24"),
    ("Sequence: 100, 90, 80, 70, ?", "60"),
    ("If the button is 'Start', what is the opposite button?", "Stop"),
    ("Calculate: 100 - 45.", "55"),
    ("Reverse the letters: ABC.", "CBA"),
    ("What is the square root of 9?", "3"),
    ("Sequence: March, May, July, September, ?", "November"),
    ("If you are asleep, are you awake?", "No"),
    ("Solve: 50 * 2 / 10 =", "10"),
    ("Reverse the number 4321.", "1234"),
    ("What is 10 + 20 + 30?", "60"),
    ("If all circles are round, is a square round?", "No"),
    ("Complete the pattern: Small, Large, Small, Large, ?", "Small"),
    ("Solve: 10 - 2 - 3 - 4 =", "1"),
]

def compute_reward(generated_text, target_text):
    # Negative absolute difference in length
    return -abs(len(generated_text) - len(target_text))
 
def force_memory_cleanup():
    """Force aggressive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()

def evaluate_model(model, tokenizer, input_text, target_text, accelerator, seed_idx=None, thread_id=None, verbose=False, return_text=False):
    """
    Generate a response from the model given an input (single or batch) and compute rewards.
    """
    if verbose:
        print(f"Process {accelerator.process_index} Thread {thread_id} evaluating seed {seed_idx}")

    # Handle both single input and batch input
    is_batch = isinstance(input_text, list)
    input_texts = input_text if is_batch else [input_text]
    target_texts = target_text if is_batch else [target_text]

    # Batch tokenization
    tokenized_inputs = tokenizer(input_texts, return_tensors="pt", padding=True, padding_side="left")
    input_ids = tokenized_inputs["input_ids"].to(accelerator.device)
    attention_mask = tokenized_inputs["attention_mask"].to(accelerator.device)
    with torch.inference_mode():
        outputs = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens, do_sample=do_sample)
        if torch.cuda.is_available():
            torch.cuda.synchronize(accelerator.device)

    # Decode batch outputs
    generated_texts = []
    for i in range(len(input_texts)):
        try:
            generated_text = tokenizer.decode(outputs[i], skip_special_tokens=True)
        except TypeError:
            tokens = tokenizer.convert_ids_to_tokens(outputs[i], skip_special_tokens=True)
            filtered = [t for t in tokens if t is not None]
            generated_text = tokenizer.convert_tokens_to_string(filtered)
        generated_texts.append(generated_text)

    del input_ids, outputs
    torch.cuda.empty_cache()

    # Compute rewards for batch texts
    rewards = [compute_reward(gen_text, tgt_text) for gen_text, tgt_text in zip(generated_texts, target_texts)]


    if return_text:
        return rewards, generated_texts
    else:
        return rewards

def process_seed(seed_args):
    """Function to process a single seed, used for thread pool"""
    seed_idx, seed, model, tokenizer, accelerator, thread_id, verbose = seed_args

    if verbose:
        print(f"Process {accelerator.process_index} Thread {thread_id} processing seed {seed_idx} (value: {seed})")

    # Weight perturbation
    for name, param in model.named_parameters():
        gen = torch.Generator(device=param.device)

        gen.manual_seed(int(seed))

        noise = torch.randn(
            param.shape,
            generator=gen,
            device=param.device,
            dtype=param.dtype
        )
        param.data.add_(SIGMA * noise)

    # Ensure weights are fully loaded before evaluation
    if torch.cuda.is_available():
        torch.cuda.synchronize(accelerator.device)

    # Evaluate all prompts with perturbed weights in batch
    input_texts = [input_text for input_text, _ in dataset]
    target_texts = [target_text for _, target_text in dataset]
    rewards = evaluate_model(model, tokenizer, input_texts, target_texts, accelerator,
                           seed_idx=seed_idx, thread_id=thread_id, verbose=verbose, return_text=False)
    total_reward = sum(rewards)

    # Restore original weights (direct inplace modification)
    for name, param in model.named_parameters():
        gen = torch.Generator(device=param.device)

        gen.manual_seed(int(seed))

        noise = torch.randn(
            param.shape,
            generator=gen,
            device=param.device,
            dtype=param.dtype
        )
        param.data.add_(-SIGMA * noise)

    if torch.cuda.is_available():
        torch.cuda.synchronize(accelerator.device)

    average_reward = total_reward / len(dataset)


    force_memory_cleanup()

    if verbose:
        print(f"Process {accelerator.process_index} Thread {thread_id} completed seed {seed_idx} with reward {average_reward:.4f}")

    return seed_idx, average_reward


# --- Main Evolution Strategies Loop ---
def main():
    accelerator = Accelerator()

    if accelerator.is_main_process:
        print(f"Total processes: {accelerator.num_processes}, GPU threads per process: {args.gpu_threads}")
        print(f"Population size: {POPULATION_SIZE}, Iterations: {NUM_ITERATIONS}")
        print(f"Sigma: {SIGMA}, Alpha: {ALPHA}")

    # Load model
    model_name = args.model_name
    hf_cache_dir = args.hf_cache_dir

    if accelerator.is_main_process:
        print(f"Loading model {model_name}...")
    

    # Load model
    model_list = []
    for model_index in range(args.gpu_threads):
        model_list.append(AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=hf_cache_dir,
            device_map={"": accelerator.process_index},  # Assign devices explicitly
            torch_dtype=torch.float16 if args.precision == 'fp16' else (torch.bfloat16 if args.precision == 'bf16' else torch.float32),
        ))
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, cache_dir=hf_cache_dir)

    if accelerator.is_main_process:
        print("Model loaded successfully")

    # Prepare model with accelerator
    for model in model_list:
        model.eval()  # Turn off dropout, etc.

    force_memory_cleanup()

    # Record total training start time
    training_start_time = time.time()

    np.random.seed(initial_seed)

    for iteration in range(NUM_ITERATIONS):
        # Record iteration start time
        iter_start_time = time.time()

        # Force garbage collection
        force_memory_cleanup()

        if args.verbose:
            print(f"Process {accelerator.process_index} starting iteration {iteration + 1}/{NUM_ITERATIONS}")

        # Generate seeds on main process only
        if accelerator.is_main_process:
            if args.verbose:
                print(f"Main process {accelerator.process_index} generating seeds")
            seeds = np.random.randint(0, 2**30, size=POPULATION_SIZE, dtype=np.int64).tolist()
            seeds_tensor = torch.tensor(seeds, device=accelerator.device)
        else:
            if args.verbose:
                print(f"Worker process {accelerator.process_index} waiting for seeds")
            seeds_tensor = torch.zeros(POPULATION_SIZE, dtype=torch.long, device=accelerator.device)

        # Broadcast seeds from main process to all processes
        if accelerator.num_processes>1:
            torch.distributed.broadcast(seeds_tensor, src=0)
        seeds = seeds_tensor.cpu().tolist()  # Convert back to list for all processes

        if args.verbose:
            print(f"Process {accelerator.process_index} received seeds")

        # Assign seeds to each process for processing
        local_seeds = []
        for seed_idx, seed in enumerate(seeds):
            # Simple task assignment: assign seeds by process ID
            if seed_idx % accelerator.num_processes == accelerator.process_index:
                local_seeds.append((seed_idx, seed))

        if args.verbose:
            print(f"Process {accelerator.process_index} assigned {len(local_seeds)} seeds: {[idx for idx, _ in local_seeds]}")

        # Process seeds in smaller batches to reduce memory pressure
        local_rewards = []
        batch_size = max(1, min(args.gpu_threads, len(local_seeds)))

        for batch_start in range(0, len(local_seeds), batch_size):
            batch_end = min(batch_start + batch_size, len(local_seeds))
            batch_seeds = local_seeds[batch_start:batch_end]

            with ThreadPoolExecutor(max_workers=len(batch_seeds)) as executor:
                # Prepare thread arguments
                thread_args = []
                for thread_id, (seed_idx, seed) in enumerate(batch_seeds):
                    # Pass verbose flag as argument to process_seed function
                    thread_args.append((seed_idx, seed, model_list[thread_id], tokenizer, accelerator, thread_id, args.verbose))

                # Execute in parallel and collect results
                results = list(executor.map(process_seed, thread_args))
                local_rewards.extend(results)

            # Clean up between batches
            force_memory_cleanup()

        # Collect rewards from all processes
        all_rewards = torch.zeros(POPULATION_SIZE, device=accelerator.device)

        # Fill in locally computed rewards
        for seed_idx, reward in local_rewards:
            all_rewards[seed_idx] = reward

        # Aggregate rewards from all processes (each process will get the full reward list)
        if accelerator.num_processes>1:
            torch.distributed.all_reduce(all_rewards, op=torch.distributed.ReduceOp.SUM)

        # Convert aggregated rewards back to Python list
        rewards = all_rewards.cpu().tolist()
        # Clean up no longer needed tensor
        del all_rewards
        force_memory_cleanup()

        # Convert rewards to a tensor and normalize.
        rewards_tensor = np.array(rewards, dtype=np.float32)
        rewards_normalized = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)

        # Aggregate perturbations and update model weights
        if args.verbose:
            print(f"Process {accelerator.process_index} updating model weights")
        original_model = model_list[0]
        for name, param in original_model.named_parameters():
            gen = torch.Generator(device=param.device)
            update = torch.zeros_like(param)
            for seed_idx in range(POPULATION_SIZE):
                r_norm = rewards_normalized[seed_idx]
                seed = seeds[seed_idx]
                gen.manual_seed(int(seed))

                noise = torch.randn(
                    param.shape,
                    generator=gen,
                    device=param.device,
                    dtype=param.dtype
                )
                noise.mul_(float(r_norm))
                update.add_(noise)
                del noise
            update.div_(POPULATION_SIZE)
            param.data.add_(ALPHA * update)
            torch.cuda.empty_cache()

        for model_idx in range(1, len(model_list)):
            original_model_tmp = model_list[model_idx]
            for name, param in original_model_tmp.named_parameters():
                param.data.copy_(original_model.get_parameter(name).data.clone())

        # Synchronize to ensure weight updates are complete
        if torch.cuda.is_available():
            torch.cuda.synchronize(accelerator.device)

        force_memory_cleanup()

        iter_time = time.time() - iter_start_time

        mean_reward = rewards_tensor.mean().item()
        min_reward = rewards_tensor.min().item()
        max_reward = rewards_tensor.max().item()

        del rewards_tensor, rewards_normalized
        force_memory_cleanup()

        if accelerator.is_main_process:
            print(f"Iteration {iteration + 1}/{NUM_ITERATIONS}, Time: {iter_time:.2f}s, Mean: {mean_reward:.2f}, Min: {min_reward:.2f}, Max: {max_reward:.2f}")
            print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.2f}MB allocated, {torch.cuda.max_memory_allocated() / 1024**2:.2f}MB peak")

    total_time = time.time() - training_start_time


    # Save the fine-tuned model weights.
    if accelerator.is_main_process:
        print(f"Training completed in {total_time:.2f}s ({total_time/60:.2f} minutes)")
        question_num = len(dataset)
        save_dir = f"finetuned_{model_name}_es_random_seed{initial_seed}_pop{POPULATION_SIZE}_iter{NUM_ITERATIONS}_sigma{SIGMA}_alpha{ALPHA}_{args.precision}_threads{args.gpu_threads}_question_num{question_num}_correct"
        print(f"Saving model to {save_dir}...")
        original_model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"Model saved successfully.")

if __name__ == "__main__":
    os.environ["PYTHONWARNINGS"] = "ignore"
    mp.set_start_method('spawn', force=True)
    main()