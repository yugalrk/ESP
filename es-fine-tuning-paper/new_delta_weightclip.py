import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging
import numpy as np
import os
import argparse
import json
from accelerate import Accelerator
import time
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import gc

# --- Setup ---
logging.set_verbosity_error()
torch.backends.cuda.matmul.allow_tf32 = True

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-0.5B-Instruct')
parser.add_argument('--save_dir', type=str, default='full_qwen_concise_es_output')
parser.add_argument('--hf_cache_dir', type=str, default='huggingface_cache')
parser.add_argument('--precision', type=str, default='bf16')
parser.add_argument('--gpu_threads', type=int, default=1, help='Number of parallel threads per GPU')
args = parser.parse_args()

# --- Stable ES Hyperparameters (Tuned) ---
NUM_ITERATIONS = 20
POPULATION_SIZE = 25
SIGMA = 0.01          # Increased for better exploration
ALPHA = 0.0001         # Decreased for weight stability
CLIPPING_THRESHOLD = 0.1 # Threshold for weight update clipping
MAX_NEW_TOKENS = 20  
initial_seed = 33

SYSTEM_PROMPT = "You are a helpful assistant. Give extremely concise answers. Only provide the final result without explanation."
COMPUTE_DTYPE = torch.bfloat16 if args.precision == 'bf16' else (torch.float16 if args.precision == 'fp16' else torch.float32)

# --- Dataset (Subset) ---
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
    """Graduated penalty to prevent reward collapse."""
    gen = generated_text.strip().lower()
    tgt = target_text.strip().lower()
    
    if tgt in gen:
        accuracy_reward = 0.0
    elif any(char.isdigit() for char in gen) and any(char.isdigit() for char in tgt):
        accuracy_reward = -5.0 
    else:
        accuracy_reward = -20.0 
        
    length_penalty = -0.1 * abs(len(gen) - len(tgt))
    return accuracy_reward + length_penalty

def force_memory_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def format_prompt(tokenizer, user_query):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_query}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def safe_decode(tokenizer, token_ids):
    try:
        return tokenizer.decode(token_ids, skip_special_tokens=True)
    except TypeError:
        tokens = tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=True)
        valid_tokens = [t for t in tokens if isinstance(t, str)]
        return tokenizer.convert_tokens_to_string(valid_tokens)

def evaluate_model(model, tokenizer, input_texts, target_texts, accelerator):
    prompts = [format_prompt(tokenizer, text) for text in input_texts]
    tokenized = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left").to(accelerator.device)
    
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=tokenized["input_ids"],
            attention_mask=tokenized["attention_mask"],
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_texts = []
    for i in range(len(input_texts)):
        full_decoded = safe_decode(tokenizer, outputs[i])
        response = full_decoded.split("assistant")[-1].strip()
        generated_texts.append(response)

    return [compute_reward(gen, tgt) for gen, tgt in zip(generated_texts, target_texts)]

def process_seed(seed_args):
    seed_idx, seed, model, tokenizer, accelerator, thread_id = seed_args
    for name, param in model.named_parameters():
        if "weight" in name:
            gen = torch.Generator(device=param.device).manual_seed(int(seed))
            noise = torch.randn(param.shape, generator=gen, device=param.device, dtype=param.dtype)
            param.data.add_(SIGMA * noise)

    input_texts = [d[0] for d in dataset]; target_texts = [d[1] for d in dataset]
    rewards = evaluate_model(model, tokenizer, input_texts, target_texts, accelerator)
    avg_reward = sum(rewards) / len(dataset)

    for name, param in model.named_parameters():
        if "weight" in name:
            gen = torch.Generator(device=param.device).manual_seed(int(seed))
            noise = torch.randn(param.shape, generator=gen, device=param.device, dtype=param.dtype)
            param.data.add_(-SIGMA * noise)

    force_memory_cleanup()
    return seed_idx, avg_reward

def main():
    accelerator = Accelerator()
    device = accelerator.device
    os.makedirs(args.save_dir, exist_ok=True)
    
    metrics_history = []
    model_list = []
    for _ in range(args.gpu_threads):
        model = AutoModelForCausalLM.from_pretrained(args.model_name, cache_dir=args.hf_cache_dir, device_map={"": device}, torch_dtype=COMPUTE_DTYPE)
        model.eval(); model_list.append(model)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    np.random.seed(initial_seed)

    

    for iteration in range(NUM_ITERATIONS):
        iter_start = time.time()
        
        if accelerator.is_main_process:
            seeds = np.random.randint(0, 2**30, size=POPULATION_SIZE).tolist()
            seeds_tensor = torch.tensor(seeds, device=device)
        else:
            seeds_tensor = torch.zeros(POPULATION_SIZE, dtype=torch.long, device=device)
        
        if accelerator.num_processes > 1: torch.distributed.broadcast(seeds_tensor, src=0)
        seeds = seeds_tensor.cpu().tolist()

        local_seeds = [(idx, s) for idx, s in enumerate(seeds) if idx % accelerator.num_processes == accelerator.process_index]
        
        with ThreadPoolExecutor(max_workers=args.gpu_threads) as executor:
            thread_args = [(idx, s, model_list[i % args.gpu_threads], tokenizer, accelerator, i) for i, (idx, s) in enumerate(local_seeds)]
            local_rewards = list(executor.map(process_seed, thread_args))

        all_rewards_tensor = torch.zeros(POPULATION_SIZE, device=device)
        for idx, rew in local_rewards: all_rewards_tensor[idx] = rew
        if accelerator.num_processes > 1: torch.distributed.all_reduce(all_rewards_tensor, op=torch.distributed.ReduceOp.SUM)
        
        rewards = all_rewards_tensor.cpu().numpy()
        mean_reward, min_reward, max_reward = np.mean(rewards), np.min(rewards), np.max(rewards)

        # Weight Update with Clipping
        rewards_norm = (rewards - mean_reward) / (np.std(rewards) + 1e-8)
        main_model = model_list[0]
        total_delta = 0
        weight_count = 0

        for name, param in main_model.named_parameters():
            if "weight" in name:
                update = torch.zeros_like(param)
                for i in range(POPULATION_SIZE):
                    gen = torch.Generator(device=param.device).manual_seed(int(seeds[i]))
                    noise = torch.randn(param.shape, generator=gen, device=param.device, dtype=param.dtype)
                    update.add_(noise * rewards_norm[i])
                
                step = (ALPHA / (POPULATION_SIZE * SIGMA) * update)
                
                # Apply Weight Clipping for stability
                step_norm = torch.norm(step)
                if step_norm > CLIPPING_THRESHOLD:
                    step = step * (CLIPPING_THRESHOLD / step_norm)
                
                param.data.add_(step)
                total_delta += step.abs().mean().item()
                weight_count += 1

        avg_weight_delta = total_delta / weight_count if weight_count > 0 else 0

        # Metrics Persistence
        iter_data = {
            "iteration": iteration + 1, 
            "mean": float(mean_reward), 
            "min": float(min_reward), 
            "max": float(max_reward), 
            "weight_delta": avg_weight_delta,
            "time": time.time() - iter_start
        }
        metrics_history.append(iter_data)

        for i in range(1, len(model_list)):
            for name, param in model_list[i].named_parameters(): param.data.copy_(main_model.get_parameter(name).data)

        if accelerator.is_main_process:
            print(f"Iter {iteration+1}/{NUM_ITERATIONS} | Mean: {mean_reward:.4f} | Max: {max_reward:.4f} | Min: {min_reward:.4f} | Delta: {avg_weight_delta:.6f}")

    if accelerator.is_main_process:
        main_model.save_pretrained(args.save_dir)
        tokenizer.save_pretrained(args.save_dir)
        with open(os.path.join(args.save_dir, "metrics.json"), "w") as f:
            json.dump(metrics_history, f, indent=4)
        print(f"Training complete. Saved to {args.save_dir}")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()