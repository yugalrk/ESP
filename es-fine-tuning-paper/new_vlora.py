import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import logging
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
import numpy as np
import os
import argparse
import json
from accelerate import Accelerator
import time
import gc

# --- Configuration ---
logging.set_verbosity_error()
torch.backends.cuda.matmul.allow_tf32 = True

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-0.5B-Instruct')
parser.add_argument('--save_dir', type=str, default='./finetuned_vlora_es_output')
parser.add_argument('--hf_cache_dir', type=str, default='huggingface_cache')
parser.add_argument('--precision', type=str, default='bf16')
args = parser.parse_args()

# --- ES Hyperparameters ---
NUM_ITERATIONS = 20
POPULATION_SIZE = 25 
SIGMA = 0.001        # Reset to your original value
ALPHA = 0.0005       # Reset to your original value
MAX_NEW_TOKENS = 50  # Matches your original
initial_seed = 33
# --- QLoRA Configuration ---
LORA_R = 8 
LORA_ALPHA = 16

if args.precision == 'fp16':
    COMPUTE_DTYPE = torch.float16
elif args.precision == 'bf16':
    COMPUTE_DTYPE = torch.bfloat16
else:
    COMPUTE_DTYPE = torch.float32


# --- Dataset (Exactly as provided in your original script) ---
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
    """Matches your original script's reward function exactly."""
    return -abs(len(generated_text) - len(target_text))

def force_memory_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# --- Vectorized Evaluation Logic ---

def vectorized_evaluate(model, tokenizer, noise_dict, accelerator):
    """
    Optimized: Perturb weights once per population member, then evaluate full dataset.
    Uses the exact prompt format from your original script.
    """
    all_pop_rewards = torch.zeros(POPULATION_SIZE, device=accelerator.device)
    
    # Matching your original prompt logic (raw string inputs)
    input_texts = [d[0] for d in dataset]
    target_texts = [d[1] for d in dataset]

    # Pre-tokenize
    tokenized = tokenizer(input_texts, return_tensors="pt", padding=True, padding_side="left").to(accelerator.device)

    for i in range(POPULATION_SIZE):
        # 1. Apply i-th noise once to LoRA parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.add_(noise_dict[name][i])
        
        # 2. Evaluate entire dataset for this individual
        total_reward_for_indiv = 0
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"],
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            
            for j in range(len(dataset)):
                # Decodes only the new tokens to calculate length reward
                decoded = tokenizer.decode(outputs[j][tokenized["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                total_reward_for_indiv += compute_reward(decoded, target_texts[j])
        
        all_pop_rewards[i] = total_reward_for_indiv / len(dataset)

        # 3. Restore weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.sub_(noise_dict[name][i])
        
        force_memory_cleanup()

    return all_pop_rewards

def main():
    accelerator = Accelerator()
    device = accelerator.device
    os.makedirs(args.save_dir, exist_ok=True)

    # 1. Initialize QLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=COMPUTE_DTYPE, bnb_4bit_use_double_quant=True,
    )
    lora_config = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, cache_dir=args.hf_cache_dir,
        device_map={"": device}, quantization_config=bnb_config,
        torch_dtype=COMPUTE_DTYPE, trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    metrics_history = []
    np.random.seed(initial_seed)

    # Training Loop
    for iteration in range(NUM_ITERATIONS):
        iter_start_time = time.time()
        
        # 1. Generate Seeds
        seeds = np.random.randint(0, 2**30, size=POPULATION_SIZE).tolist()
        
        # 2. Noise Generation for LoRA params
        noise_dict = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                noises = []
                for seed in seeds:
                    gen = torch.Generator(device=param.device).manual_seed(int(seed))
                    noises.append(torch.randn(param.shape, generator=gen, device=param.device, dtype=param.dtype))
                noise_dict[name] = torch.stack(noises) * SIGMA

        # 3. Vectorized (In-Place) Evaluation
        pop_rewards = vectorized_evaluate(model, tokenizer, noise_dict, accelerator)

        # 4. Global Update
        mean_reward = pop_rewards.mean().item()
        rewards_normalized = (pop_rewards - mean_reward) / (pop_rewards.std() + 1e-8)

        total_delta = 0
        weight_count = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                r_norm_expanded = rewards_normalized.view(POPULATION_SIZE, *([1] * len(param.shape)))
                update = (noise_dict[name] * r_norm_expanded).mean(dim=0)
                
                # Applying ALPHA * update / POPULATION_SIZE logic from your original script
                param.data.add_(update, alpha=ALPHA/SIGMA)
                
                total_delta += update.abs().mean().item()
                weight_count += 1

        avg_weight_delta = total_delta / weight_count if weight_count > 0 else 0
        
        # Logging
        iter_data = {
            "iteration": iteration + 1, "mean": mean_reward, 
            "max": pop_rewards.max().item(), "min": pop_rewards.min().item(),
            "weight_delta": avg_weight_delta, "time": time.time() - iter_start_time
        }
        metrics_history.append(iter_data)

        if accelerator.is_main_process:
            print(f"Iter {iteration + 1}/{NUM_ITERATIONS}, Time: {iter_data['time']:.2f}s, Mean Reward: {mean_reward:.2f}, Min: {iter_data['min']:.2f}, Max: {iter_data['max']:.2f}")

    if accelerator.is_main_process:
        model.save_pretrained(args.save_dir)
        tokenizer.save_pretrained(args.save_dir)
        with open(os.path.join(args.save_dir, "metrics.json"), "w") as f:
            json.dump(metrics_history, f, indent=4)
        print(f"Training complete. Saved to {args.save_dir}")

if __name__ == "__main__":
    main()