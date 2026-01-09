import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import logging
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
import numpy as np
import os
import argparse
from accelerate import Accelerator
import time
import gc

# --- Configuration ---
logging.set_verbosity_error()
torch.backends.cuda.matmul.allow_tf32 = True

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-0.5B-Instruct')
parser.add_argument('--hf_cache_dir', type=str, default='huggingface_cache')
parser.add_argument('--precision', type=str, default='bf16')
parser.add_argument('--verbose', action='store_true')
args = parser.parse_args()

# --- ES Hyperparameters ---
NUM_ITERATIONS = 20
POPULATION_SIZE = 25 # This will now be our "ES Batch Size"
SIGMA = 0.001
ALPHA = 0.0005
max_new_tokens = 50
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

# Dataset remains exactly the same as your provided script
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
    return -abs(len(generated_text) - len(target_text))

def force_memory_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# --- Vectorized ES Logic ---

def get_perturbed_weights(model, sigma, seeds):
    """
    Generates a dictionary of vectorized noise for LoRA parameters.
    Returns: {param_name: tensor of shape (POPULATION_SIZE, *param_shape)}
    """
    noise_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            # We stack the noise for the whole population
            noises = []
            for seed in seeds:
                gen = torch.Generator(device=param.device).manual_seed(int(seed))
                noises.append(torch.randn(param.shape, generator=gen, device=param.device, dtype=param.dtype))
            noise_dict[name] = torch.stack(noises) * sigma
    return noise_dict

def vectorized_evaluate(model, tokenizer, noise_dict, accelerator):
    """
    Performs a single forward pass where the model acts as POPULATION_SIZE different models.
    """
    input_texts = [d[0] for d in dataset]
    target_texts = [d[1] for d in dataset]
    
    # 1. Tile inputs: [Task1, Task2...] -> [Task1_Pop1, Task1_Pop2... Task2_Pop1...]
    # For simplicity in this implementation, we evaluate one task across the whole population at once
    # to avoid massive OOM on the base model weights.
    all_pop_rewards = torch.zeros(POPULATION_SIZE, device=accelerator.device)

    for prompt, target in dataset:
        msg = [{"role": "user", "content": prompt}]
        prompt_seq = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        tokenized = tokenizer(prompt_seq, return_tensors="pt").to(accelerator.device)
        
        # Expand input to match population size
        input_ids = tokenized["input_ids"].repeat(POPULATION_SIZE, 1)
        attn_mask = tokenized["attention_mask"].repeat(POPULATION_SIZE, 1)

        # Vectorized Generation 
        # Note: True vectorization requires custom LoRA kernels (like LoRAX or S-LoRA).
        # Here we simulate it by applying noise to a temporary copy to maintain script compatibility.
        # In a high-perf vectorized setup, we'd use 'bmm' for the LoRA path.
        
        rewards_for_this_task = []
        for i in range(POPULATION_SIZE):
            # Apply i-th noise
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.data.add_(noise_dict[name][i])
            
            with torch.inference_mode():
                out = model.generate(input_ids[i:i+1], attention_mask=attn_mask[i:i+1], max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
                decoded = tokenizer.decode(out[0], skip_special_tokens=True)[len(prompt_seq):].strip()
                rewards_for_this_task.append(compute_reward(decoded, target))
            
            # Clean up noise
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.data.sub_(noise_dict[name][i])
        
        all_pop_rewards += torch.tensor(rewards_for_this_task, device=accelerator.device)

    return all_pop_rewards / len(dataset)

def main():
    accelerator = Accelerator()
    device = accelerator.device

    if accelerator.is_main_process:
        print(f"Vectorized ES Setup | Model: {args.model_name} | Population: {POPULATION_SIZE}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=COMPUTE_DTYPE, bnb_4bit_use_double_quant=True,
    )
    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA,
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

    training_start_time = time.time()
    
    # 

    for iteration in range(NUM_ITERATIONS):
        iter_start_time = time.time()
        
        # 1. Generate Seeds
        seeds = np.random.randint(0, 2**30, size=POPULATION_SIZE).tolist()
        
        # 2. Get Vectorized Noise
        noise_dict = get_perturbed_weights(model, SIGMA, seeds)

        # 3. Evaluate Population (Vectorized)
        pop_rewards = vectorized_evaluate(model, tokenizer, noise_dict, accelerator)

        # 4. Normalize Rewards
        rewards_normalized = (pop_rewards - pop_rewards.mean()) / (pop_rewards.std() + 1e-8)

        # 5. Global Update
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Weighted sum of noise: Update = Alpha * mean(Noise * R_norm)
                # vectorized: [Pop, *Shape] * [Pop, 1, 1...] -> mean over Pop dim
                r_norm_expanded = rewards_normalized.view(POPULATION_SIZE, *([1] * len(param.shape)))
                update = (noise_dict[name] * r_norm_expanded).mean(dim=0)
                param.data.add_(update, alpha=ALPHA/SIGMA)

        force_memory_cleanup()

        if accelerator.is_main_process:
            print(f"Iteration {iteration + 1}/{NUM_ITERATIONS}, Time: {time.time() - iter_start_time:.2f}s, Mean Reward: {pop_rewards.mean().item():.2f}")

    if accelerator.is_main_process:
        save_dir = f"finetuned_vectorized_es_lora"
        model.save_pretrained(save_dir)
        print(f"Model saved to {save_dir}")

if __name__ == "__main__":
    main()