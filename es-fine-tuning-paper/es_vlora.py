import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging
import numpy as np
import os
import argparse
from accelerate import Accelerator
import time
import math
import gc
# --- Import dataset loader (required) ---
from datasets import load_dataset 
from peft import LoraConfig, get_peft_model, TaskType
import copy 
# --- Import ThreadPoolExecutor (required for reward evaluation, though vectorized ES is typically single-threaded for evaluation) ---
# Since this version is single-threaded (no GPU threads used), we remove the unused Executor import.


logging.set_verbosity_error()
torch.backends.cuda.matmul.allow_tf32 = True

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-0.5B-Instruct')
parser.add_argument('--hf_cache_dir', type=str, default='huggingface_cache')
parser.add_argument('--precision', type=str, default='bf16')
parser.add_argument('--verbose', action='store_true', help='Print verbose logs')
parser.add_argument('--lora_r', type=int, default=16, help='LoRA attention dimension')
parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha parameter')
args = parser.parse_args()


# --- ES Hyperparameters ---
NUM_ITERATIONS = 100       # Number of ES iterations (increased for stability)
POPULATION_SIZE = 50       # Population size (increased for better gradient)
SIGMA = 0.05          # Noise scale (increased slightly from 0.003 for better exploration)
ALPHA = 0.00005         # Learning rate (Reduced for stability on a simple task)
DATASET_SIZE = 500        # Number of samples to use from SST-2
initial_seed = 33        # Initial random seed


# --- Data Loading and Preprocessing for SST-2 ---
print("Loading SST-2 dataset...")
sst2 = load_dataset("glue", "sst2", split="train") 
dataset = sst2.select(range(DATASET_SIZE))

def preprocess_dataset(dataset):
  """Applies formatting and extracts the prompt and target key for SST-2."""
  processed_data = []
  for item in dataset:
    # Formulate the prompt for binary classification
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
  gc.collect()
  if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.synchronize()

# --- REPLACED evaluate_model_batch with Log-Probability Reward ---
def evaluate_model_log_prob(model, tokenizer, accelerator, input_texts, target_keys):
  """Calculates the dense reward based on the log-probability of the correct answer token."""

  # 1. Get the token ID for the answer labels ('0' and '1')
  answer_token_ids = {}
  for label in ["0", "1"]: 
    token_id = tokenizer.encode(label, add_special_tokens=False)
    if token_id and len(token_id) == 1:
      answer_token_ids[label] = token_id[0]
    else:
      # This should not happen with Qwen tokenizer and '0'/'1'
      print(f"Warning: Tokenizer failed to map '{label}' to a single token ID.")
      answer_token_ids[label] = -1

  # 2. Tokenize inputs
  tokenized_inputs = tokenizer(
    input_texts,
    return_tensors="pt",
    padding=True,
    padding_side="left",
  )
  input_ids = tokenized_inputs["input_ids"].to(accelerator.device)
  attention_mask = tokenized_inputs["attention_mask"].to(accelerator.device)

  # 3. Forward pass to get logits
  with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits

  # 4. Calculate reward (log-probability of the correct token)
  rewards = []
  # Index of the last token in the sequence (where the model is predicting next)
  last_token_indices = attention_mask.sum(dim=1) - 1

  for i in range(len(input_texts)):
    target_key = target_keys[i]
    target_token_id = answer_token_ids.get(target_key, -1)

    if target_token_id == -1 or target_token_id not in range(logits.shape[-1]):
      rewards.append(-10.0) # Penalty
      continue

    # Get the logits for the position right after the prompt
    last_logit_vector = logits[i, last_token_indices[i], :]

    # Convert logits to log probabilities
    log_probs = torch.log_softmax(last_logit_vector, dim=-1)

    # Reward is the log probability of the correct answer token.
    reward = log_probs[target_token_id].item()
    rewards.append(reward)

  del input_ids, attention_mask, outputs, tokenized_inputs, logits
  torch.cuda.empty_cache()

  return sum(rewards) / len(input_texts)
# -------------------------------------------------------------


# --- Vectorization Class (Unchanged and Correct) ---

class PerturbationMap:
  """Manages the flattening and reconstruction of LoRA adapter weights."""
  def __init__(self, model):
    self.map = []
    self.total_size = 0
    self.device = model.device
    
    # 1. Create mapping and calculate total size
    for name, param in model.named_parameters():
      if param.requires_grad:
        size = param.numel()
        self.map.append({
          'name': name,
          'start': self.total_size,
          'end': self.total_size + size,
          'shape': param.shape,
          'dtype': param.dtype
        })
        self.total_size += size

  def flatten_weights(self, model):
    """Extracts all trainable LoRA weights into a single 1D tensor."""
    vector = torch.empty(self.total_size, dtype=self.map[0]['dtype'], device=self.device)
    for item in self.map:
      param = model.get_parameter(item['name'])
      vector[item['start']:item['end']] = param.data.flatten()
    return vector

  def apply_vector(self, model, vector):
    """Copies the 1D vector back into the corresponding LoRA module tensors."""
    for item in self.map:
      # Slicing and viewing is highly efficient as it's a single copy operation
      param = model.get_parameter(item['name'])
      param.data.copy_(vector[item['start']:item['end']].view(item['shape']))


# --- Main Vectorized Evolution Strategies Loop ---

def main():
  accelerator = Accelerator()
  device = accelerator.device
  torch.manual_seed(initial_seed)
  
  if accelerator.is_main_process:
    print(f"Total processes: {accelerator.num_processes}")
    print(f"Population size: {POPULATION_SIZE}, Iterations: {NUM_ITERATIONS}")
    print(f"Sigma: {SIGMA}, Alpha: {ALPHA}, LoRA R: {args.lora_r}, LoRA Alpha: {args.lora_alpha}")

  # Load tokenizer
  model_name = args.model_name
  hf_cache_dir = args.hf_cache_dir
  tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, cache_dir=hf_cache_dir)
  
  if accelerator.is_main_process:
    print(f"Loading and configuring ONE model instance {model_name} with LoRA...")

  # --- LoRA Configuration ---
  lora_config = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    # Expanded targets for better optimization, especially for classification
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], 
    lora_dropout=0.0,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
  )
  # ---------------------------

  dtype = torch.float16 if args.precision == 'fp16' else (torch.bfloat16 if args.precision == 'bf16' else torch.float32)
  
  # Load and wrap model
  base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=hf_cache_dir,
    device_map={"": device}, 
    torch_dtype=dtype,
  )
  original_model = get_peft_model(base_model, lora_config)
  original_model.eval() 
  
  # --- Initialization for Vectorized ES ---
  mapper = PerturbationMap(original_model)
  
  # V_current: The single vector of all current LoRA weights (the optimization state)
  # **CRITICAL: Store V_current in FP32 to prevent precision loss during accumulation, even if original weights are BF16**
  V_current = mapper.flatten_weights(original_model).float() 
  total_trainable_params = V_current.numel()
  
  if accelerator.is_main_process:
    print("LoRA PEFT model loaded successfully.")
    original_model.print_trainable_parameters()
    print(f"Total trainable parameters (vectorized): {total_trainable_params:,}")

  force_memory_cleanup()

  # Pre-generate noise vectors for the entire population in FP32
  # This stores the noise for the whole run, saving CPU time during iterations
  noise_vectors = []
  
  for seed in np.random.randint(0, 2**30, size=POPULATION_SIZE, dtype=np.int64):
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    noise = torch.randn(
      total_trainable_params,
      generator=gen,
      device=device,
      dtype=torch.float32 # <-- Ensure noise is always FP32
    )
    noise_vectors.append(noise)
    
  # Generate the initial set of seeds for reward normalization
  # seeds = np.random.randint(0, 2**30, size=POPULATION_SIZE, dtype=np.int64).tolist()
  # The noise vectors are already pre-generated and stored in `noise_vectors`
  
  # Record total training start time
  training_start_time = time.time()

  # --- ES Training Loop ---
  for iteration in range(NUM_ITERATIONS):
    iter_start_time = time.time()
    force_memory_cleanup()

    rewards = []
    
    # 1. Evaluation (Sequential but efficient)
    for seed_idx in range(POPULATION_SIZE):
      noise_vector = noise_vectors[seed_idx]
      
      # --- Perturbation (One vector operation) ---
      # V_current is FP32, SIGMA is float, noise_vector is FP32. Operation is stable.
      V_perturbed = V_current + SIGMA * noise_vector 

      # --- Reconstruction (One efficient copy) ---
      # V_perturbed (FP32) is cast back to the model's native dtype (BF16/FP16)
      mapper.apply_vector(original_model, V_perturbed)

      # --- Evaluation (Using Log-Prob Reward) ---
      avg_reward = evaluate_model_log_prob(original_model, tokenizer, accelerator, INPUT_TEXTS, TARGET_KEYS)
      rewards.append(avg_reward)
      
    del V_perturbed
    force_memory_cleanup()
    
    # 2. Update Step (One efficient vector operation)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
    rewards_normalized = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
    
    # Update vector is also FP32
    update = torch.zeros_like(V_current) 
    
    # Compute the weighted sum (R_i * Z_i) efficiently
    for seed_idx in range(POPULATION_SIZE):
      r_norm = rewards_normalized[seed_idx]
      # Accumulate the weighted noise into the update vector (FP32 accumulation)
      update.add_(noise_vectors[seed_idx], alpha=r_norm)
      
    update.div_(POPULATION_SIZE)
    
    # Apply the final update to the current weight vector (V_current is FP32)
    V_current.data.add_(ALPHA * update)
    
    # --- Final Reconstruction ---
    # Apply the final optimized V_current back to the physical model weights
    mapper.apply_vector(original_model, V_current)
    
    del rewards_tensor, rewards_normalized, update
    
    # Final synchronization
    if torch.cuda.is_available():
      torch.cuda.synchronize(accelerator.device)

    force_memory_cleanup()

    iter_time = time.time() - iter_start_time

    mean_reward = np.mean(rewards)
    min_reward = np.min(rewards)
    max_reward = np.max(rewards)

    if accelerator.is_main_process:
      print(f"Iteration {iteration + 1}/{NUM_ITERATIONS}, Time: {iter_time:.2f}s, Mean Reward (Log-Prob): {mean_reward:.4f}, Min: {min_reward:.4f}, Max: {max_reward:.4f}")
      if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.2f}MB allocated, {torch.cuda.max_memory_allocated() / 1024**2:.2f}MB peak")
      else:
        print("GPU Memory: N/A (CUDA not available)")


  total_time = time.time() - training_start_time


  # Save the fine-tuned model weights (only the LoRA adapters)
  if accelerator.is_main_process:
    print(f"Training completed in {total_time:.2f}s ({total_time/60:.2f} minutes)")
    question_num = len(processed_dataset)
    save_dir = f"finetuned_{model_name.split('/')[-1]}_vector_es_r{args.lora_r}_alpha{args.lora_alpha}_pop{POPULATION_SIZE}_iter{NUM_ITERATIONS}_sigma{SIGMA}_alpha{ALPHA}_{args.precision}_question_num{question_num}"
    print(f"Saving LoRA adapters to {save_dir}...")
    original_model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"LoRA adapters saved successfully. To load: `model = PeftModel.from_pretrained(base_model, save_dir)`")

if __name__ == "__main__":
  os.environ["PYTHONWARNINGS"] = "ignore"
  main()