import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- 1. Paths ---
BASE_MODEL_PATH = r"C:\Users\YUGAL\Desktop\thesis\huggingface_cache\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"
MERGED_LORA_DIR = r"C:\Users\YUGAL\Desktop\thesis\merged_qwen_lora"
FULL_FT_DIR     = r"C:\Users\YUGAL\Desktop\thesis\finetuned-full"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRECISION = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

# --- 2. Load Tokenizer ---
print("Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

# --- 3. Load Models ---
print("Loading Models (This may take a minute)...")
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, torch_dtype=PRECISION, device_map="auto")
lora_model = AutoModelForCausalLM.from_pretrained(MERGED_LORA_DIR, torch_dtype=PRECISION, device_map="auto")
fft_model  = AutoModelForCausalLM.from_pretrained(FULL_FT_DIR, torch_dtype=PRECISION, device_map="auto")

# --- 4. Comparison Function ---
def run_3way_steerability_test(prompt):
    # The instruction used to "steer" the model toward conciseness
    system_instruction = "You are a helpful assistant. Be extremely concise. Give only the answer, no explanation."
    
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": prompt}
    ]
    
    # Apply chat template
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(DEVICE)
    input_len = inputs.input_ids.shape[1]
    
    with torch.no_grad():
        out_base = base_model.generate(**inputs, max_new_tokens=150, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        out_lora = lora_model.generate(**inputs, max_new_tokens=150, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        out_fft  = fft_model.generate(**inputs, max_new_tokens=150, do_sample=False, pad_token_id=tokenizer.eos_token_id)

    def process_output(output_ids):
        # Slice off the input prompt tokens
        resp_ids = output_ids[0][input_len:]
        text = tokenizer.decode(resp_ids, skip_special_tokens=True).strip()
        return len(resp_ids), len(text), text

    base_t, base_c, base_txt = process_output(out_base)
    lora_t, lora_c, lora_txt = process_output(out_lora)
    fft_t,  fft_c,  fft_txt  = process_output(out_fft)

    # --- Print Formatting ---
    print(f"\nPROMPT: {prompt}")
    print(f"Input Context: {input_len} tokens")
    print("-" * 100)
    print(f"{'MODEL':<15} | {'TOKENS':<8} | {'CHARS':<8} | {'ACTUAL OUTPUT'}")
    print("-" * 100)
    print(f"{'BASE':<15} | {base_t:<8} | {base_c:<8} | {base_txt}")
    print(f"{'MERGED LORA':<15} | {lora_t:<8} | {lora_c:<8} | {lora_txt}")
    print(f"{'FULL FT':<15} | {fft_t:<8} | {fft_c:<8} | {fft_txt}")
    print("=" * 100)

# --- 5. Run Tests ---
test_queries = [
    "Solve: 12 + 7 =",
    "Complete the sequence: 2, 4, 8, 16, ?",
    "What is the square root of 9?",
    "If A is taller than B and B is taller than C, who is the shortest?",
    "Name the capitals of France, Germany, and Italy?",
    "What is the missing number in the series: 5, 10, 20, 40, ?",
    "Is umbrella needed during a dry and cloudy day?"
]

print("\n### STARTING COMPARISON (SYSTEM PROMPT: CONCISE) ###")
for q in test_queries:
    run_3way_steerability_test(q)