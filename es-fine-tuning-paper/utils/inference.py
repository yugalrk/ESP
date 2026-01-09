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
print("Loading Base Model...")
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, torch_dtype=PRECISION, device_map="auto")

print("Loading Merged LoRA Model...")
lora_model = AutoModelForCausalLM.from_pretrained(MERGED_LORA_DIR, torch_dtype=PRECISION, device_map="auto")

print("Loading Full Fine-Tuned Model...")
fft_model = AutoModelForCausalLM.from_pretrained(FULL_FT_DIR, torch_dtype=PRECISION, device_map="auto")

# --- 4. Comparison Function ---
def run_3way_comparison(prompt):
    # Higher limit to see the "tail" of the generation
    compare_max_tokens = 200 
    
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_len = inputs.input_ids.shape[1]
    
    with torch.no_grad():
        # Generation
        out_base = base_model.generate(**inputs, max_new_tokens=compare_max_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        out_lora = lora_model.generate(**inputs, max_new_tokens=compare_max_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        out_fft  = fft_model.generate(**inputs, max_new_tokens=compare_max_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)

    # 5. Extract responses (New Tokens Only)
    resp_ids_base = out_base[0][input_len:]
    resp_ids_lora = out_lora[0][input_len:]
    resp_ids_fft  = out_fft[0][input_len:]

    # Decode
    res_base = tokenizer.decode(resp_ids_base, skip_special_tokens=True).strip()
    res_lora = tokenizer.decode(resp_ids_lora, skip_special_tokens=True).strip()
    res_fft  = tokenizer.decode(resp_ids_fft, skip_special_tokens=True).strip()

    print(f"\nPROMPT: {prompt}")
    print(f"Input Tokens: {input_len}")
    print(f"{'='*60}")
    
    print(f"BASE RESPONSE:")
    print(f"Tokens: {len(resp_ids_base)} | Chars: {len(res_base)}")
    print(f"Text: {res_base}")
    print(f"{'-'*30}")

    print(f"MERGED LORA RESPONSE:")
    print(f"Tokens: {len(resp_ids_lora)} | Chars: {len(res_lora)}")
    print(f"Text: {res_lora}")
    print(f"{'-'*30}")

    print(f"FULL FT RESPONSE:")
    print(f"Tokens: {len(resp_ids_fft)} | Chars: {len(res_fft)}")
    print(f"Text: {res_fft}")
    print(f"{'='*60}")

# --- 6. Run Tests ---
test_queries = [
    "Solve: 12 + 7 =",
    "Complete the sequence: 2, 4, 8, 16, ?",
    "What is the square root of 9?",
    "If A is taller than B and B is taller than C, who is the shortest?"
]

print("\n### STARTING 3-WAY COMPARISON (METRICS MODE) ###")
for q in test_queries:
    run_3way_comparison(q)