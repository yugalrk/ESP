import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Configuration ---
# Point to your NEWLY saved model
FINETUNED_PATH = os.path.abspath("./full_qwen_concise_es_output")
# Point to the original snapshot in your cache
BASE_MODEL_PATH = os.path.abspath("./huggingface_cache/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRECISION = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# --- 1. Load Both Models ---
print("Loading Original Base Model...")
base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, torch_dtype=PRECISION, device_map="auto")

print("Loading ES-Finetuned Model...")
ft_tokenizer = AutoTokenizer.from_pretrained(FINETUNED_PATH)
ft_model = AutoModelForCausalLM.from_pretrained(FINETUNED_PATH, torch_dtype=PRECISION, device_map="auto")

# --- 2. Inference Function ---
def run_comparison(prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Give extremely concise answers. Only provide the final result without explanation."},
        {"role": "user", "content": prompt}
    ]
    
    # Format for Qwen
    text = base_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = base_tokenizer(text, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        # Generate Base
        out_base = base_model.generate(**inputs, max_new_tokens=50, do_sample=False, pad_token_id=base_tokenizer.eos_token_id)
        # Generate Finetuned
        out_ft = ft_model.generate(**inputs, max_new_tokens=50, do_sample=False, pad_token_id=ft_tokenizer.eos_token_id)

    # Slice out the assistant's response
    res_base = base_tokenizer.decode(out_base[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()
    res_ft = ft_tokenizer.decode(out_ft[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()

    print(f"\nPROMPT: {prompt}")
    print(f"{'-'*30}")
    print(f"BASE RESPONSE: {res_base}")
    print(f"FT RESPONSE:   {res_ft}")
    print(f"{'='*60}")

# --- 3. Execute ---
test_queries = [
    "Solve: 12 + 7 =",
    "Complete the sequence: 2, 4, 8, 16, ?",
    "What is the square root of 9?",
    "If A is taller than B and B is taller than C, who is the shortest?",
    "who is the president of the united states in 2024?"
]

print("\n" + "### STARTING SIDE-BY-SIDE COMPARISON ###")
for q in test_queries:
    run_comparison(q)