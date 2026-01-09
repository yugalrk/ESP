import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_PATH = r"C:\Users\YUGAL\Desktop\thesis\huggingface_cache\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"
ADAPTER_PATH = r"C:\Users\YUGAL\Desktop\thesis\finetuned-Vlora"
SAVE_PATH = "./merged_qwen_lora"

print("Loading base model...")
base = AutoModelForCausalLM.from_pretrained(BASE_PATH, torch_dtype=torch.bfloat16, device_map="cpu")
tokenizer = AutoTokenizer.from_pretrained(BASE_PATH)

print("Applying adapter...")
model = PeftModel.from_pretrained(base, ADAPTER_PATH)

print("Merging weights...")
model = model.merge_and_unload()

print(f"Saving to {SAVE_PATH}...")
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
print("Done!")