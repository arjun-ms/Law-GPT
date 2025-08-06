from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", use_auth_token=True)
model = PeftModel.from_pretrained(base, "arjun-ms/mistral-lora-ipc", use_auth_token=True)

# Create proper adapter folder
model.save_pretrained("exported_lora_ipc_adapter")