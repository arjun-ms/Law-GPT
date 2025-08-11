import os
from pydantic import BaseModel
import torch
import gradio as gr
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# HF token from environment
token = os.environ.get("HF_TOKEN")
if token:
    print("HF TOKEN FOUND AND LOADING....")
    login(token)
else:
    print("NO HF TOKEN FOUND!")

# Model & adapter
# base_id = "mistralai/Mistral-7B-Instruct-v0.2"
# lora_id = "arjun-ms/mistral-lora-ipc"

#  Merged model path
merged_model_id = "arjun-ms/mistral-7b-ipc-merged"

# ✅ Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    llm_int8_enable_fp32_cpu_offload=True  # CPU offloading
)


# # Tokenizer
# tokenizer = AutoTokenizer.from_pretrained(base_id)

# # Base model with quantization
# base_model = AutoModelForCausalLM.from_pretrained(
#     base_id,
#     quantization_config=bnb_config,
#     device_map="auto",                     # Automatically offloads to CPU/GPU/disk
#     trust_remote_code=True,
#     torch_dtype=torch.float16             # Helps reduce memory use
# )


# Apply your LoRA adapter
# model = PeftModel.from_pretraineBaseModelel, lora_id)


# ✅ Tokenizer for merged model
tokenizer = AutoTokenizer.from_pretrained(merged_model_id)

# ✅ Load merged model
model = AutoModelForCausalLM.from_pretrained(
    merged_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16
)



# Response generator
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True,
        top_p=0.95,
        top_k=40
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Gradio UI
iface = gr.Interface(
    fn=generate_response,
    inputs="text",
    outputs="text",
    title="IPC Mistral 7B",
    description="LoRA fine-tuned Mistral 7B for Indian Penal Code questions"
)

iface.launch()