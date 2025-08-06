import google.generativeai as genai
import os

API_KEY = os.environ.get("GEMINI_API_KEY")

genai.configure(api_key=API_KEY)

# for model in genai.list_models():
#     print(model.name)


model = genai.GenerativeModel("models/gemini-2.5-flash")


from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

def chunk_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    return splitter.split_documents(pages)

ipc_chunks = chunk_pdf("data/IPC-Sections.pdf")
bns_chunks = chunk_pdf("data/BNS-Sections.pdf")
map_chunks = chunk_pdf("data/BNS_IPC_Comparative.pdf")

import json
import time
from tqdm import tqdm
import re

def extract_json(text):
    try:
        match = re.search(r'\{.*?\}', text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return None

def generate_qa(chunk):
    prompt = f"""
ONLY using the legal text below, generate a single relevant question and answer.
Format it strictly as:
{{
  "instruction": "<question>",
  "output": "<answer>"
}}

Legal Text:
\"\"\"{chunk}\"\"\"
"""
    try:
        response = model.generate_content(prompt)
        print("🟢 Raw:", response.text[:150])  # Peek at response
        return extract_json(response.text)
    except Exception as e:
        print("❌ Error:", e)
        return None


qa_pairs = []
for i, chunk in tqdm(enumerate(ipc_chunks[:50]), total=50):
    qa = generate_qa(chunk.page_content[:600])  # trim if needed
    if qa:
        qa_pairs.append(qa)
    time.sleep(1.5)

with open("ipc_qa.jsonl", "w") as f:
    for qa in qa_pairs:
        f.write(json.dumps(qa) + "\n")

print("✅ Saved IPC Q&A.")
