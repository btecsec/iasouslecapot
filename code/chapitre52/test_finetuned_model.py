from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from peft import PeftModel
import torch

# Configuration
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = "./concierge_model"

print("Chargement du modèle de base et de l'adaptateur LoRA...")

# 1. Chargement du tokenizer
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

# 2. Chargement du modèle de base en 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto"
)

# 3. Chargement de l'adaptateur fine-tuné (LoRA)
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()

# 4. Pipeline de génération
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# 5. Tests de validation
test_prompts = [
    "<|system|>\nTu es un concierge de luxe sophistiqué.</s>\n<|user|>\nOù est ma commande ?</s>\n<|assistant|>\n",
    "<|system|>\nTu es un concierge de luxe sophistiqué.</s>\n<|user|>\nC'est beaucoup trop cher !</s>\n<|assistant|>\n",
    "<|system|>\nTu es un concierge de luxe sophistiqué.</s>\n<|user|>\nBonjour, qui êtes-vous ?</s>\n<|assistant|>\n",
]

print("\n--- Résultats des tests ---")
for p in test_prompts:
    result = pipe(p, max_new_tokens=100, do_sample=True, temperature=0.7)
    print(f"User: {p.split('<|user|>\n')[1].split('</s>')[0]}")
    print(f"Concierge: {result[0]['generated_text'].split('<|assistant|>\n')[1]}")
    print("-" * 30)
