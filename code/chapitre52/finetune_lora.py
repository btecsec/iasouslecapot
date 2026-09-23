"""
SCRIPT DE FINE-TUNING : SPÉCIALISATION D'UN LLM GÉNÉRAL

OBJECTIF :
Ce script transforme un modèle de langage généraliste (qui sait parler de tout)
en un expert spécifique : un "Concierge de Luxe". On ne change pas ses connaissances,
mais on modifie son COMPORTEMENT et son STYLE (ton sophistiqué).

APPROCHE TECHNIQUE :
Le script utilise le "Supervised Fine-Tuning" (SFT) avec la méthode LoRA.

OUTILS ET FONCTIONS CLÉS :
1. 'transformers' (Hugging Face) : Pour charger le modèle de base et le tokenizer.
2. 'datasets' : Pour gérer et formater les données d'entraînement en JSONL.
3. 'bitsandbytes' (via load_in_4bit) : Permet la QUANTISATION. On réduit la précision
   du modèle pour qu'il consomme beaucoup moins de mémoire vidéo (VRAM).
4. 'peft' (Parameter-Efficient Fine-Tuning) :
   - 'prepare_model_for_kbit_training' : Prépare le modèle quantifié pour l'entraînement.
   - 'LoraConfig' & 'get_peft_model' : Implémente LoRA (Low-Rank Adaptation). Au lieu de
     modifier les milliards de paramètres du modèle, on ajoute de petites matrices
     entraînables. C'est 100x plus léger et rapide.
5. 'trl' (Transformer Reinforcement Learning) :
   - 'SFTTrainer' : Une fonction haut niveau qui encapsule toute la boucle d'entraînement
     complexe (optimiseur, calcul de loss, itérations) pour la rendre simple.

RÉSUMÉ DU FLUX :
1) Modèle Général → 2)Quantisation (4-bit) → 3) Ajout de couches LoRA →
4) Entraînement sur données spécifiques → 5) Sauvegarde de l'adaptateur spécialisé.

Explication DU FLUX :
1) Modèle général : on part d'un LLM pré-entraîné (Llama, Mistral, Gemma...) qui possède des connaissances générales mais n'est pas spécialisé sur une tâche précise.
2) Quantisation 4-bit : les poids du modèle de base sont compressés de FP16/FP32 vers 4 bits (souvent au format NF4, NormalFloat 4-bit). Cela divise la mémoire nécessaire par environ 4x, permettant de charger un modèle de 70 milliards de paramètres sur un seul GPU 24 Go au lieu d'en nécessiter plusieurs.
3) Ajout de couches LoRA : plutôt que de ré-entraîner tous les poids (des milliards), on injecte de petites matrices de rang réduit (A et B) à côté des couches existantes, typiquement sur les projections d'attention (q_proj, v_proj, k_proj, o_proj) et les couches feed-forward.
4) Entraînement sur données spécifiques : seuls les paramètres LoRA sont mis à jour pendant le fine-tuning ; les poids d'origine (gelés en 4-bit) ne bougent pas. Cela réduit drastiquement le nombre de paramètres entraînables et donc le coût mémoire/calcul.
5) Sauvegarde de l'adaptateur spécialisé : à la fin, on ne sauvegarde que les matrices LoRA (généralement 20-100 Mo), pas le modèle complet. Cet adaptateur peut ensuite être combiné (mergé) au modèle de base pour l'inférence

"""

import torch
import os
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
# On définit tout ici pour pouvoir changer facilement sans fouiller dans le code
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TRAIN_DATA_PATH = os.path.join(SCRIPT_DIR, "data", "train.jsonl")
TEST_DATA_PATH = os.path.join(SCRIPT_DIR, "data", "test.jsonl")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "concierge_model")

# Paramètres d'entraînement (Hyperparamètres)
LEARNING_RATE = 2e-4
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
MAX_STEPS = 40 # Nombre d'étapes d'entraînement

# ==============================================================================
# 2. PRÉPARATION DES DONNÉES
# ==============================================================================
def format_prompt(example):
    """
    Transforme une ligne JSONL en format de chat compréhensible par le modèle.
    Format : <|system|> ... </s><|user|> ... </s><|assistant|> ... </s>
    """
    return f"<|system|>\nTu es un concierge de luxe sophistiqué.</s>\n<|user|>\n{example['instruction']}</s>\n<|assistant|>\n{example['output']}</s>"

print("📦 Chargement et formatage des données...")
# Chargement des fichiers JSONL
train_dataset = load_dataset("json", data_files=TRAIN_DATA_PATH, split="train")
test_dataset = load_dataset("json", data_files=TEST_DATA_PATH, split="train")

# Application du formatage à chaque exemple
train_dataset = train_dataset.map(lambda x: {"text": format_prompt(x)})
test_dataset = test_dataset.map(lambda x: {"text": format_prompt(x)})

# ==============================================================================
# 3. CONFIGURATION DU MODÈLE (L'ARCHITECTURE)
# ==============================================================================
print("🤖 Chargement du modèle et configuration LoRA...")

# Tokenizer : convertit le texte en nombres (tokens)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Chargement du modèle de base en 4-bit (Quantisation)
# Cela permet de faire tenir le modèle sur un GPU grand public
# La quantisation consiste à réduire la précision numérique des poids d'un modèle IA (ex : passer de 32 bits à 8 ou 4 bits) pour le rendre plus léger et plus rapide, sans trop sacrifier sa qualité.
# Un modèle stocke normalement ses poids en FP32 (32 bits) ou FP16 (16 bits) — des nombres à virgule flottante très précis mais volumineux.
# La quantisation convertit ces poids vers des formats plus compacts comme INT8 (8 bits) ou INT4 (4 bits)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)

# Étape cruciale pour LoRA : prépare le modèle quantifié pour l'entraînement
model = prepare_model_for_kbit_training(model)

# Configuration LoRA (Low-Rank Adaptation)
# Au lieu de modifier 1 milliard de paramètres, on en ajoute quelques milliers
lora_config = LoraConfig(
    r=8,                        # Taille des matrices de bas rang
    lora_alpha=32,              # Facteur d'échelle
    target_modules=["q_proj", "v_proj"], # On cible les couches d'attention
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# ==============================================================================
# 4. LE FINE-TUNING (L'ENTRAÎNEMENT)
# ==============================================================================
print("\n🚀 Lancement du Fine-Tuning...")

# Paramètres techniques de l'entraînement
training_args = SFTConfig(
    output_dir="./results_temp",
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    logging_steps=1,
    max_steps=MAX_STEPS,
    bf16=True,                 # Utilisation de BFloat16 (plus stable pour LLM)
    eval_strategy="steps",
    eval_steps=10,
    max_length=512,
)

# SFTTrainer : L'outil simplifié pour le "Supervised Fine-Tuning"
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    args=training_args,
)

# On lance l'apprentissage
trainer.train()

# ==============================================================================
# 5. SAUVEGARDE ET EXPORT
# ==============================================================================
print("\n💾 Sauvegarde du modèle spécialisé...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"✅ Terminé ! Modèle enregistré dans : {OUTPUT_DIR}")
