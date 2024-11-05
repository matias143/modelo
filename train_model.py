from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
from accelerate import Accelerator
import torch
import torch.nn as nn
import os

# Configuración para evitar fragmentación de memoria
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Ruta del archivo JSONL
jsonl_file_path = "preguntas_respuestas_detalladas.jsonl"

# Cargar el dataset
dataset = load_dataset("json", data_files=jsonl_file_path)
print("Dataset cargado:", dataset)

# Configuración de cuantización
quant_config = BitsAndBytesConfig(load_in_8bit=True)

# Cargar el modelo y el tokenizador
model_name = "lmsys/vicuna-7b-v1.5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto"  # Distribuye el modelo en las GPUs disponibles automáticamente
)

# Configuración de adaptadores LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    r=16,
    lora_alpha=32,
    lora_dropout=0.05
)
model = get_peft_model(model, lora_config)

# Asegurar que los parámetros LoRA sean entrenables y de tipo bfloat16
for param in model.parameters():
    param.data = param.data.to(torch.bfloat16)  # Convertir a bfloat16 para reducir uso de memoria
    param.requires_grad = True  # Habilitar el cálculo de gradientes

# Desactivar `use_cache` para evitar conflictos con `gradient_checkpointing`
model.config.use_cache = False

# Liberar memoria no utilizada en la GPU
torch.cuda.empty_cache()

# Función de tokenización con etiquetas
def tokenize_function(example):
    tokens = tokenizer(example["pregunta"], example["respuesta"], truncation=True, padding="max_length", max_length=128)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

# Tokenizar el dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Configuración de entrenamiento
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,  # Ajusta este valor si es necesario
    gradient_accumulation_steps=16,
    num_train_epochs=3,
    fp16=False,  # Desactivar precisión mixta
    gradient_checkpointing=True,
    optim="adamw_bnb_8bit"
)

# Inicializar Accelerator para manejo de múltiples GPUs
accelerator = Accelerator()

# Preparar el modelo y el dataset con Accelerator
model, tokenized_dataset = accelerator.prepare(model, tokenized_dataset)

# Crear el objeto Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"]
)

# Iniciar el entrenamiento
print("Entrenamiento iniciado en GPUs")
trainer.train()
