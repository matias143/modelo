from datasets import load_dataset

# Carga tu dataset en formato JSONL
dataset = load_dataset("json", data_files="preguntas_respuestas_detalladas.jsonl")
print(dataset)

from transformers import AutoModelForCausalLM, AutoTokenizer

# Cargar el modelo y el tokenizador
model_name = "lmsys/vicuna-7b-v1.5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map="auto")

def tokenize_function(examples):
    return tokenizer(examples["pregunta"], examples["respuesta"], truncation=True, padding="max_length", max_length=128)

# Aplicamos la tokenización al dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

from transformers import Trainer, TrainingArguments

# Configuramos los parámetros de entrenamiento
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,  # acumular gradientes en 16 pasos
    num_train_epochs=3,
    save_steps=500,
    logging_dir="./logs",
    logging_steps=10,
    fp16=True,
)

# Creamos el objeto Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
)

# Iniciar entrenamiento
trainer.train()

while True:
    question = input("Pregunta: ")
    inputs = tokenizer(question, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_length=100)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Respuesta:", answer)
