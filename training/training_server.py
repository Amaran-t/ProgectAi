from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import time
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset

app = Flask(__name__)
CORS(app)

training_thread = None
training_running = False
log_messages = []

MODEL_PATH = "./trained_model"
DATASET_PATH = "C:/Users/User/Desktop/Ai_Chat/training/dialogues.json"

# Функция для логирования событий
def log(message):
    log_messages.append(message)
    print(message)

# Функция загрузки данных
def load_dataset():
    if os.path.exists(DATASET_PATH):
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            log(f"Загружено {len(data)} записей из dialogues.json")
            return data
    return []

# Функция для сохранения новых диалогов
def save_dialogue(user_input, bot_response):
    data = load_dataset()
    new_entry = {"context": user_input, "responses": [bot_response]}
    data.append(new_entry)
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# Функция обучения модели
def train_model():
    global training_running
    log("Функция train_model() вызвана! Начинаем обучение...")
    
    data = load_dataset()
    if len(data) < 10:
        log("Недостаточно данных для обучения.")
        training_running = False
        return
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    train_texts = [f"<|startoftext|>{d['context']}<|sep|>{r}<|endoftext|>" for d in data for r in d["responses"]]
    train_dataset = Dataset.from_dict({"text": train_texts})
    
    def tokenize_function(examples):
        tokenized = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    tokenized_dataset = train_dataset.map(tokenize_function, batched=True)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=4,
        num_train_epochs=3,
        logging_dir="./logs",
        logging_steps=10,
        save_steps=500,
        save_total_limit=2
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    
    trainer.train()
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)
    log("Обучение завершено. Модель сохранена.")
    training_running = False

# Запуск обучения
@app.route("/start-training", methods=["POST"])
def start_training():
    global training_thread, training_running
    if training_running:
        return jsonify({"status": "Обучение уже запущено"})
    training_running = True
    training_thread = threading.Thread(target=train_model, daemon=True)
    training_thread.start()
    return jsonify({"status": "Обучение запущено"})

# Остановка обучения
@app.route("/stop-training", methods=["POST"])
def stop_training():
    global training_running
    if not training_running:
        return jsonify({"status": "Обучение не запущено"})
    training_running = False
    return jsonify({"status": "Обучение остановлено, сохранение модели..."})

# Получение логов обучения
@app.route("/get-log", methods=["GET"])
def get_log():
    return "\n".join(log_messages)

# Сохранение диалогов при общении
@app.route("/save-dialogue", methods=["POST"])
def save_user_dialogue():
    data = request.json
    user_input = data.get("input", "")
    bot_response = data.get("response", "")
    save_dialogue(user_input, bot_response)
    return jsonify({"status": "Диалог сохранён"})

if __name__ == "__main__":
    app.run(port=5001)
