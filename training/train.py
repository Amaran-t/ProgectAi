import json
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset

# Загрузка модели и токенайзера
model_name = "sberbank-ai/rugpt3small_based_on_gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Функция загрузки датасета
def load_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Преобразуем данные в список строк (вопрос + ответ)
    train_data = []
    for entry in data:
        context = entry["context"]  # Используем ключ "context"
        for response in entry["responses"]:  # Перебираем все ответы
            train_data.append({"text": f"<|startoftext|>{context}<|sep|>{response}<|endoftext|>"})
    return Dataset.from_dict({"text": [item["text"] for item in train_data]})

# Загрузка датасета
dataset = load_dataset("dataset_fixed.json")

# Токенизация данных
def tokenize_function(examples):
    # Токенизируем текст
    tokenized = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
    # Добавляем метки (labels) для расчёта loss
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Настройки обучения
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="no",
    learning_rate=1e-5,  # Понижаем скорость обучения
    per_device_train_batch_size=4,
    num_train_epochs=10,  # Увеличиваем количество эпох
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10
)

# Настройка тренера
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

# Обучение модели
trainer.train()

# Сохранение модели
model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")
