import json

# Загрузка датасета
with open("dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Приведение всех ключей к правильному виду
corrected_data = []
for entry in data:
    new_entry = {
        "context": entry.get("context") or entry.get("Context"),  # Учитываем возможные ошибки
        "responses": entry.get("responses", [])  # Берём список ответов
    }
    corrected_data.append(new_entry)

# Сохранение исправленного файла
with open("dataset_fixed.json", "w", encoding="utf-8") as f:
    json.dump(corrected_data, f, indent=4, ensure_ascii=False)

print("Исправленный датасет сохранён в dataset_fixed.json")
