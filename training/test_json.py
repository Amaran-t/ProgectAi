import json

try:
    with open("C:/Users/User/Desktop/Ai_Chat/training/dialogues.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        print(f"Загружено {len(data)} записей")
except Exception as e:
    print("Ошибка чтения файла:", e)
