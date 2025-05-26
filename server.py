from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)
CORS(app)  # Включаем CORS для всех маршрутов

# Указываем путь к обученной модели
model_name = "C:\\Users\\User\\Desktop\\Ai_Chat\\trained_model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("input", "")

    # Генерация ответа
    inputs = tokenizer.encode(user_input, return_tensors="pt")
    outputs = model.generate(
    inputs,
    max_length=50,
    temperature=0.9,  # Повышаем креативность
    top_k=50,
    top_p=0.9,  # Увеличиваем вариативность
    do_sample=True
)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(port=5000)
