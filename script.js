const chatBox = document.getElementById("chatBox");
const userInput = document.getElementById("userInput");

function sendMessage() {
    const message = userInput.value.trim();
    if (message) {
        appendMessage("user", "text", message);

        // Очистка поля ввода
        userInput.value = "";

        // Отправляем запрос на сервер Flask
        fetch("http://127.0.0.1:5000/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ input: message })
        })
        .then(response => response.json())
        .then(data => {
            appendMessage("ai", "text", data.response);
        })
        .catch(error => {
            console.error("Ошибка:", error);
            appendMessage("ai", "text", "Произошла ошибка, попробуйте позже!");
        });
    }
}

function appendMessage(sender, type, content) {
    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${sender}`;
    if (type === "text") {
        messageDiv.textContent = content;
    }
    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

userInput.addEventListener("keypress", function(event) {
    if (event.key === "Enter") {
        sendMessage();
    }
});
