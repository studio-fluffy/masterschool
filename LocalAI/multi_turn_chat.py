from ollama import chat

messages = []
print("Chat mit llama3.2:1b — tippe quit zum Beenden")

while True:
    user_input = input("Du: ").strip()

    if user_input.lower() in ["quit", "exit"]:
        break

    messages.append({"role": "user", "content": user_input})

    response = chat(model="llama3.2:1b", messages=messages)
    reply = response.message.content

    messages.append({"role": "assistant", "content": reply})
    print(f"KI: {reply}\n")
