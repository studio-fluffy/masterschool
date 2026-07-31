from ollama import chat

response = chat(
    model="llama3.2:1b",
    messages=[
        {
            "role": "user",
            "content": "wie funktioniert quantisierung bei llms? Erkläre in 2 sätzen"
        }
    ]
)

print(response.message.content)
