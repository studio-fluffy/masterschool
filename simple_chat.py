from ollama import chat

response = chat(
    model="llama3.2:1b",
    messages=[
        {
            "role": "user",
            "content": "Erkläre neuronale Netze in 2 Sätzen"
        }
    ]
)

print(response.message.content)
