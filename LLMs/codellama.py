from transformers import pipeline

pipe = pipeline("text-generation", model="codellama/CodeLlama-7b-Python-hf", max_length = 512)

messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a thug",
    },
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
]

print(pipe(messages))