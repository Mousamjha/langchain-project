from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(
    model="gemini-3.5-flash",
    model_provider="google_genai"
)

messages = [
    {
        "role": "system",
        "content": "You are an helpful assistant that provides accurate and concise answers to user questions."
    },
    {
    "role": "user",
    "content": "What is the capital of France?"
    }
]

response = model.invoke(messages)
print(response.content[0]['text'])