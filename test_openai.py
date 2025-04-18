import openai
from openai import OpenAI # Import the client
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize the client (it automatically loads the API key from the environment variable)
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "Ты опытный врач. Расшифруй медицинский анализ."},
        {"role": "user", "content": "Гемоглобин: 160 г/л, Лейкоциты: 12.0, С-реактивный белок: 30 мг/л"}
    ]
)

# Access the response content correctly
print(response.choices[0].message.content) 