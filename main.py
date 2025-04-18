from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import pdfplumber
from openai import OpenAI
import os
from dotenv import load_dotenv
import io

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Initialize the client (it automatically loads the API key from the environment variable)
client = OpenAI()

@app.post("/analyze")
async def analyze_pdf(file: UploadFile = File(...)):
    contents = await file.read()
    pdf_text = ""
    with pdfplumber.open(io.BytesIO(contents)) as pdf:
        for page in pdf.pages:
            pdf_text += page.extract_text() + "\n"

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Ты опытный врач. Расшифруй медицинский анализ, дай краткий вывод и рекомендации."},
            {"role": "user", "content": pdf_text}
        ],
        temperature=0.5,
        max_tokens=1000
    )

    # Access the response content correctly
    result = response.choices[0].message.content
    return JSONResponse(content={"analysis": result})
