from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import pdfplumber
from openai import OpenAI
import os
from dotenv import load_dotenv
import io

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize FastAPI
app = FastAPI()

@app.post("/analyze")
async def analyze_pdf(file: UploadFile = File(...)):
    contents = await file.read()
    pdf_text = ""

    # Extract text from PDF
    with pdfplumber.open(io.BytesIO(contents)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:  # Check for empty page
                pdf_text += text + "\n"

    # Send to OpenAI GPT
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # Can be replaced with "gpt-4" if available
        messages=[
            {
                "role": "system",
                "content": "Ты опытный врач. Расшифруй медицинский анализ, выдели важные отклонения и дай рекомендации."
            },
            {
                "role": "user",
                "content": pdf_text
            }
        ],
        temperature=0.5,
        max_tokens=1000
    )

    result = response.choices[0].message.content
    return JSONResponse(content={"analysis": result})
