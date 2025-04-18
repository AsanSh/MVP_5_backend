from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pdfplumber
from openai import OpenAI
import os
from dotenv import load_dotenv
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/analyze")
async def analyze_pdf(file: UploadFile = File(...)):
    try:
        # Check if file is PDF
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF")

        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="OpenAI API key not configured")
        
        client = OpenAI(
            api_key=api_key,
            timeout=60.0  # Increase timeout to 60 seconds
        )

        contents = await file.read()
        pdf_text = ""

        # Extract text from PDF
        try:
            with pdfplumber.open(io.BytesIO(contents)) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:  # Check for empty page
                        pdf_text += text + "\n"
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise HTTPException(status_code=400, detail="Error extracting text from PDF")

        if not pdf_text:
            raise HTTPException(status_code=400, detail="No text could be extracted from the PDF")

        # Send to OpenAI GPT
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
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
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {str(e)}")
            raise HTTPException(status_code=500, detail="Error processing the analysis")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")
