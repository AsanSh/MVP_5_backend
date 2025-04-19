from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pdfplumber
import google.generativeai as genai
import os
from dotenv import load_dotenv
import io
import logging
import traceback
from datetime import datetime

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load environment
load_dotenv()
logger.info("Environment variables loaded")

# Configure Gemini
try:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")

    genai.configure(api_key=GOOGLE_API_KEY)

    # ✅ Явно указываем стабильную версию API
    model = genai.GenerativeModel("gemini-pro", model_name="models/gemini-pro")
    logger.info("Gemini AI initialized successfully")

except Exception as e:
    logger.error(f"Failed to initialize Gemini AI: {str(e)}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    raise Exception("Failed to initialize Gemini AI")

# FastAPI
app = FastAPI(
    title="Medical Analysis API",
    description="API for analyzing medical PDF documents using Google Gemini AI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "Medical Analysis API is running",
        "version": "1.0.0",
        "endpoints": {
            "/": "This information",
            "/docs": "Swagger UI",
            "/redoc": "ReDoc UI",
            "/analyze": "POST endpoint for medical PDF analysis",
            "/test_gemini": "POST endpoint for testing Gemini with text"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/analyze")
async def analyze_pdf(file: UploadFile = File(...)):
    request_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"[{request_id}] New request - Filename: {file.filename}")

    try:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="File must be a PDF")

        contents = await file.read()
        logger.info(f"[{request_id}] PDF size: {len(contents)} bytes")

        pdf_text = ""
        with pdfplumber.open(io.BytesIO(contents)) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text:
                    pdf_text += text + "\n"
                    logger.info(f"[{request_id}] Extracted text from page {i}")
                else:
                    logger.warning(f"[{request_id}] Page {i} has no extractable text")

        if not pdf_text.strip():
            raise HTTPException(status_code=400, detail="No readable text in PDF")

        prompt = (
            "Ты опытный врач. Проанализируй медицинский анализ, выдели важные отклонения и дай рекомендации.\n\n"
            + pdf_text
        )

        if len(prompt) > 30000:
            logger.warning(f"[{request_id}] Prompt too long, truncating")
            prompt = prompt[:30000] + "\n...[текст был сокращён]..."

        chat = model.start_chat()
        response = chat.send_message(prompt)

        if not response.text:
            raise ValueError("Empty response from Gemini")

        logger.info(f"[{request_id}] Gemini analysis completed")
        return JSONResponse(content={"analysis": response.text})

    except Exception as e:
        logger.error(f"[{request_id}] Error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error analyzing medical data: {str(e)}")

@app.post("/test_gemini")
async def test_gemini(text: str = "Привет, Gemini!"):
    """
    Эндпоинт для проверки базовой работы с Gemini API и текстовыми запросами.
    Отправляет простой текстовый запрос и возвращает ответ.
    """
    logger.info(f"Testing Gemini with text: '{text}'")
    try:
        # ✅ Явно указываем стабильную версию API и имя модели
        model = genai.GenerativeModel("gemini-pro", model_name="models/gemini-pro")
        response = model.generate_content(text)
        if response and response.text:
            logger.info(f"Gemini test successful, response: '{response.text[:50]}...'")
            return JSONResponse(content={"response": response.text})
        else:
            logger.warning("Gemini test returned an empty response.")
            raise HTTPException(status_code=500, detail="Empty response from Gemini")
    except Exception as e:
        logger.error(f"Error during Gemini test: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error testing Gemini: {str(e)}")