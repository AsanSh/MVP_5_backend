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

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
logger.info("Environment variables loaded")

# Initialize Gemini
try:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyBovhxSsOp8OSGcHnZOcEsAlelK94YtEu8")
    genai.configure(api_key=GOOGLE_API_KEY)
    
    # Configure the model with safety settings
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 40
    }
    
    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE"
        }
    ]
    
    model = genai.GenerativeModel(model_name="gemini-pro",
                                generation_config=generation_config,
                                safety_settings=safety_settings)
    logger.info("Gemini AI initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Gemini AI: {str(e)}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    raise Exception("Failed to initialize Gemini AI")

# Initialize FastAPI
app = FastAPI(
    title="Medical Analysis API",
    description="API for analyzing medical PDF documents using Google Gemini AI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)
logger.info("FastAPI application initialized")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
logger.info("CORS middleware configured")

@app.get("/")
async def root():
    """
    Root endpoint that returns API information.
    """
    return {
        "message": "Medical Analysis API is running",
        "version": "1.0.0",
        "endpoints": {
            "/": "This information",
            "/docs": "API documentation (Swagger UI)",
            "/redoc": "API documentation (ReDoc)",
            "/analyze": "POST endpoint for analyzing medical PDFs"
        }
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the service is running.
    """
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/analyze")
async def analyze_pdf(file: UploadFile = File(...)):
    request_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"[{request_id}] New request received - Filename: {file.filename}")
    
    try:
        # Check if file is PDF
        if not file.filename.endswith('.pdf'):
            logger.error(f"[{request_id}] Invalid file type: {file.filename}")
            raise HTTPException(status_code=400, detail="File must be a PDF")

        # Read file contents
        try:
            logger.info(f"[{request_id}] Reading file contents")
            contents = await file.read()
            logger.info(f"[{request_id}] File contents read successfully. Size: {len(contents)} bytes")
        except Exception as e:
            logger.error(f"[{request_id}] Failed to read file contents: {str(e)}")
            logger.error(f"[{request_id}] Traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=400, detail="Failed to read file contents")

        pdf_text = ""

        # Extract text from PDF
        try:
            logger.info(f"[{request_id}] Starting PDF text extraction")
            with pdfplumber.open(io.BytesIO(contents)) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    logger.info(f"[{request_id}] Processing page {page_num}")
                    text = page.extract_text()
                    if text:
                        pdf_text += text + "\n"
                        logger.info(f"[{request_id}] Successfully extracted text from page {page_num}")
                    else:
                        logger.warning(f"[{request_id}] No text found on page {page_num}")
            
            logger.info(f"[{request_id}] PDF text extraction completed. Total text length: {len(pdf_text)}")
        except Exception as e:
            logger.error(f"[{request_id}] Error extracting text from PDF: {str(e)}")
            logger.error(f"[{request_id}] Traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=400, detail="Error extracting text from PDF")

        if not pdf_text:
            logger.error(f"[{request_id}] No text could be extracted from the PDF")
            raise HTTPException(status_code=400, detail="No text could be extracted from the PDF")

        # Send to Gemini
        try:
            logger.info(f"[{request_id}] Sending request to Gemini AI")
            
            prompt = """Ты опытный врач. Расшифруй медицинский анализ, выдели важные отклонения и дай рекомендации.

Анализ:
"""
            # Check text length and truncate if necessary
            if len(pdf_text) > 30000:  # Gemini's approximate token limit
                logger.warning(f"[{request_id}] Text too long ({len(pdf_text)} chars), truncating...")
                pdf_text = pdf_text[:30000] + "\n...[текст был сокращен из-за длины]..."

            # Try chat completion
            chat = model.start_chat(history=[])
            response = chat.send_message(prompt + pdf_text, timeout=60)  # Add timeout
            
            if not response.text:
                raise ValueError("Empty response received from Gemini")
                
            logger.info(f"[{request_id}] Successfully received response from Gemini AI")
            return JSONResponse(content={"analysis": response.text})

        except Exception as e:
            error_msg = str(e)
            logger.error(f"[{request_id}] Error in Gemini AI call: {error_msg}")
            logger.error(f"[{request_id}] Traceback: {traceback.format_exc()}")
            raise HTTPException(
                status_code=500,
                detail=f"Error analyzing medical data: {error_msg}"
            )

    except HTTPException as he:
        logger.error(f"[{request_id}] HTTPException: {str(he.detail)}")
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error: {str(e)}")
        logger.error(f"[{request_id}] Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")
