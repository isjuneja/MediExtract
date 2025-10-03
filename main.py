import os
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import json
import logging

from audio_processor import AudioProcessor
from gemini_extractor import MedicineExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Medical Prescription Processing System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

audio_processor = AudioProcessor()
medicine_extractor = MedicineExtractor()

class MedicineData(BaseModel):
    medicines: List[dict]
    raw_text: str

class TextInput(BaseModel):
    text: str

@app.post("/api/process-audio")
async def process_audio(file: UploadFile = File(...)):
    """
    Process audio file: convert to text with noise reduction, then extract medicine data
    """
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        file_extension = os.path.splitext(file.filename)[1].lower()
        allowed_extensions = ['.mp3', '.wav', '.m4a', '.ogg', '.flac', '.webm', '.aac', '.wma', '.opus', '.aiff', '.3gp', '.amr']
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format. Allowed formats: {', '.join(allowed_extensions)}"
            )
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_audio:
            content = await file.read()
            temp_audio.write(content)
            temp_audio_path = temp_audio.name
        
        try:
            logger.info(f"Processing audio file: {file.filename}")
            transcribed_text = audio_processor.process_audio(temp_audio_path)
            
            if not transcribed_text or transcribed_text.strip() == "":
                raise HTTPException(
                    status_code=422, 
                    detail="Could not transcribe audio. Please ensure the audio is clear and contains speech."
                )
            
            logger.info(f"Transcribed text: {transcribed_text}")
            
            medicine_data = medicine_extractor.extract_medicine_data(transcribed_text)
            
            return JSONResponse(content={
                "success": True,
                "transcribed_text": transcribed_text,
                "medicine_data": medicine_data
            })
            
        finally:
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

@app.post("/api/process-text")
async def process_text(input_data: TextInput):
    """
    Process text input directly to extract medicine data
    """
    try:
        if not input_data.text or input_data.text.strip() == "":
            raise HTTPException(status_code=400, detail="Text input cannot be empty")
        
        logger.info(f"Processing text input: {input_data.text}")
        
        medicine_data = medicine_extractor.extract_medicine_data(input_data.text)
        
        return JSONResponse(content={
            "success": True,
            "input_text": input_data.text,
            "medicine_data": medicine_data
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing text: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Medical Prescription Processing System"}

app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
