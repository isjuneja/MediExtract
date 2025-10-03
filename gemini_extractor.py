import os
import json
import logging
from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import List, Optional

logger = logging.getLogger(__name__)

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

class Medicine(BaseModel):
    name: str
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    duration: Optional[str] = None
    instructions: Optional[str] = None

class PrescriptionData(BaseModel):
    medicines: List[Medicine]

class MedicineExtractor:
    def __init__(self):
        self.model = "gemini-2.5-pro"
    
    def extract_medicine_data(self, text: str) -> dict:
        """
        Extract structured medicine data from prescription text using Gemini Vision Pro
        """
        try:
            system_prompt = """You are a medical prescription analyzer. Extract medicine information from the given prescription text.
            
For each medicine mentioned, extract:
- name: The medicine name
- dosage: The dosage amount (e.g., "500mg", "10ml", "2 tablets")
- frequency: How often to take (e.g., "twice daily", "three times a day", "every 8 hours")
- duration: How long to take (e.g., "7 days", "2 weeks", "1 month")
- instructions: Any special instructions (e.g., "after meals", "before bedtime", "with water")

Return a JSON object with a 'medicines' array containing all extracted medicine data.
If any field is not mentioned, use null for that field.
Be precise and extract exact information from the text."""

            logger.info(f"Extracting medicine data from text: {text[:100]}...")
            
            response = client.models.generate_content(
                model=self.model,
                contents=[
                    types.Content(role="user", parts=[types.Part(text=text)])
                ],
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    response_schema=PrescriptionData,
                ),
            )
            
            raw_json = response.text
            logger.info(f"Gemini response: {raw_json}")
            
            if raw_json:
                data = json.loads(raw_json)
                return data
            else:
                raise ValueError("Empty response from Gemini model")
        
        except Exception as e:
            logger.error(f"Error extracting medicine data: {str(e)}", exc_info=True)
            return {
                "medicines": [],
                "error": str(e),
                "raw_text": text
            }
