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

class TestPrescribed(BaseModel):
    test_name: str
    purpose: Optional[str] = None
    instructions: Optional[str] = None

class TestResult(BaseModel):
    test_name: str
    result: str
    unit: Optional[str] = None
    reference_range: Optional[str] = None
    date: Optional[str] = None

class MedicalData(BaseModel):
    clinical_notes: Optional[str] = None
    diagnosis: Optional[str] = None
    past_history: Optional[str] = None
    medicines: List[Medicine] = []
    tests_prescribed: List[TestPrescribed] = []
    test_results: List[TestResult] = []
    other_observations: Optional[str] = None

class MedicineExtractor:
    def __init__(self):
        self.model = "gemini-2.5-pro"
    
    def extract_medicine_data(self, text: str) -> dict:
        """
        Extract comprehensive medical data from prescription text using Gemini Vision Pro
        """
        try:
            system_prompt = """You are a comprehensive medical data analyzer. Extract all relevant medical information from the given text.

Extract the following information:

1. **Clinical Notes**: Any clinical observations, patient complaints, or physician notes
2. **Diagnosis**: The diagnosed condition(s) or disease(s)
3. **Past History**: Patient's medical history, previous conditions, or past treatments mentioned
4. **Medicines**: For each medicine extract:
   - name: Medicine name
   - dosage: Dosage amount (e.g., "500mg", "10ml", "2 tablets")
   - frequency: How often to take (e.g., "twice daily", "three times a day")
   - duration: How long to take (e.g., "7 days", "2 weeks")
   - instructions: Special instructions (e.g., "after meals", "before bedtime")
5. **Tests Prescribed**: Laboratory tests or diagnostic procedures prescribed
   - test_name: Name of the test
   - purpose: Why the test is needed (if mentioned)
   - instructions: Any preparation or special instructions
6. **Test Results**: Any test results mentioned in the text
   - test_name: Name of the test
   - result: The test result value
   - unit: Unit of measurement (if applicable)
   - reference_range: Normal range (if mentioned)
   - date: Date of the test (if mentioned)
7. **Other Observations**: Any other relevant medical information not covered above

If any field is not mentioned in the text, use null or an empty array as appropriate.
Be precise and extract exact information from the text."""

            logger.info(f"Extracting comprehensive medical data from text: {text[:100]}...")
            
            response = client.models.generate_content(
                model=self.model,
                contents=[
                    types.Content(role="user", parts=[types.Part(text=text)])
                ],
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    response_schema=MedicalData,
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
            logger.error(f"Error extracting medical data: {str(e)}", exc_info=True)
            return {
                "clinical_notes": None,
                "diagnosis": None,
                "past_history": None,
                "medicines": [],
                "tests_prescribed": [],
                "test_results": [],
                "other_observations": None,
                "error": str(e),
                "raw_text": text
            }
