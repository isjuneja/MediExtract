# Medical Prescription Processing System

## Overview

This is a comprehensive medical data processing application that extracts structured medical information from audio recordings or text input. The system uses speech recognition to convert audio files to text, applies noise reduction for better accuracy, and leverages Google's Gemini AI to extract comprehensive medical data including clinical notes, diagnosis, past history, medicines, tests prescribed, test results, and other relevant observations.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Technology**: Vanilla HTML/CSS/JavaScript served as static files
- **Design Pattern**: Single-page application with client-side form handling
- **Rationale**: Lightweight approach suitable for a focused prescription processing tool without the complexity of a full frontend framework

### Backend Architecture
- **Framework**: FastAPI (Python)
- **API Design**: RESTful endpoints for audio and text processing
- **Key Components**:
  - `AudioProcessor` - Handles audio format conversion, noise reduction, and speech-to-text
  - `MedicineExtractor` - Interfaces with Gemini AI for structured data extraction
  - Main FastAPI application - Orchestrates the processing pipeline

**Design Decisions**:
- **Modular separation**: Audio processing and AI extraction are separated into distinct classes for maintainability and testability
- **CORS enabled**: Allows cross-origin requests for flexible frontend deployment
- **Temporary file handling**: Uses Python's `tempfile` module for intermediate audio file storage during processing
- **Structured output**: Leverages Pydantic models for type-safe data validation and serialization

### Audio Processing Pipeline
1. **Format Conversion**: Converts various audio formats (MP3, M4A, OGG, FLAC, WebM) to WAV using PyDub
2. **Noise Reduction**: Applies `noisereduce` library with 80% noise reduction to improve transcription accuracy
3. **Speech Recognition**: Uses SpeechRecognition library to convert cleaned audio to text

**Rationale**: Multi-step preprocessing ensures optimal input quality for transcription, addressing real-world scenarios where medical audio recordings may have background noise.

### AI Integration
- **Model**: Google Gemini 2.5 Pro
- **Approach**: Structured extraction using comprehensive system prompts that define expected JSON schema
- **Output Format**: Pydantic models ensure type safety and consistent API responses
- **Fields Extracted**: 
  - Clinical notes
  - Diagnosis
  - Past medical history
  - Medicines (name, dosage, frequency, duration, instructions)
  - Tests prescribed (test name, purpose, instructions)
  - Test results (test name, result, unit, reference range, date)
  - Other observations

**Alternatives Considered**: 
- Local NLP models were considered but Gemini provides superior accuracy for medical terminology
- Pros: High accuracy, handles medical terminology well, structured output, comprehensive data extraction
- Cons: Requires API key, external dependency, potential latency

### Error Handling
- Comprehensive logging throughout the pipeline
- Try-except blocks around critical operations (file conversion, noise reduction, API calls)
- HTTPException raised for client errors with descriptive messages

## External Dependencies

### Third-Party Services
- **Google Gemini API**: Core AI service for extracting structured prescription data
  - Requires `GEMINI_API_KEY` environment variable
  - Model: gemini-2.5-pro
  - Used for natural language understanding and structured data extraction

### Python Libraries
- **FastAPI**: Web framework for building the REST API
- **PyDub**: Audio format conversion and manipulation
- **librosa**: Audio loading and processing
- **soundfile**: Audio file I/O operations
- **noisereduce**: Noise reduction algorithm implementation
- **SpeechRecognition**: Speech-to-text conversion
- **google-genai**: Official Google Generative AI Python SDK
- **Pydantic**: Data validation and serialization
- **NumPy**: Numerical operations for audio processing

### Audio Processing Dependencies
- The system supports multiple audio formats requiring appropriate codec support
- WAV format used as intermediate format for compatibility
- Sample rate preservation during processing for quality maintenance

### Development & Deployment
- **Environment Variables Required**:
  - `GEMINI_API_KEY`: Authentication for Google Gemini API
- **Static File Serving**: Frontend HTML/CSS/JS served via FastAPI StaticFiles middleware
- **CORS Configuration**: Wildcard origins enabled for development flexibility