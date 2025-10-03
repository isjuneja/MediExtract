import os
import tempfile
import logging
import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr
from pydub import AudioSegment
import speech_recognition as sr

logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self):
        self.recognizer = sr.Recognizer()
    
    def convert_to_wav(self, audio_path: str) -> str:
        """Convert audio file to WAV format if needed"""
        try:
            file_extension = os.path.splitext(audio_path)[1].lower()
            
            if file_extension == '.wav':
                return audio_path
            
            logger.info(f"Converting {file_extension} to WAV format")
            audio = AudioSegment.from_file(audio_path)
            
            wav_path = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
            audio.export(wav_path, format='wav')
            
            return wav_path
        except Exception as e:
            logger.error(f"Error converting audio to WAV: {str(e)}")
            raise
    
    def reduce_noise(self, audio_path: str) -> str:
        """Apply noise reduction to audio file"""
        try:
            logger.info("Loading audio for noise reduction")
            audio_data, sample_rate = librosa.load(audio_path, sr=None)
            
            logger.info("Applying noise reduction")
            reduced_noise = nr.reduce_noise(y=audio_data, sr=sample_rate, prop_decrease=0.8)
            
            cleaned_path = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
            sf.write(cleaned_path, reduced_noise, sample_rate)
            
            logger.info(f"Noise reduction complete, saved to {cleaned_path}")
            return cleaned_path
        except Exception as e:
            logger.error(f"Error reducing noise: {str(e)}")
            return audio_path
    
    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio to text using speech recognition"""
        try:
            with sr.AudioFile(audio_path) as source:
                logger.info("Reading audio file")
                audio_data = self.recognizer.record(source)
                
                logger.info("Transcribing audio using Google Speech Recognition")
                text = self.recognizer.recognize_google(audio_data)
                
                return text
        except sr.UnknownValueError:
            logger.error("Speech recognition could not understand audio")
            raise Exception("Could not understand the audio. Please ensure it contains clear speech.")
        except sr.RequestError as e:
            logger.error(f"Could not request results from speech recognition service: {e}")
            raise Exception(f"Speech recognition service error: {e}")
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            raise
    
    def process_audio(self, audio_path: str) -> str:
        """
        Complete audio processing pipeline:
        1. Convert to WAV if needed
        2. Apply noise reduction
        3. Transcribe to text
        """
        wav_path = None
        cleaned_path = None
        
        try:
            wav_path = self.convert_to_wav(audio_path)
            
            cleaned_path = self.reduce_noise(wav_path)
            
            text = self.transcribe_audio(cleaned_path)
            
            return text
        finally:
            if wav_path and wav_path != audio_path and os.path.exists(wav_path):
                os.unlink(wav_path)
            if cleaned_path and os.path.exists(cleaned_path):
                os.unlink(cleaned_path)
