"""
Speech Recognition Module
Handles voice input and wake word detection - SIMPLE VERSION
"""

import logging
import threading
import queue
import time
from typing import Optional, Callable

logger = logging.getLogger(__name__)

class SpeechRecognizer:
    def __init__(self, config: dict, on_command: Callable):
        """
        Initialize speech recognizer - SIMPLE VERSION (no Picovoice needed)
        
        Args:
            config: Speech configuration dictionary
            on_command: Callback function for recognized commands
        """
        self.config = config
        self.on_command = on_command
        self.listening = False
        self.stop_event = threading.Event()
        
        # Simple wake word configuration
        self.wake_word = config.get('wake_word', 'jarvis').lower()
        self.wake_words = [self.wake_word, f"hey {self.wake_word}", f"okay {self.wake_word}"]
        
        # Try to import speech recognition
        self.speech_lib_available = False
        self._init_speech_libs()
        
        if self.speech_lib_available:
            logger.info(f"✅ Speech recognizer initialized with wake word: '{self.wake_word}'")
        else:
            logger.warning("⚠️ Speech recognition not available")
    
    def _init_speech_libs(self):
        """Initialize speech recognition libraries"""
        try:
            import speech_recognition as sr
            self.speech_lib = sr
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            self.speech_lib_available = True
        except ImportError:
            self.speech_lib_available = False
        except Exception:
            self.speech_lib_available = False
    
    def start_listening(self):
        """Start listening for voice commands"""
        if not self.speech_lib_available:
            print("❌ Speech recognition requires: pip install speechrecognition pyaudio")
            return
        
        if self.listening:
            print("⚠️ Already listening")
            return
        
        self.listening = True
        self.stop_event.clear()
        
