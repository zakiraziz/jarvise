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
                # Start listening thread
        self.listening_thread = threading.Thread(
            target=self._listen_loop,
            daemon=True
        )
        self.listening_thread.start()
        
        print(f"✅ Listening for voice commands... Say '{self.wake_word}' to activate")
    
    def stop_listening(self):
        """Stop listening for voice commands"""
        self.listening = False
        self.stop_event.set()
        print("🔇 Voice listening stopped")
    
    def _listen_loop(self):
        """Main listening loop"""
        try:
            # Adjust for ambient noise
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                
                while self.listening and not self.stop_event.is_set():
                    try:
                        print("🎤 Listening... (say wake word)", end='\r')
                        audio = self.recognizer.listen(
                            source, 
                            timeout=3,
                            phrase_time_limit=5
                        )
                        
                        # Recognize speech
                        try:
                            text = self.recognizer.recognize_google(audio).lower()
                            
                            # Check for wake word
                            for wake_word in self.wake_words:
                                if wake_word in text:
                                    print(f"\n✅ Wake word detected: {wake_word}")
                                    
                                    # Extract command after wake word
                                    idx = text.find(wake_word) + len(wake_word)
                                    command = text[idx:].strip()
                                    
                                    if command:
                                        print(f"🤖 Processing: {command}")
                                        self.on_command(command)
                                    else:
                                        print("❓ What can I help you with?")
                                    
                                    break
                            
                        except self.speech_lib.UnknownValueError:
                            continue
                        except self.speech_lib.RequestError:
                            print("🌐 Network error")
                            time.sleep(2)
                        
                    except self.speech_lib.WaitTimeoutError:
                        continue
                    except Exception:
                        time.sleep(1)
        
        except Exception as e:
            print(f"❌ Voice error: {e}")
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop_listening()

