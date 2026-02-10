"""
Speech Recognition Module
Handles voice input and wake word detection - ENHANCED VERSION
"""

import logging
import threading
import queue
import time
import json
import os
from typing import Optional, Callable, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

class RecognitionMode(Enum):
    """Different speech recognition modes"""
    WAKE_WORD_ONLY = "wake_word_only"
    CONTINUOUS = "continuous"
    PRESS_TO_TALK = "press_to_talk"

@dataclass
class SpeechConfig:
    """Configuration for speech recognition"""
    wake_word: str = "jarvis"
    sensitivity: float = 0.5
    energy_threshold: int = 300
    dynamic_energy_threshold: bool = True
    pause_threshold: float = 0.8
    phrase_time_limit: int = 5
    recognition_mode: RecognitionMode = RecognitionMode.WAKE_WORD_ONLY
    language: str = "en-US"
    save_audio_logs: bool = False
    audio_log_path: str = "audio_logs"
    command_history_size: int = 100

class SpeechRecognizer:
    def __init__(self, config: SpeechConfig, on_command: Callable):
        """
        Initialize speech recognizer - ENHANCED VERSION
        
        Args:
            config: Speech configuration object
            on_command: Callback function for recognized commands
        """
        self.config = config
        self.on_command = on_command
        self.listening = False
        self.stop_event = threading.Event()
        self.command_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        self.command_history = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize wake words
        self.wake_words = self._generate_wake_words(config.wake_word)
        self.active_wake_word = config.wake_word
        
        # Performance tracking
        self.stats = {
            "wake_word_detections": 0,
            "commands_processed": 0,
            "recognition_errors": 0,
            "audio_samples_processed": 0,
            "start_time": time.time()
        }
        
        # Try to import speech recognition
        self.speech_lib_available = False
        self.audio_available = False
        self._init_speech_libs()
        
        if self.speech_lib_available:
            logger.info(f"✅ Speech recognizer initialized with wake word: '{config.wake_word}'")
            logger.info(f"   Mode: {config.recognition_mode.value}")
            logger.info(f"   Language: {config.language}")
            
            # Create audio log directory if needed
            if config.save_audio_logs:
                os.makedirs(config.audio_log_path, exist_ok=True)
        else:
            logger.warning("⚠️ Speech recognition not available")
    
    def _generate_wake_words(self, base_word: str) -> List[str]:
        """Generate variations of wake words"""
        base_word = base_word.lower()
        return [
            base_word,
            f"hey {base_word}",
            f"okay {base_word}",
            f"hello {base_word}",
            f"hi {base_word}",
            f"yo {base_word}",
            f"listen {base_word}"
        ]
    
    def _init_speech_libs(self):
        """Initialize speech recognition libraries"""
        try:
            import speech_recognition as sr
            import pyaudio
            
            self.speech_lib = sr
            self.recognizer = sr.Recognizer()
            
            # Configure recognizer settings
            self.recognizer.energy_threshold = self.config.energy_threshold
            self.recognizer.dynamic_energy_threshold = self.config.dynamic_energy_threshold
            self.recognizer.pause_threshold = self.config.pause_threshold
            
            # Get microphone
            try:
                self.microphone = sr.Microphone()
                self.audio_available = True
                self.speech_lib_available = True
            except Exception as e:
                logger.error(f"Microphone initialization failed: {e}")
                self.speech_lib_available = False
                
        except ImportError as e:
            logger.error(f"Required libraries not available: {e}")
            self.speech_lib_available = False
        except Exception as e:
            logger.error(f"Speech library initialization error: {e}")
            self.speech_lib_available = False
    
    def start_listening(self):
        """Start listening for voice commands"""
        if not self.speech_lib_available:
            logger.error("❌ Speech recognition requires: pip install speechrecognition pyaudio")
            print("❌ Please install required packages: pip install speechrecognition pyaudio")
            return
        
        if self.listening:
            logger.warning("⚠️ Already listening")
            return
        
        self.listening = True
        self.stop_event.clear()
        
        # Start multiple threads for different tasks
        self.listening_thread = threading.Thread(
            target=self._listen_loop,
            daemon=True,
            name="SpeechListener"
        )
        
        self.processing_thread = threading.Thread(
            target=self._process_commands,
            daemon=True,
            name="CommandProcessor"
        )
        
        self.listening_thread.start()
        self.processing_thread.start()
        
        logger.info(f"✅ Listening for voice commands... Say '{self.active_wake_word}' to activate")
        print(f"🎤 Listening... (Mode: {self.config.recognition_mode.value})")
        print(f"   Say: {', '.join(self.wake_words[:3])}")
    
    def stop_listening(self):
        """Stop listening for voice commands"""
        self.listening = False
        self.stop_event.set()
        logger.info("🔇 Voice listening stopped")
        print("🔇 Voice listening stopped")
    
    def _listen_loop(self):
        """Main listening loop with different recognition modes"""
        try:
            with self.microphone as source:
                logger.info("Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=2)
                
                if self.config.dynamic_energy_threshold:
                    self.recognizer.dynamic_energy_adjustment_damping = 0.15
                    self.recognizer.dynamic_energy_ratio = 1.5
                
                while self.listening and not self.stop_event.is_set():
                    try:
                        # Different listening modes
                        if self.config.recognition_mode == RecognitionMode.WAKE_WORD_ONLY:
                            self._wake_word_mode(source)
                        elif self.config.recognition_mode == RecognitionMode.CONTINUOUS:
                            self._continuous_mode(source)
                        elif self.config.recognition_mode == RecognitionMode.PRESS_TO_TALK:
                            self._press_to_talk_mode(source)
                        
                        time.sleep(0.1)  # Prevent tight loop
                        
                    except KeyboardInterrupt:
                        break
                    except Exception as e:
                        logger.error(f"Error in listen loop: {e}")
                        time.sleep(1)
        
        except Exception as e:
            logger.error(f"Microphone error: {e}")
            print(f"❌ Microphone error: {e}")
    
    def _wake_word_mode(self, source):
        """Listen for wake word only"""
        try:
            audio = self.recognizer.listen(
                source,
                timeout=3,
                phrase_time_limit=self.config.phrase_time_limit,
                snowboy_configuration=None
            )
            
            self._process_audio(audio)
            
        except self.speech_lib.WaitTimeoutError:
            pass  # Normal timeout, continue listening
        except Exception as e:
            logger.debug(f"Audio capture error: {e}")
    
    def _continuous_mode(self, source):
        """Continuous listening mode"""
        try:
            print("🔊 Continuous listening... Speak now", end='\r')
            audio = self.recognizer.listen(
                source,
                phrase_time_limit=self.config.phrase_time_limit
            )
            
            # Process all speech in continuous mode
            text = self._recognize_speech(audio)
            if text:
                # Check if any wake word is present
                wake_word_detected = False
                for wake_word in self.wake_words:
                    if wake_word in text.lower():
                        wake_word_detected = True
                        idx = text.lower().find(wake_word) + len(wake_word)
                        command = text[idx:].strip()
                        break
                
                if wake_word_detected:
                    self.command_queue.put(("wake_word", command if command else "no_command"))
                elif text.strip():  # Direct command without wake word
                    self.command_queue.put(("direct", text.strip()))
            
        except Exception as e:
            logger.debug(f"Continuous mode error: {e}")
    
    def _press_to_talk_mode(self, source):
        """Press-to-talk simulation (enter to start recording)"""
        print("⏺️  Press Enter to start recording, then speak...")
        input()  # Wait for Enter key
        
        print("🎤 Recording... (speak now)")
        try:
            audio = self.recognizer.listen(
                source,
                phrase_time_limit=10  # Longer for press-to-talk
            )
            
            text = self._recognize_speech(audio)
            if text:
                self.command_queue.put(("direct", text.strip()))
                print(f"📝 Recognized: {text}")
        
        except Exception as e:
            logger.error(f"Press-to-talk error: {e}")
    
    def _process_audio(self, audio):
        """Process captured audio"""
        try:
            # Save audio if configured
            if self.config.save_audio_logs:
                self._save_audio_sample(audio)
            
            text = self._recognize_speech(audio)
            
            if text:
                self.stats["audio_samples_processed"] += 1
                
                # Check for wake words
                for wake_word in self.wake_words:
                    if wake_word in text.lower():
                        self.stats["wake_word_detections"] += 1
                        print(f"\n✅ Wake word detected: {wake_word}")
                        
                        # Extract command
                        idx = text.lower().find(wake_word) + len(wake_word)
                        command = text[idx:].strip()
                        
                        if command:
                            print(f"🤖 Command: {command}")
                            self.command_queue.put((wake_word, command))
                        else:
                            print("❓ What can I help you with?")
                            self.command_queue.put((wake_word, "wake_only"))
                        
                        return True
                
                # If no wake word but continuous mode, queue as direct speech
                if self.config.recognition_mode == RecognitionMode.CONTINUOUS and text.strip():
                    self.command_queue.put(("continuous_speech", text.strip()))
            
            return False
            
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            self.stats["recognition_errors"] += 1
            return False
    
    def _recognize_speech(self, audio) -> Optional[str]:
        """Convert audio to text using available services"""
        try:
            # Try multiple recognition engines in order of preference
            try:
                # Primary: Google Web Speech API
                text = self.recognizer.recognize_google(
                    audio,
                    language=self.config.language,
                    show_all=False
                )
                return text
                
            except self.speech_lib.RequestError:
                # Fallback: Sphinx (offline)
                try:
                    text = self.recognizer.recognize_sphinx(audio)
                    return text
                except:
                    pass
                    
            except self.speech_lib.UnknownValueError:
                pass  # Speech not understood
                
        except Exception as e:
            logger.debug(f"Speech recognition failed: {e}")
        
        return None
    
    def _process_commands(self):
        """Process commands from queue"""
        while self.listening or not self.command_queue.empty():
            try:
                wake_word, command = self.command_queue.get(timeout=1)
                
                # Add to history
                self._add_to_history(wake_word, command)
                
                # Process command
                if command != "wake_only" and command != "no_command":
                    self.stats["commands_processed"] += 1
                    self.on_command(command)
                
                self.command_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Command processing error: {e}")
    
    def _save_audio_sample(self, audio):
        """Save audio sample to file for debugging"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"{self.config.audio_log_path}/audio_{self.session_id}_{timestamp}.wav"
            
            # Save as WAV file
            with open(filename, "wb") as f:
                f.write(audio.get_wav_data())
            
            logger.debug(f"Audio saved: {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
    
    def _add_to_history(self, wake_word: str, command: str):
        """Add command to history"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "wake_word": wake_word,
            "command": command,
            "mode": self.config.recognition_mode.value
        }
        
        self.command_history.append(entry)
        
  
