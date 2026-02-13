"""
Speech Handler Module for Jarvis
Handles text-to-speech functionality using pyttsx3 with enhanced features
"""

import os
import sys
import logging
import threading
import re
import time
from typing import Optional, List, Dict, Any
from queue import Queue
from enum import Enum

logger = logging.getLogger(__name__)

# Try to import pyttsx3, provide fallback if not available
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    logger.warning("pyttsx3 package not installed. Speech features will be limited.")

# Optional imports for enhanced functionality
try:
    import pywhatkit as kit
    KIT_AVAILABLE = True
except ImportError:
    KIT_AVAILABLE = False

try:
    from gtts import gTTS
    import pygame
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False


class SpeechPriority(Enum):
    """Priority levels for speech queue"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class SpeechHandler:
    """Enhanced text-to-speech handler for Jarvis with queue system and multiple TTS engines"""
    
    def __init__(self, config: dict):
        """Initialize speech handler with enhanced features"""
        self.config = config
        self.speech_config = config.get('speech', {})
        
        # Speech settings
        self.enabled = self.speech_config.get('enabled', True)
        self.voice_rate = self.speech_config.get('voice_rate', 150)
        self.voice_volume = self.speech_config.get('voice_volume', 1.0)
        self.voice_gender = self.speech_config.get('voice_gender', 'male')
        self.voice_id = self.speech_config.get('voice_id', None)
        self.tts_engine = self.speech_config.get('tts_engine', 'pyttsx3')  # pyttsx3, gtts, pywhatkit
        
        # Enhanced features settings
        self.speech_queue_enabled = self.speech_config.get('speech_queue_enabled', True)
        self.max_queue_size = self.speech_config.get('max_queue_size', 50)
        self.default_priority = SpeechPriority[self.speech_config.get('default_priority', 'NORMAL')]
        self.interruptible = self.speech_config.get('interruptible', True)
        self.cache_enabled = self.speech_config.get('cache_enabled', False)
        self.cache_dir = self.speech_config.get('cache_dir', 'cache/speech')
        
        # Engine instance and queue
        self.engine = None
        self.speech_queue = Queue()
        self.currently_speaking = False
        self.queue_thread = None
        self.queue_running = False
        self.speech_lock = threading.Lock()
        self.speech_cache = {}
        
        # Initialize components
        self._init_cache_directory()
        self._init_engine()
        self._start_queue_processor()
        
        logger.info(f"Speech handler initialized - Enabled: {self.enabled}, Engine: {self.tts_engine}")
    
    def _init_cache_directory(self):
        """Initialize speech cache directory"""
        if self.cache_enabled:
            try:
                os.makedirs(self.cache_dir, exist_ok=True)
                logger.info(f"Speech cache directory created at {self.cache_dir}")
            except Exception as e:
                logger.error(f"Failed to create cache directory: {e}")
                self.cache_enabled = False
    
    def _init_engine(self):
        """Initialize the text-to-speech engine based on configuration"""
        if not self.enabled:
            logger.info("Speech is disabled in configuration")
            return
        
        # Try to initialize selected engine
        if self.tts_engine == 'pyttsx3' and PYTTSX3_AVAILABLE:
            self._init_pyttsx3_engine()
        elif self.tts_engine == 'gtts' and GTTS_AVAILABLE:
            self._init_gtts_engine()
        elif self.tts_engine == 'pywhatkit' and KIT_AVAILABLE:
            self._init_whatkit_engine()
        else:
            # Fallback to available engine
            if PYTTSX3_AVAILABLE:
                self.tts_engine = 'pyttsx3'
                self._init_pyttsx3_engine()
            elif GTTS_AVAILABLE:
                self.tts_engine = 'gtts'
                self._init_gtts_engine()
            elif KIT_AVAILABLE:
                self.tts_engine = 'pywhatkit'
                self._init_whatkit_engine()
            else:
                logger.warning("No TTS engine available")
                self.enabled = False
    
    def _init_pyttsx3_engine(self):
        """Initialize pyttsx3 engine"""
        try:
            self.engine = pyttsx3.init()
            self._configure_pyttsx3_engine()
            logger.info("pyttsx3 TTS engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize pyttsx3 TTS engine: {e}")
            self.engine = None
    
    def _configure_pyttsx3_engine(self):
        """Configure pyttsx3 engine with settings"""
        if self.engine is None:
            return
        
        try:
            # Set speech rate
            self.engine.setProperty('rate', self.voice_rate)
            
            # Set volume (0.0 to 1.0)
            volume = min(1.0, max(0.0, self.voice_volume))
            self.engine.setProperty('volume', volume)
            
            # Set voice if specified
            if self.voice_id:
                self.engine.setProperty('voice', self.voice_id)
            else:
                self._select_voice_by_gender()
            
        except Exception as e:
            logger.error(f"Error configuring pyttsx3 engine: {e}")
    
    def _select_voice_by_gender(self):
        """Select voice based on gender preference"""
        try:
            voices = self.engine.getProperty('voices')
            
            if voices:
                selected_voice = None
                
                for voice in voices:
                    voice_name = voice.name.lower() if voice.name else ''
                    voice_id = voice.id.lower() if voice.id else ''
                    
                    if self.voice_gender == 'male':
                        if 'male' in voice_name or 'male' in voice_id or 'david' in voice_name:
                            selected_voice = voice
                            break
                    else:
                        if 'female' in voice_name or 'female' in voice_id or 'zira' in voice_name:
                            selected_voice = voice
                            break
                
                # If no gender-specific voice found, use first available
                if selected_voice is None:
                    selected_voice = voices[0]
                
                self.engine.setProperty('voice', selected_voice.id)
                self.voice_id = selected_voice.id
                logger.info(f"Selected voice: {selected_voice.name}")
                
        except Exception as e:
            logger.error(f"Error selecting voice: {e}")
    
    def _init_gtts_engine(self):
        """Initialize gTTS engine"""
        # gTTS doesn't need a persistent engine instance
        self.engine = "gtts"
        logger.info("gTTS engine initialized")
    
    def _init_whatkit_engine(self):
        """Initialize pywhatkit engine"""
        self.engine = "pywhatkit"
        logger.info("pywhatkit engine initialized")
    
    def _start_queue_processor(self):
        """Start the speech queue processing thread"""
        if not self.speech_queue_enabled:
            return
        
        self.queue_running = True
        self.queue_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.queue_thread.start()
        logger.info("Speech queue processor started")
    
    def _process_queue(self):
        """Process speech queue in background"""
        while self.queue_running:
            try:
                if not self.speech_queue.empty() and not self.currently_speaking:
                    item = self.speech_queue.get(timeout=1)
                    if item:
                        text, priority, block, callback = item
                        self._speak_internal(text, block, callback)
                else:
                    time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error processing speech queue: {e}")
                time.sleep(1)
    
    def speak(self, text: str, priority: SpeechPriority = None, 
              block: bool = False, callback: Optional[callable] = None) -> bool:
        """Enhanced speak method with queue support"""
        if not self.enabled:
            logger.debug("Speech is disabled")
            return False
        
        if not text or not text.strip():
            logger.debug("Empty text provided for speech")
            return False
        
        # Set default priority
        if priority is None:
            priority = self.default_priority
        
        # Clean and prepare text
        text = self._prepare_text(text)
        
        # Check cache
        if self.cache_enabled and text in self.speech_cache:
            logger.info(f"Using cached speech for: {text[:50]}...")
            return self._play_cached(text)
        
        # Add to queue if enabled and not high priority/blocking
        if self.speech_queue_enabled and priority != SpeechPriority.CRITICAL and not block:
            # Check queue size
            if self.speech_queue.qsize() >= self.max_queue_size:
                logger.warning("Speech queue full, dropping message")
                return False
            
            # For high priority, clear queue first
            if priority == SpeechPriority.HIGH and self.interruptible:
                self.clear_queue()
                self.stop()
            
            self.speech_queue.put((text, priority, block, callback))
            logger.info(f"Added to speech queue: {text[:50]}... (Priority: {priority.name})")
            return True
        else:
            # Speak immediately
            return self._speak_internal(text, block, callback)
    
    def _speak_internal(self, text: str, block: bool, callback: Optional[callable] = None) -> bool:
        """Internal method to handle actual speech"""
        with self.speech_lock:
            self.currently_speaking = True
        
        try:
            success = False
            
            if self.tts_engine == 'pyttsx3' and PYTTSX3_AVAILABLE:
                success = self._speak_pyttsx3(text, block)
            elif self.tts_engine == 'gtts' and GTTS_AVAILABLE:
                success = self._speak_gtts(text, block)
            elif self.tts_engine == 'pywhatkit' and KIT_AVAILABLE:
                success = self._speak_whatkit(text, block)
            
            # Cache the speech if enabled
            if success and self.cache_enabled:
                self._cache_speech(text)
            
            # Execute callback if provided
            if callback and callable(callback):
                try:
                    callback(success)
                except Exception as e:
                    logger.error(f"Error in speech callback: {e}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error in text-to-speech: {e}")
            return False
        finally:
            with self.speech_lock:
                self.currently_speaking = False
    
    def _speak_pyttsx3(self, text: str, block: bool) -> bool:
        """Speak using pyttsx3"""
        try:
            if block:
                self.engine.say(text)
                self.engine.runAndWait()
            else:
                thread = threading.Thread(target=self._speak_pyttsx3_async, args=(text,))
                thread.daemon = True
                thread.start()
            
            logger.info(f"Spoke text (pyttsx3): {text[:50]}...")
            return True
        except Exception as e:
            logger.error(f"Error in pyttsx3 speech: {e}")
            return False
    
    def _speak_pyttsx3_async(self, text: str):
        """Async pyttsx3 speech"""
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            logger.error(f"Async pyttsx3 speech error: {e}")
    
    def _speak_gtts(self, text: str, block: bool) -> bool:
        """Speak using gTTS"""
        try:
            # Create temporary file
            temp_file = os.path.join(self.cache_dir if self.cache_enabled else '/tmp', 
                                     f"speech_{int(time.time())}.mp3")
            
            # Generate speech
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(temp_file)
            
            # Play audio
            pygame.mixer.init()
            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.play()
            
            if block:
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
            
            # Cleanup
            pygame.mixer.quit()
            if not self.cache_enabled:
                os.remove(temp_file)
            
            logger.info(f"Spoke text (gTTS): {text[:50]}...")
            return True
        except Exception as e:
            logger.error(f"Error in gTTS speech: {e}")
            return False
    
    def _speak_whatkit(self, text: str, block: bool) -> bool:
        """Speak using pywhatkit"""
        try:
            if block:
                kit.text_to_speech(text, 'speech.mp3', True)
            else:
                thread = threading.Thread(target=kit.text_to_speech, args=(text, 'speech.mp3', True))
                thread.daemon = True
                thread.start()
            
            logger.info(f"Spoke text (pywhatkit): {text[:50]}...")
            return True
        except Exception as e:
            logger.error(f"Error in pywhatkit speech: {e}")
            return False
    
    def _prepare_text(self, text: str) -> str:
        """Enhanced text preparation for speech"""
        # Remove markdown/formatting
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters
        special_chars = ['*', '#', '_', '`', '~', '>', '<', '|', '\\', '/']
        for char in special_chars:
            text = text.replace(char, '')
        
        # Handle abbreviations
        abbreviations = {
            'Mr.': 'Mister',
            'Mrs.': 'Misses',
            'Dr.': 'Doctor',
            'Prof.': 'Professor',
            'e.g.': 'for example',
            'i.e.': 'that is',
            'vs.': 'versus',
            'etc.': 'et cetera'
        }
        for abbr, full in abbreviations.items():
            text = text.replace(abbr, full)
        
