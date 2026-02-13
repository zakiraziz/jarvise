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
        # Handle URLs and emails
        text = re.sub(r'https?://\S+', 'link', text)
        text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', 'email address', text)
        
        # Truncate very long text
        max_chars = 500
        if len(text) > max_chars:
            text = text[:max_chars] + '...'
        
        return text.strip()
    
    def _cache_speech(self, text: str):
        """Cache speech for future use"""
        if not self.cache_enabled:
            return
        
        try:
            cache_key = hash(text)
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.mp3")
            
            if not os.path.exists(cache_file):
                # Generate and cache the speech
                tts = gTTS(text=text, lang='en', slow=False)
                tts.save(cache_file)
                self.speech_cache[text] = cache_file
                logger.info(f"Cached speech: {text[:50]}...")
        except Exception as e:
            logger.error(f"Error caching speech: {e}")
    
    def _play_cached(self, text: str) -> bool:
        """Play cached speech"""
        try:
            cache_file = self.speech_cache.get(text)
            if cache_file and os.path.exists(cache_file):
                pygame.mixer.init()
                pygame.mixer.music.load(cache_file)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                pygame.mixer.quit()
                return True
        except Exception as e:
            logger.error(f"Error playing cached speech: {e}")
        return False
    
    def set_rate(self, rate: int):
        """Set speech rate (words per minute)"""
        self.voice_rate = rate
        if self.tts_engine == 'pyttsx3' and self.engine:
            self.engine.setProperty('rate', rate)
        logger.info(f"Speech rate set to {rate}")
    
    def set_volume(self, volume: float):
        """Set speech volume (0.0 to 1.0)"""
        self.voice_volume = max(0.0, min(1.0, volume))
        if self.tts_engine == 'pyttsx3' and self.engine:
            self.engine.setProperty('volume', self.voice_volume)
        logger.info(f"Volume set to {self.voice_volume}")
    
    def set_voice(self, voice_id: str):
        """Set voice by ID"""
        self.voice_id = voice_id
        if self.tts_engine == 'pyttsx3' and self.engine:
            self.engine.setProperty('voice', voice_id)
        logger.info(f"Voice set to {voice_id}")
    
    def set_enabled(self, enabled: bool):
        """Enable or disable speech"""
        self.enabled = enabled
        if not enabled:
            self.stop()
            self.clear_queue()
        logger.info(f"Speech {'enabled' if enabled else 'disabled'}")
    
    def set_tts_engine(self, engine: str):
        """Change TTS engine"""
        if engine in ['pyttsx3', 'gtts', 'pywhatkit']:
            self.tts_engine = engine
            self._init_engine()
            logger.info(f"TTS engine changed to {engine}")
        else:
            logger.error(f"Invalid TTS engine: {engine}")
    
    def get_voices(self) -> List[Dict[str, Any]]:
        """Get available voices with enhanced information"""
        if self.tts_engine != 'pyttsx3' or self.engine is None:
            return []
        
        try:
            voices = self.engine.getProperty('voices')
            voice_list = []
            
            for v in voices:
                voice_info = {
                    'id': v.id,
                    'name': v.name,
                    'languages': v.languages,
                    'gender': self._detect_voice_gender(v),
                    'age': self._detect_voice_age(v)
                }
                voice_list.append(voice_info)
            
            return voice_list
        except Exception as e:
            logger.error(f"Error getting voices: {e}")
            return []
    
    def _detect_voice_gender(self, voice) -> str:
        """Detect voice gender from voice properties"""
        voice_str = str(voice.name).lower() + str(voice.id).lower()
        if 'female' in voice_str or 'zira' in voice_str:
            return 'female'
        elif 'male' in voice_str or 'david' in voice_str:
            return 'male'
        return 'unknown'
    
    def _detect_voice_age(self, voice) -> str:
        """Detect voice age from voice properties"""
        voice_str = str(voice.name).lower()
        if 'child' in voice_str:
            return 'child'
        elif 'teen' in voice_str:
            return 'teen'
        elif 'adult' in voice_str:
            return 'adult'
        elif 'senior' in voice_str:
            return 'senior'
        return 'unknown'
    
    def say_hello(self, custom_name: str = None):
        """Say personalized hello message"""
        name = custom_name or self.config.get('assistant', {}).get('name', 'Jarvis')
        greeting = self.speech_config.get('greeting', f"Hello! I am {name}. How can I assist you today?")
        self.speak(greeting, priority=SpeechPriority.NORMAL)
    
    def say_goodbye(self, custom_message: str = None):
        """Say personalized goodbye message"""
        message = custom_message or self.speech_config.get('farewell', "Goodbye! Have a wonderful day!")
        self.speak(message, priority=SpeechPriority.NORMAL)
    
    def say_error(self, error_message: str):
        """Announce error message"""
        self.speak(f"Sorry, an error occurred: {error_message}", priority=SpeechPriority.HIGH)
    
    def say_waiting(self):
        """Announce waiting state"""
        self.speak("Please wait a moment", priority=SpeechPriority.LOW)
    
    def stop(self):
        """Stop current speech"""
        if self.tts_engine == 'pyttsx3' and self.engine:
            try:
                self.engine.stop()
            except Exception as e:
                logger.error(f"Error stopping speech: {e}")
        elif self.tts_engine == 'gtts':
            try:
                pygame.mixer.music.stop()
            except:
                pass
    
    def clear_queue(self):
        """Clear all pending speech from queue"""
        while not self.speech_queue.empty():
            try:
                self.speech_queue.get_nowait()
            except:
                break
        logger.info("Speech queue cleared")
    
    def pause(self):
        """Pause speech temporarily"""
        if self.tts_engine == 'pyttsx3' and self.engine:
            try:
                self.engine.stop()  # pyttsx3 doesn't support pause, so stop
            except:
                pass
        logger.info("Speech paused")
    
    def resume(self):
        """Resume speech"""
        # pyttsx3 doesn't support resume, so nothing to do
        logger.info("Speech resumed")
    
    def cleanup(self):
        """Enhanced cleanup resources"""
        try:
            self.queue_running = False
            if self.queue_thread and self.queue_thread.is_alive():
                self.queue_thread.join(timeout=2)
            
            self.stop()
            self.clear_queue()
            
            if self.tts_engine == 'pyttsx3' and self.engine:
                self.engine.stop()
            
            if self.tts_engine == 'gtts':
                pygame.mixer.quit()
            
            logger.info("Speech handler cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def is_available(self) -> bool:
        """Check if speech is available"""
        if not self.enabled:
            return False
        
        if self.tts_engine == 'pyttsx3':
            return PYTTSX3_AVAILABLE and self.engine is not None
        elif self.tts_engine == 'gtts':
            return GTTS_AVAILABLE
        elif self.tts_engine == 'pywhatkit':
            return KIT_AVAILABLE
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """Enhanced status information"""
        return {
            'enabled': self.enabled,
            'tts_engine': self.tts_engine,
            'rate': self.voice_rate,
            'volume': self.voice_volume,
            'gender': self.voice_gender,
            'available': self.is_available(),
            'queue_enabled': self.speech_queue_enabled,
            'queue_size': self.speech_queue.qsize(),
            'currently_speaking': self.currently_speaking,
            'interruptible': self.interruptible,
            'cache_enabled': self.cache_enabled,
            'cache_size': len(self.speech_cache),
            'voices_count': len(self.get_voices())
        }
    
    def get_queue_info(self) -> Dict[str, Any]:
        """Get information about speech queue"""
        return {
            'enabled': self.speech_queue_enabled,
            'size': self.speech_queue.qsize(),
            'max_size': self.max_queue_size,
            'currently_speaking': self.currently_speaking
        }
    
    def test_speech(self, text: str = "This is a test of the speech system.") -> bool:
        """Test speech functionality"""
        logger.info("Testing speech system...")
        return self.speak(text, priority=SpeechPriority.CRITICAL, block=True)
