"""
Speech Handler Module for Jarvis
Handles text-to-speech functionality using pyttsx3
"""

import os
import sys
import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import pyttsx3, provide fallback if not available
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    logger.warning("pyttsx3 package not installed. Speech features will be limited.")


class SpeechHandler:
    """Handles text-to-speech for Jarvis"""
    
    def __init__(self, config: dict):
        """Initialize speech handler"""
        self.config = config
        self.speech_config = config.get('speech', {})
        
        # Speech settings
        self.enabled = self.speech_config.get('enabled', True)
        self.voice_rate = self.speech_config.get('voice_rate', 150)
        self.voice_volume = self.speech_config.get('voice_volume', 1.0)
        self.voice_gender = self.speech_config.get('voice_gender', 'male')
        
        # Engine instance
        self.engine = None
        self._init_engine()
        
        logger.info(f"Speech handler initialized - Enabled: {self.enabled}")
    
    def _init_engine(self):
        """Initialize the text-to-speech engine"""
        if not PYTTSX3_AVAILABLE:
            logger.warning("pyttsx3 not available")
            return
        
        if not self.enabled:
            logger.info("Speech is disabled in configuration")
            return
        
        try:
            self.engine = pyttsx3.init()
            self._configure_engine()
            logger.info("TTS engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize TTS engine: {e}")
            self.engine = None
    
    def _configure_engine(self):
        """Configure the TTS engine with settings"""
        if self.engine is None:
            return
        
        try:
            # Set speech rate
            self.engine.setProperty('rate', self.voice_rate)
            
            # Set volume (0.0 to 1.0)
            volume = min(1.0, max(0.0, self.voice_volume))
            self.engine.setProperty('volume', volume)
            
            # Try to select voice by gender
            voices = self.engine.getProperty('voices')
            
            if voices:
                # Try to find a male or female voice based on gender setting
                selected_voice = None
                
                for voice in voices:
                    voice_name = voice.name.lower() if voice.name else ''
                    
                    if self.voice_gender == 'male':
                        if 'male' in voice_name or 'man' in voice_name:
                            selected_voice = voice
                            break
                    else:
                        if 'female' in voice_name or 'woman' in voice_name:
                            selected_voice = voice
                            break
                
                # If no gender-specific voice found, use first available
                if selected_voice is None:
                    selected_voice = voices[0]
                
                self.engine.setProperty('voice', selected_voice.id)
                logger.info(f"Selected voice: {selected_voice.name}")
            
        except Exception as e:
            logger.error(f"Error configuring TTS engine: {e}")
    
    def speak(self, text: str, block: bool = True) -> bool:
        """Convert text to speech"""
        if not self.enabled:
            logger.debug("Speech is disabled")
            return False
        
        if not text or not text.strip():
            logger.debug("Empty text provided for speech")
            return False
        
        if self.engine is None:
            logger.warning("TTS engine not available")
            return False
        
        try:
            # Clean and prepare text
            text = self._prepare_text(text)
            
            # Speak the text
            if block:
                self.engine.say(text)
                self.engine.runAndWait()
            else:
                # Speak in a separate thread for non-blocking mode
                thread = threading.Thread(target=self._speak_async, args=(text,))
                thread.daemon = True
                thread.start()
            
            logger.info(f"Spoke text: {text[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Error in text-to-speech: {e}")
            return False
    
    def _speak_async(self, text: str):
        """Speak text asynchronously"""
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            logger.error(f"Async speech error: {e}")
    
    def _prepare_text(self, text: str) -> str:
        """Prepare text for speech"""
        # Remove markdown/formatting
        import re
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that don't read well
        text = text.replace('*', '')
        text = text.replace('#', '')
        text = text.replace('_', '')
        
        # Truncate very long text
        max_chars = 500
        if len(text) > max_chars:
            text = text[:max_chars] + '...'
        
        return text.strip()
    
    def set_rate(self, rate: int):
        """Set speech rate (words per minute)"""
        self.voice_rate = rate
        if self.engine:
            self.engine.setProperty('rate', rate)
        logger.info(f"Speech rate set to {rate}")
    
    def set_volume(self, volume: float):
        """Set speech volume (0.0 to 1.0)"""
        self.voice_volume = max(0.0, min(1.0, volume))
        if self.engine:
            self.engine.setProperty('volume', self.voice_volume)
        logger.info(f"Volume set to {self.voice_volume}")
    
    def set_enabled(self, enabled: bool):
        """Enable or disable speech"""
        self.enabled = enabled
        logger.info(f"Speech {'enabled' if enabled else 'disabled'}")
    
    def get_voices(self) -> list:
        """Get available voices"""
        if self.engine is None:
            return []
        
        try:
            voices = self.engine.getProperty('voices')
            return [{'id': v.id, 'name': v.name, 'languages': v.languages} for v in voices]
        except Exception as e:
            logger.error(f"Error getting voices: {e}")
            return []
    
    def say_hello(self):
        """Say hello message"""
        name = self.config.get('assistant', {}).get('name', 'Jarvis')
        self.speak(f"Hello! I am {name}. How can I assist you today?")
    
    def say_goodbye(self):
        """Say goodbye message"""
        self.speak("Goodbye! Have a wonderful day!")
    
    def stop(self):
        """Stop current speech"""
        if self.engine:
            try:
                self.engine.stop()
            except Exception as e:
                logger.error(f"Error stopping speech: {e}")
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.engine:
                self.engine.stop()
                logger.info("TTS engine cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def is_available(self) -> bool:
        """Check if speech is available"""
        return PYTTSX3_AVAILABLE and self.enabled and self.engine is not None
    
    def get_status(self) -> dict:
        """Get speech handler status"""
        return {
            'enabled': self.enabled,
            'rate': self.voice_rate,
            'volume': self.voice_volume,
            'gender': self.voice_gender,
            'available': self.is_available(),
            'voices_count': len(self.get_voices())
        }
