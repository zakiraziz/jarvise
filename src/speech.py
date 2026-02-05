"""
Speech Handler Module
"""

import logging

logger = logging.getLogger(__name__)


class SpeechHandler:
    """Handles text-to-speech operations"""
    
    def __init__(self, config: dict):
        self.config = config
        self.enabled = config['speech']['enabled']
        self.engine = None
        
        if self.enabled:
            self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize TTS engine"""
        try:
            import pyttsx3
            
            self.engine = pyttsx3.init()
            
            # Configure voice
            voices = self.engine.getProperty('voices')
            
            if self.config['speech']['voice_gender'].lower() == 'female':
                # Try to find a female voice
                for voice in voices:
                    if 'female' in voice.name.lower():
                        self.engine.setProperty('voice', voice.id)
                        break
            
            # Set rate and volume
            rate = self.engine.getProperty('rate')
            self.engine.setProperty('rate', rate * (self.config['speech']['voice_rate'] / 150))
            self.engine.setProperty('volume', self.config['speech']['voice_volume'])
            
            logger.info("TTS engine initialized successfully")
            
        except ImportError:
            logger.warning("pyttsx3 not installed. Speech disabled.")
            self.enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize TTS engine: {e}")
            self.enabled = False
    
    def speak(self, text: str):
        """Speak text"""
        if not self.enabled or not self.engine:
            return
        
        try:
            # Clean text for speech
            text = self._clean_text(text)
            
            if not text.strip():
                return
            
            self.engine.say(text)
            self.engine.runAndWait()
                
        except Exception as e:
            logger.error(f"Speech error: {e}")
    
    def _clean_text(self, text: str) -> str:
        """Clean text for TTS"""
        import re
        
        # Remove markdown code blocks
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`[^`]*`', '', text)
        
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        
        # Remove special characters
        text = re.sub(r'[^\w\s.,!?;:\'"-]', ' ', text)
        
        # Replace multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Truncate if too long
        if len(text) > 500:
            text = text[:500] + "..."
        
        return text.strip()
    
    def cleanup(self):
        """Cleanup resources"""
        if self.engine:
            try:
                self.engine.stop()
            except:
                pass
