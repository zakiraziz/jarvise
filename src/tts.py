import pyttsx3
import threading
import queue
import logging

logger = logging.getLogger(__name__)

class TextToSpeech:
    """Text-to-speech engine for voice responses."""

    def __init__(self, voice_id: int = 0, rate: int = 200, volume: float = 1.0):
        """
        Initialize TTS engine.

        Args:
            voice_id: Voice index to use
            rate: Speech rate (words per minute)
            volume: Volume level (0.0 to 1.0)
        """
        self.engine = pyttsx3.init()
        self.engine.setProperty('voice', self.engine.getProperty('voices')[voice_id].id)
        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', volume)

        self.speech_queue = queue.Queue()
        self.is_speaking = False
        self.thread = None

    def speak(self, text: str):
        """Add text to speech queue."""
        self.speech_queue.put(text)
        if not self.is_speaking:
            self._start_speaking()

    def _start_speaking(self):
        """Start speaking thread."""
        if self.thread and self.thread.is_alive():
            return

        self.is_speaking = True
        self.thread = threading.Thread(target=self._speak_loop, daemon=True)
        self.thread.start()

    def _speak_loop(self):
        """Main speaking loop."""
        while self.is_speaking:
            try:
                text = self.speech_queue.get(timeout=1)
                self.engine.say(text)
                self.engine.runAndWait()
            except queue.Empty:
                self.is_speaking = False
            except Exception as e:
                logger.error(f"TTS error: {e}")

    def stop(self):
        """Stop speaking."""
        self.is_speaking = False
        if self.engine:
            self.engine.stop()

    def __del__(self):
        """Cleanup."""
        self.stop()</content>
</xai:function_call name="execute_command">
<parameter name="command">cd code_assistant && pip install pyttsx3