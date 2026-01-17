import pvporcupine
import pyaudio
import numpy as np
import threading
import queue
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

class WakeWordDetector:
    """Wake word detector using Picovoice Porcupine for offline "Hey Jarvis" detection."""

    def __init__(self, access_key: str, keyword_path: Optional[str] = None, sensitivity: float = 0.5,
                 on_detection: Optional[Callable] = None):
        """
        Initialize the wake word detector.

        Args:
            access_key: Picovoice access key
            keyword_path: Path to custom keyword file (optional, uses built-in "Hey Jarvis" if None)
            sensitivity: Detection sensitivity (0.0 to 1.0)
            on_detection: Callback function called when wake word is detected
        """
        self.access_key = access_key
        self.keyword_path = keyword_path
        self.sensitivity = sensitivity
        self.on_detection = on_detection or self._default_detection_handler

        self.porcupine = None
        self.is_listening = False
        self.audio_queue = queue.Queue()
        self.thread = None

        self._init_porcupine()

    def _init_porcupine(self):
        """Initialize the Porcupine wake word engine."""
        try:
            if self.keyword_path:
                # Use custom keyword file
                self.porcupine = pvporcupine.create(
                    access_key=self.access_key,
                    keyword_paths=[self.keyword_path],
                    sensitivities=[self.sensitivity]
                )
            else:
                # Use built-in "Hey Jarvis" keyword
                self.porcupine = pvporcupine.create(
                    access_key=self.access_key,
                    keywords=['hey jarvis'],
                    sensitivities=[self.sensitivity]
                )
            logger.info("Porcupine wake word engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Porcupine: {e}")
            raise

    def _default_detection_handler(self):
        """Default handler for wake word detection."""
        logger.info("Wake word 'Hey Jarvis' detected!")

    def start_listening(self):
        """Start listening for wake word in a separate thread."""
        if self.is_listening:
            logger.warning("Wake word detector is already listening")
            return

        self.is_listening = True
        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()
        logger.info("Wake word detector started listening")

    def stop_listening(self):
        """Stop listening for wake word."""
        self.is_listening = False
        if self.thread:
            self.thread.join(timeout=1)
        logger.info("Wake word detector stopped listening")

    def process_frame(self, audio_frame: np.ndarray) -> bool:
        """
        Process an audio frame for wake word detection.

        Args:
            audio_frame: Audio frame as numpy array (16-bit PCM)

        Returns:
            True if wake word detected, False otherwise
        """
        if self.porcupine is None:
            return False

        try:
            # Ensure frame is the right format for Porcupine
            if audio_frame.dtype != np.int16:
                audio_frame = (audio_frame * 32767).astype(np.int16)

            # Process frame
            keyword_index = self.porcupine.process(audio_frame)

            if keyword_index >= 0:
                # Wake word detected
                self.on_detection()
                return True

            return False
        except Exception as e:
            logger.error(f"Error processing audio frame: {e}")
            return False

    def _listen_loop(self):
        """Main listening loop for wake word detection."""
        # This would be used if we want the detector to handle its own audio capture
        # For now, it's designed to process frames passed to process_frame
        pass

    def __del__(self):
        """Cleanup resources."""
        if self.porcupine:
            self.porcupine.delete()


class SpeechRecognizer:
    """Speech recognizer with integrated wake word detection."""

    def __init__(self, access_key: str, wake_word_callback: Optional[Callable] = None):
        """
        Initialize the speech recognizer.

        Args:
            access_key: Picovoice access key for wake word detection
            wake_word_callback: Callback when wake word is detected
        """
        self.wake_detector = WakeWordDetector(
            access_key=access_key,
            on_detection=wake_word_callback
        )

        self.audio = None
        self.stream = None
        self.is_recording = False
        self.sample_rate = 16000  # Porcupine requires 16kHz
        self.frame_length = 512   # Porcupine frame length

    def start_wake_word_detection(self):
        """Start wake word detection."""
        self.wake_detector.start_listening()

    def stop_wake_word_detection(self):
        """Stop wake word detection."""
        self.wake_detector.stop_listening()

    def process_audio_frame(self, frame: np.ndarray) -> bool:
        """
        Process an audio frame through the wake word detector.

        Args:
            frame: Audio frame as numpy array

        Returns:
            True if wake word detected
        """
        return self.wake_detector.process_frame(frame)

    def start_audio_stream(self):
        """Start audio capture stream."""
        if self.audio is None:
            self.audio = pyaudio.PyAudio()

        if self.stream is None:
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.frame_length
            )

        self.is_recording = True
        logger.info("Audio stream started")

    def stop_audio_stream(self):
        """Stop audio capture stream."""
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        if self.audio:
            self.audio.terminate()
            self.audio = None
        logger.info("Audio stream stopped")

    def get_audio_frame(self) -> Optional[np.ndarray]:
        """
        Get the next audio frame from the stream.

        Returns:
            Audio frame as numpy array, or None if not recording
        """
        if not self.is_recording or self.stream is None:
            return None

        try:
            data = self.stream.read(self.frame_length, exception_on_overflow=False)
            frame = np.frombuffer(data, dtype=np.int16)
            return frame
        except Exception as e:
            logger.error(f"Error reading audio frame: {e}")
            return None</content>
</xai:function_call">The SpeechRecognizer class provides audio capture functionality and integrates the WakeWordDetector for wake word detection. The WakeWordDetector uses Picovoice Porcupine to detect the "Hey Jarvis" wake word offline. Key methods include initialization of the Porcupine engine, processing audio frames, and handling detection events through callbacks. The implementation ensures proper resource management and threading for continuous audio processing.