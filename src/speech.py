import pvporcupine
try:
    import pyaudio
except ImportError:
    print("Warning: pyaudio not available, audio features disabled")
    pyaudio = None
import numpy as np
import threading
import queue
import logging
import time
import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, List, Dict, Any
from collections import deque
import wave
import tempfile

logger = logging.getLogger(__name__)

class DetectionMode(Enum):
    """Mode of operation for the wake word detector."""
    CONTINUOUS = "continuous"
    ON_DEMAND = "on_demand"
    ENERGY_BASED = "energy_based"

@dataclass
class WakeWordConfig:
    """Configuration for wake word detection."""
    keyword_path: Optional[str] = None
    keyword: str = "jarvis"
    sensitivity: float = 0.5
    auto_reload: bool = False
    detection_mode: DetectionMode = DetectionMode.CONTINUOUS

@dataclass
class AudioConfig:
    """Configuration for audio capture."""
    sample_rate: int = 16000
    frame_length: int = 512
    channels: int = 1
    format: int = pyaudio.paInt16 if pyaudio else None
    input_device_index: Optional[int] = None
    frames_per_buffer: int = 512

class AudioBuffer:
    """Circular buffer for audio data storage and processing."""
    
    def __init__(self, max_duration_seconds: float = 10, sample_rate: int = 16000):
        """
        Initialize audio buffer.
        
        Args:
            max_duration_seconds: Maximum duration to store in seconds
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration_seconds * sample_rate)
        self.buffer = np.zeros(self.max_samples, dtype=np.float32)
        self.write_pos = 0
        self.is_full = False
        self.lock = threading.RLock()
        
    def add_frame(self, frame: np.ndarray):
        """Add an audio frame to the buffer."""
        with self.lock:
            frame = frame.astype(np.float32) / 32768.0  # Normalize to [-1, 1]
            n_samples = len(frame)
            
            if n_samples + self.write_pos <= self.max_samples:
                self.buffer[self.write_pos:self.write_pos + n_samples] = frame
                self.write_pos += n_samples
            else:
                # Wrap around
                remaining = self.max_samples - self.write_pos
                self.buffer[self.write_pos:] = frame[:remaining]
                self.buffer[:n_samples - remaining] = frame[remaining:]
                self.write_pos = n_samples - remaining
                self.is_full = True
                
    def get_recent(self, duration_seconds: float) -> np.ndarray:
        """
        Get recent audio data.
        
        Args:
            duration_seconds: Duration of audio to retrieve in seconds
            
        Returns:
            Audio data as numpy array
        """
        with self.lock:
            n_samples = int(duration_seconds * self.sample_rate)
            n_samples = min(n_samples, self.max_samples if self.is_full else self.write_pos)
            
            start_pos = (self.write_pos - n_samples) % self.max_samples
            if start_pos + n_samples <= self.max_samples:
                return self.buffer[start_pos:start_pos + n_samples]
            else:
                return np.concatenate([
                    self.buffer[start_pos:],
                    self.buffer[:n_samples - (self.max_samples - start_pos)]
                ])
    
    def clear(self):
        """Clear the buffer."""
        with self.lock:
            self.buffer.fill(0)
            self.write_pos = 0
            self.is_full = False

class EnergyVAD:
    """Voice Activity Detector based on energy threshold."""
    
    def __init__(self, sample_rate: int = 16000, frame_length: int = 512,
                 threshold: float = 0.01, silence_duration: float = 0.5):
        """
        Initialize Energy VAD.
        
        Args:
            sample_rate: Audio sample rate
            frame_length: Frame length in samples
            threshold: Energy threshold for voice detection
            silence_duration: Duration of silence to consider speech ended
        """
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.threshold = threshold
        self.silence_frames = int(silence_duration * sample_rate / frame_length)
        self.silence_counter = 0
        self.is_speech = False
        self.speech_start_time = None
        
    def process_frame(self, frame: np.ndarray) -> bool:
        """
        Process audio frame for voice activity detection.
        
        Args:
            frame: Audio frame
            
        Returns:
            True if speech is detected
        """
        # Convert to float for energy calculation
        frame_float = frame.astype(np.float32) / 32768.0
        energy = np.mean(frame_float ** 2)
        
        if energy > self.threshold:
            self.silence_counter = 0
            if not self.is_speech:
                self.is_speech = True
                self.speech_start_time = time.time()
        else:
            if self.is_speech:
                self.silence_counter += 1
                if self.silence_counter >= self.silence_frames:
                    self.is_speech = False
                    self.speech_start_time = None
        
        return self.is_speech
    
    def get_speech_duration(self) -> Optional[float]:
        """Get duration of current speech segment."""
        if self.is_speech and self.speech_start_time:
            return time.time() - self.speech_start_time
        return None

class WakeWordDetector:
    """Enhanced wake word detector with multiple features."""

    def __init__(self, access_key: str, config: Optional[WakeWordConfig] = None,
                 on_detection: Optional[Callable] = None,
                 on_error: Optional[Callable] = None):
        """
        Initialize the wake word detector.

        Args:
            access_key: Picovoice access key
            config: Wake word configuration
            on_detection: Callback function called when wake word is detected
            on_error: Callback function for error handling
        """
        self.access_key = access_key
        self.config = config or WakeWordConfig()
        self.on_detection = on_detection or self._default_detection_handler
        self.on_error = on_error or self._default_error_handler
        
        self.porcupine = None
        self.is_listening = False
        self.audio_queue = queue.Queue(maxsize=100)
        self.processing_thread = None
        self.audio_thread = None
        self.stats = {
            'detections': 0,
            'errors': 0,
            'frames_processed': 0,
            'last_detection_time': None
        }
        self.audio_buffer = AudioBuffer(max_duration_seconds=5)
        self.energy_vad = EnergyVAD()
        self.detection_history = deque(maxlen=10)
        
        self._init_porcupine()
        logger.info(f"Wake word detector initialized with mode: {self.config.detection_mode}")

    def _init_porcupine(self):
        """Initialize the Porcupine wake word engine."""
        try:
            if self.config.keyword_path:
                self.porcupine = pvporcupine.create(
                    access_key=self.access_key,
                    keyword_paths=[self.config.keyword_path],
                    sensitivities=[self.config.sensitivity]
                )
            else:
                self.porcupine = pvporcupine.create(
                    access_key=self.access_key,
                    keywords=[self.config.keyword],
                    sensitivities=[self.config.sensitivity]
                )
            logger.info("Porcupine wake word engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Porcupine: {e}")
            self.on_error(e)

    def _default_detection_handler(self):
        """Default handler for wake word detection."""
        logger.info(f"Wake word '{self.config.keyword}' detected!")
        
    def _default_error_handler(self, error: Exception):
        """Default error handler."""
        logger.error(f"Wake word detector error: {error}")

    def start_listening(self):
        """Start listening for wake word with audio capture."""
        if self.is_listening:
            logger.warning("Wake word detector is already listening")
            return

        self.is_listening = True
        
        # Start audio capture thread
        self.audio_thread = threading.Thread(target=self._audio_capture_loop, daemon=True)
        self.audio_thread.start()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
        logger.info("Wake word detector started listening")

    def stop_listening(self):
        """Stop listening for wake word."""
        self.is_listening = False
        
        # Wait for threads to finish
        if self.audio_thread:
            self.audio_thread.join(timeout=2)
        if self.processing_thread:
            self.processing_thread.join(timeout=2)
            
        logger.info("Wake word detector stopped listening")

    def _audio_capture_loop(self):
        """Audio capture loop."""
        audio = None
        stream = None
        
        try:
            if pyaudio:
                audio = pyaudio.PyAudio()
                stream = audio.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=512,
                    stream_callback=self._audio_callback
                )
                stream.start_stream()
                
                # Keep thread alive while listening
                while self.is_listening and stream.is_active():
                    time.sleep(0.1)
                    
        except Exception as e:
            logger.error(f"Audio capture error: {e}")
            self.on_error(e)
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
            if audio:
                audio.terminate()

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for audio capture."""
        if status:
            logger.warning(f"Audio stream status: {status}")
        
        # Convert audio data to numpy array
        frame = np.frombuffer(in_data, dtype=np.int16)
        
        # Add to buffer for processing
        self.audio_buffer.add_frame(frame)
        
        # Add to queue for wake word processing
        try:
            self.audio_queue.put(frame, timeout=0.01)
        except queue.Full:
            pass  # Drop frame if queue is full
            
        return (in_data, pyaudio.paContinue)

    def _processing_loop(self):
        """Main processing loop for wake word detection."""
        while self.is_listening:
            try:
                # Get audio frame from queue
                frame = self.audio_queue.get(timeout=0.1)
                self.stats['frames_processed'] += 1
                
                # Process for wake word detection
                if self._process_frame(frame):
                    self.stats['detections'] += 1
                    self.stats['last_detection_time'] = time.time()
                    
                    # Add to history
                    self.detection_history.append({
                        'time': time.time(),
                        'mode': self.config.detection_mode.value
                    })
                    
            except queue.Empty:
                continue  # No audio data available
            except Exception as e:
                logger.error(f"Processing error: {e}")
                self.stats['errors'] += 1
                self.on_error(e)

    def _process_frame(self, frame: np.ndarray) -> bool:
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
            # Apply detection mode logic
            if self.config.detection_mode == DetectionMode.ENERGY_BASED:
                if not self.energy_vad.process_frame(frame):
                    return False  # Skip processing if no voice activity
            
            # Ensure frame is the right format for Porcupine
            if frame.dtype != np.int16:
                frame = (frame * 32767).astype(np.int16)

            # Process frame
            keyword_index = self.porcupine.process(frame)

            if keyword_index >= 0:
                # Wake word detected
                threading.Thread(target=self.on_detection, daemon=True).start()
                return True

            return False
        except Exception as e:
            logger.error(f"Error processing audio frame: {e}")
            return False

    def process_frame(self, audio_frame: np.ndarray) -> bool:
        """
        Process an audio frame for wake word detection.
        
        Args:
            audio_frame: Audio frame as numpy array (16-bit PCM)

        Returns:
            True if wake word detected, False otherwise
        """
        return self._process_frame(audio_frame)

    def save_audio_context(self, duration_before: float = 1.0, 
                          duration_after: float = 2.0) -> Optional[str]:
        """
        Save audio context around last detection to a WAV file.
        
        Args:
            duration_before: Seconds before detection to save
            duration_after: Seconds after detection to save
            
        Returns:
            Path to saved WAV file or None
        """
        try:
            # Get recent audio
            total_duration = duration_before + duration_after
            audio_data = self.audio_buffer.get_recent(total_duration)
            
            if len(audio_data) == 0:
                return None
                
            # Convert back to 16-bit PCM
            audio_data = (audio_data * 32767).astype(np.int16)
            
            # Save to temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                with wave.open(f.name, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(16000)
                    wav_file.writeframes(audio_data.tobytes())
                
                return f.name
        except Exception as e:
            logger.error(f"Failed to save audio context: {e}")
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get detector statistics."""
        return {
            **self.stats,
            'buffer_duration': self.audio_buffer.write_pos / self.audio_buffer.sample_rate,
            'buffer_full': self.audio_buffer.is_full,
            'detection_history': list(self.detection_history),
            'config': {
                'keyword': self.config.keyword,
                'sensitivity': self.config.sensitivity,
                'mode': self.config.detection_mode.value
            }
        }

    def reload_config(self, config: WakeWordConfig):
        """Reload detector with new configuration."""
        self.stop_listening()
        
        # Delete old porcupine instance
        if self.porcupine:
            self.porcupine.delete()
            
        # Update config
        self.config = config
        
        # Reinitialize
        self._init_porcupine()
        
        # Restart if was listening
        if self.is_listening:
            self.start_listening()

    def __del__(self):
        """Cleanup resources."""
        self.stop_listening()
        if self.porcupine:
            self.porcupine.delete()


class SpeechRecognizer:
    """Enhanced speech recognizer with integrated wake word detection."""

    def __init__(self, access_key: str, wake_word_callback: Optional[Callable] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the speech recognizer.

        Args:
            access_key: Picovoice access key for wake word detection
            wake_word_callback: Callback when wake word is detected
            config: Configuration dictionary
        """
        self.access_key = access_key
        self.config = config or {}
        
        # Initialize wake word detector with config
        wake_word_config = WakeWordConfig(
            keyword=self.config.get('keyword', 'jarvis'),
            sensitivity=self.config.get('sensitivity', 0.5),
            detection_mode=DetectionMode(self.config.get('detection_mode', 'continuous')),
            auto_reload=self.config.get('auto_reload', False)
        )
        
        self.wake_detector = WakeWordDetector(
            access_key=access_key,
            config=wake_word_config,
            on_detection=wake_word_callback
        )

        self.audio = None
        self.stream = None
        self.is_recording = False
        self.audio_config = AudioConfig()
        
        # State management
        self.state = 'idle'  # idle, listening, processing
        self.state_lock = threading.RLock()
        
        # Audio recording buffers
        self.recording_buffer = []
        self.max_recording_duration = self.config.get('max_recording_duration', 30)
        
        # Callbacks
        self.on_recording_start = None
        self.on_recording_stop = None
        self.on_audio_frame = None

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

    def start_audio_stream(self, callback_mode: bool = False):
        """Start audio capture stream."""
        if self.audio is None and pyaudio:
            self.audio = pyaudio.PyAudio()

        if self.stream is None and pyaudio:
            self.stream = self.audio.open(
                format=self.audio_config.format,
                channels=self.audio_config.channels,
                rate=self.audio_config.sample_rate,
                input=True,
                frames_per_buffer=self.audio_config.frames_per_buffer,
                input_device_index=self.audio_config.input_device_index,
                stream_callback=self._stream_callback if callback_mode else None
            )

        self.is_recording = True
        logger.info("Audio stream started")

    def stop_audio_stream(self):
        """Stop audio capture stream."""
        with self.state_lock:
            self.is_recording = False
            
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
                
            if self.audio:
                self.audio.terminate()
                self.audio = None
                
            logger.info("Audio stream stopped")

    def _stream_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream (used in callback mode)."""
        if status:
            logger.warning(f"Stream status: {status}")
            
        frame = np.frombuffer(in_data, dtype=np.int16)
        
        # Process wake word
        if self.wake_detector.process_frame(frame):
            self._on_wake_word_detected()
        
        # Call custom frame callback if set
        if self.on_audio_frame:
            self.on_audio_frame(frame)
        
        return (in_data, pyaudio.paContinue)

    def _on_wake_word_detected(self):
        """Handle wake word detection."""
        with self.state_lock:
            if self.state == 'idle':
                self.state = 'listening'
                self._start_recording()
                
                if self.on_recording_start:
                    self.on_recording_start()

    def _start_recording(self):
        """Start recording audio after wake word detection."""
        self.recording_buffer = []
        self.recording_start_time = time.time()
        
        # Start a thread to monitor recording duration
        threading.Thread(target=self._recording_monitor, daemon=True).start()

    def _recording_monitor(self):
        """Monitor recording duration and stop if too long."""
        while self.state == 'listening':
            if time.time() - self.recording_start_time > self.max_recording_duration:
                self.stop_recording()
                break
            time.sleep(0.1)
        }
