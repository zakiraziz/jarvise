#!/usr/bin/env python3
"""
Jarvis AI Assistant - Event-Driven Main Loop
Integrates voice interface, LLM, and modular commands for intelligent assistance.
"""

import sys
import os
import yaml
import logging
import asyncio
import threading
import queue
from pathlib import Path
from typing import Optional, Dict, Any
import time

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from speech import SpeechRecognizer, WakeWordDetector
from tts import TextToSpeech
from ai import AIAssistant
from commands import CommandRegistry, WebSearchCommand, AutomationCommand, SmartHomeCommand, ScheduleCommand

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Jarvis:
    """Jarvis AI Assistant - Event-driven main system."""

    def __init__(self, config_path: Optional[Path] = None):
        self.config = self._load_config(config_path)
        self._setup_components()

        # System state
        self.is_active = False
        self.is_listening = False

        # Asyncio event loop
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        # Event queues for coordination
        self.input_queue = asyncio.Queue()
        self.decision_queue = asyncio.Queue()
        self.action_queue = asyncio.Queue()

        # Threading for concurrent processing
        self.audio_thread = None
        self.processing_thread = None
        self.shutdown_event = threading.Event()

    def _load_config(self, config_path: Optional[Path]) -> dict:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'

        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> dict:
        """Get default configuration."""
        return {
            'openai': {
                'api_key': os.getenv('OPENAI_API_KEY', 'your_key_here'),
                'model': 'gpt-4-turbo-preview'
            },
            'speech': {
                'picovoice_key': os.getenv('PICOVOICE_KEY', 'your_key_here'),
                'wake_word_sensitivity': 0.5
            },
            'tts': {
                'voice_id': 0,
                'rate': 200,
                'volume': 1.0
            },
            'commands': {
                'smart_home_api_url': 'https://api.smarthome.com',
                'smart_home_api_key': 'your_key_here'
            }
        }

    def _setup_components(self):
        """Initialize all Jarvis components."""
        # Voice interface
        self.speech_recognizer = SpeechRecognizer(
            access_key=self.config['speech']['picovoice_key']
        )
        self.tts = TextToSpeech(
            voice_id=self.config['tts']['voice_id'],
            rate=self.config['tts']['rate'],
            volume=self.config['tts']['volume']
        )

        # AI/LLM
        self.ai = AIAssistant(
            api_key=self.config['openai']['api_key'],
            model=self.config['openai']['model']
        )

        # Command system
        self.command_registry = CommandRegistry()
        self._register_commands()

    def _register_commands(self):
        """Register available commands."""
        self.command_registry.register(WebSearchCommand())
        self.command_registry.register(AutomationCommand())
        self.command_registry.register(SmartHomeCommand(
            api_url=self.config['commands']['smart_home_api_url'],
            api_key=self.config['commands']['smart_home_api_key']
        ))
        self.command_registry.register(ScheduleCommand(self.command_registry))

    async def run(self):
        """Main event-driven application loop."""
        logger.info("Starting Jarvis AI Assistant...")
        self._show_welcome()

        # Start concurrent threads
        self._start_threads()

        try:
            # Start the main event loop
            await self._main_event_loop()
        except KeyboardInterrupt:
            logger.info("Shutdown requested by user")
        except Exception as e:
            logger.error(f"Fatal error in main loop: {e}")
        finally:
            self._stop_threads()
            self._handle_exit()

    async def _main_event_loop(self):
        """Main asyncio event loop coordinating all components."""
        # Start wake word detection
        self._start_wake_word_detection()

        while not self.shutdown_event.is_set():
            try:
                # Wait for events with timeout
                input_event = await asyncio.wait_for(
                    self.input_queue.get(), timeout=1.0
                )

                # Process input event
                await self._handle_input_event(input_event)

            except asyncio.TimeoutError:
                # No input events, continue monitoring
                continue
            except Exception as e:
                logger.error(f"Error in main event loop: {e}")

    async def _handle_input_event(self, event: Dict[str, Any]):
        """Handle input events from voice/text."""
        event_type = event.get('type')

        if event_type == 'wake_word_detected':
            await self._handle_wake_word()
        elif event_type == 'speech_recognized':
            await self._handle_speech_input(event['text'])
        elif event_type == 'shutdown':
            self.shutdown_event.set()

    async def _handle_wake_word(self):
        """Handle wake word detection."""
        logger.info("Wake word detected - activating Jarvis")
        self.is_active = True
        self.tts.speak("Yes, how can I help you?")
        # Start listening for speech input
        self._start_speech_recognition()

    async def _handle_speech_input(self, text: str):
        """Handle recognized speech input."""
        if not self.is_active:
            return

        logger.info(f"Processing speech input: {text}")

        # Put input into decision queue
        await self.decision_queue.put({'type': 'process_input', 'text': text})

        # Process decision in thread pool
        decision = await self.loop.run_in_executor(None, self._process_decision, text)

        # Execute action
        await self._execute_action(decision)

    def _process_decision(self, text: str) -> Dict[str, Any]:
        """Process user input and decide on action (runs in thread pool)."""
        return self.ai.decide_action(text)

    async def _execute_action(self, decision: Dict[str, Any]):
        """Execute the decided action."""
        action_type = decision.get('action')
        params = decision.get('params', {})

        if action_type == 'command':
            await self._execute_command(params)
        elif action_type == 'respond':
            response = params.get('response_text', 'I heard you.')
            self.tts.speak(response)
        elif action_type == 'search':
            await self._execute_search(params)
        elif action_type == 'schedule':
            await self._execute_schedule(params)

        # Deactivate after response
        self.is_active = False

    async def _execute_command(self, params: Dict[str, Any]):
        """Execute a modular command."""
        command_name = params.get('command_name', 'unknown')
        command_params = params.get('command_params', '')

        result = await self.loop.run_in_executor(
            None, self.command_registry.execute, command_name, command_params
        )

        if result:
            self.tts.speak(f"Command executed: {result}")

    async def _execute_search(self, params: Dict[str, Any]):
        """Execute web search."""
        query = params.get('query', '')
        result = await self.loop.run_in_executor(
            None, self.command_registry.execute, 'web_search', f'query={query}'
        )
        self.tts.speak(f"Search result: {result}")

    async def _execute_schedule(self, params: Dict[str, Any]):
        """Execute scheduling command."""
        task = params.get('task', '')
        time_str = params.get('time', 'now')
        result = await self.loop.run_in_executor(
            None, self.command_registry.execute, 'schedule',
            f'command=respond,command_params=response_text=Reminder: {task},time={time_str}'
        )
        self.tts.speak(f"Scheduled: {result}")

    def _start_wake_word_detection(self):
        """Start wake word detection in background thread."""
        def wake_callback():
            asyncio.run_coroutine_threadsafe(
                self.input_queue.put({'type': 'wake_word_detected'}),
                self.loop
            )

        self.speech_recognizer.wake_word_callback = wake_callback
        self.speech_recognizer.start_wake_word_detection()
        self.speech_recognizer.start_audio_stream()

        # Start audio processing thread
        self.audio_thread = threading.Thread(target=self._audio_processing_loop, daemon=True)
        self.audio_thread.start()

    def _start_speech_recognition(self):
        """Start speech recognition for active listening."""
        self.is_listening = True
        # Note: In a full implementation, this would start speech-to-text
        # For now, we'll simulate with text input

    def _audio_processing_loop(self):
        """Background thread for processing audio frames."""
        while not self.shutdown_event.is_set():
            try:
                frame = self.speech_recognizer.get_audio_frame()
                if frame is not None:
                    detected = self.speech_recognizer.process_audio_frame(frame)
                    if detected and not self.is_active:
                        # Wake word detected
                        asyncio.run_coroutine_threadsafe(
                            self.input_queue.put({'type': 'wake_word_detected'}),
                            self.loop
                        )
                time.sleep(0.1)  # Small delay to prevent busy waiting
            except Exception as e:
                logger.error(f"Audio processing error: {e}")

    def _start_threads(self):
        """Start all background threads."""
        # Audio processing is already started in wake word detection
        pass

    def _stop_threads(self):
        """Stop all background threads."""
        self.shutdown_event.set()

        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=2)

        self.speech_recognizer.stop_wake_word_detection()
        self.speech_recognizer.stop_audio_stream()
        self.tts.stop()

    def _show_welcome(self):
        """Display welcome message."""
        welcome_text = """
# 🤖 Jarvis AI Assistant

Hello! I'm Jarvis, your intelligent AI assistant. I'm now listening for the wake word "Hey Jarvis" to activate.

**Features:**
- Voice-activated with wake word detection
- Natural language processing and responses
- Modular command execution (web search, automation, smart home, scheduling)
- Concurrent processing for smooth operation

**Commands you can try:**
- "Hey Jarvis, what's the weather like?"
- "Hey Jarvis, search for Python tutorials"
- "Hey Jarvis, schedule a reminder for tomorrow at 3 PM"

Say "Hey Jarvis" to get started!
        """
        print(welcome_text)  # Use print instead of console for simplicity

    def _handle_exit(self):
        """Handle application exit."""
        logger.info("Jarvis shutting down...")
        print("\nGoodbye! Jarvis signing off.")


def main():
    """Main entry point."""
    try:
        jarvis = Jarvis()
        asyncio.run(jarvis.run())
    except Exception as e:
        print(f"Fatal error: {e}")
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()