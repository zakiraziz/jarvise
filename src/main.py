"""
Jarvis AI Assistant - Enhanced with more features
Added: System monitoring, emotion detection, conversation history, multimodal support
"""

import pvporcupine
import sys
import os
import yaml
import logging
import asyncio
import threading
import queue
import json
import datetime
import psutil
import platform
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import time
import pickle
from collections import deque
import numpy as np

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from speech import SpeechRecognizer, WakeWordDetector
from tts import TextToSpeech
from ai import AIAssistant
from commands import CommandRegistry, WebSearchCommand, AutomationCommand, SmartHomeCommand, ScheduleCommand

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('jarvis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EmotionState(Enum):
    """Emotion states for Jarvis."""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    CONCERNED = "concerned"
    EXCITED = "excited"
    CALM = "calm"
    FOCUSED = "focused"

@dataclass
class ConversationEntry:
    """Single conversation entry."""
    id: str
    timestamp: datetime.datetime
    user_input: str
    ai_response: str
    context: Dict[str, Any]
    emotion: EmotionState
    command_executed: Optional[str] = None
    execution_result: Optional[str] = None

@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: datetime.datetime
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    temperature: Optional[float] = None

class MultiModalProcessor:
    """Process multimodal inputs (text, audio, eventually images)."""
    
    def __init__(self):
        self.active_modalities = set()
        
    async def process_image(self, image_path: str) -> Dict[str, Any]:
        """Process image input (placeholder for future implementation)."""
        return {"description": "Image processing not yet implemented", "objects": []}
    
    async def process_audio_emotion(self, audio_data: np.ndarray) -> EmotionState:
        """Analyze audio for emotional tone (placeholder)."""
        # In a real implementation, this would use ML models
        return EmotionState.NEUTRAL

class PersonalityManager:
    """Manage Jarvis's personality traits and responses."""
    
    def __init__(self):
        self.traits = {
            "formality": 0.7,  # 0=casual, 1=formal
            "humor": 0.5,      # 0=serious, 1=humorous
            "verbosity": 0.6,  # 0=concise, 1=verbose
            "initiative": 0.8, # 0=passive, 1=proactive
        }
        self.mood = EmotionState.NEUTRAL
        
    def adjust_trait(self, trait: str, adjustment: float):
        """Adjust a personality trait."""
        if trait in self.traits:
            self.traits[trait] = max(0.0, min(1.0, self.traits[trait] + adjustment))
            
    def get_response_style(self) -> Dict[str, Any]:
        """Get current response style based on personality and mood."""
        return {
            "greeting_style": "formal" if self.traits["formality"] > 0.7 else "casual",
            "include_humor": np.random.random() < self.traits["humor"],
            "response_length": "detailed" if self.traits["verbosity"] > 0.6 else "brief",
            "proactive_suggestions": np.random.random() < self.traits["initiative"]
        }

class ContextManager:
    """Manage conversation context and memory."""
    
    def __init__(self, max_history: int = 50):
        self.conversation_history = deque(maxlen=max_history)
        self.user_preferences = {}
        self.session_context = {
            "current_topic": None,
            "mentioned_entities": [],
            "active_tasks": [],
            "user_mood": "neutral"
        }
        
    def add_conversation_entry(self, entry: ConversationEntry):
        """Add a conversation to history."""
        self.conversation_history.append(entry)
        
    def get_recent_context(self, n: int = 5) -> List[ConversationEntry]:
        """Get recent conversation context."""
        return list(self.conversation_history)[-n:]
        
    def update_user_preference(self, key: str, value: Any):
        """Update user preferences."""
        self.user_preferences[key] = value
        
    def get_user_preference(self, key: str, default: Any = None) -> Any:
        """Get user preference."""
        return self.user_preferences.get(key, default)

class Jarvis:
    """Jarvis AI Assistant - Enhanced with more features."""

    def __init__(self, config_path: Optional[Path] = None):
        self.config = self._load_config(config_path)
        self._setup_components()

        # System state
        self.is_active = False
        self.is_listening = False
        self.session_id = str(uuid.uuid4())
        self.start_time = datetime.datetime.now()

        # Enhanced components
        self.multimodal_processor = MultiModalProcessor()
        self.personality_manager = PersonalityManager()
        self.context_manager = ContextManager()
        
        # Performance monitoring
        self.system_metrics_history = deque(maxlen=100)
        self.response_times = deque(maxlen=100)
        
        # Asyncio event loop
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        # Event queues for coordination
        self.input_queue = asyncio.Queue()
        self.decision_queue = asyncio.Queue()
        self.action_queue = asyncio.Queue()
        self.alert_queue = asyncio.Queue()

        # Threading for concurrent processing
        self.audio_thread = None
        self.processing_thread = None
        self.monitoring_thread = None
        self.shutdown_event = threading.Event()
        
        # Initialize system monitoring
        self._init_system_monitoring()

    def _load_config(self, config_path: Optional[Path]) -> dict:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'

        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            
            # Set environment variables from config
            for section, values in config.items():
                for key, value in values.items():
                    env_var = f"{section.upper()}_{key.upper()}"
                    if isinstance(value, str) and value.startswith("env:"):
                        config[section][key] = os.getenv(value[4:], value)
                        
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> dict:
        """Get default configuration."""
        return {
            'openai': {
                'api_key': os.getenv('OPENAI_API_KEY', 'your_key_here'),
                'model': 'gpt-4-turbo-preview',
                'temperature': 0.7,
                'max_tokens': 1500
            },
            'speech': {
                'picovoice_key': os.getenv('PICOVOICE_KEY', 'your_key_here'),
                'wake_word_sensitivity': 0.5,
                'wake_word': 'jarvis',
                'language': 'en-US'
            },
            'tts': {
                'voice_id': 0,
                'rate': 200,
                'volume': 1.0,
                'pitch': 100
            },
            'commands': {
                'smart_home_api_url': 'https://api.smarthome.com',
                'smart_home_api_key': 'your_key_here',
                'weather_api_key': os.getenv('WEATHER_API_KEY', ''),
                'news_api_key': os.getenv('NEWS_API_KEY', '')
            },
            'system': {
                'max_conversation_history': 50,
                'auto_save_interval': 300,  # seconds
                'performance_monitoring': True,
                'backup_enabled': True
            }
        }

    def _setup_components(self):
        """Initialize all Jarvis components."""
        # Voice interface
        self.speech_recognizer = SpeechRecognizer(
            access_key=self.config['speech']['picovoice_key'],
            sensitivity=self.config['speech']['wake_word_sensitivity'],
            keywords=[self.config['speech']['wake_word']]
        )
        self.tts = TextToSpeech(
            voice_id=self.config['tts']['voice_id'],
            rate=self.config['tts']['rate'],
            volume=self.config['tts']['volume'],
            pitch=self.config['tts']['pitch']
        )

        # AI/LLM with enhanced configuration
        self.ai = AIAssistant(
            api_key=self.config['openai']['api_key'],
            model=self.config['openai']['model'],
            temperature=self.config['openai']['temperature'],
            max_tokens=self.config['openai']['max_tokens']
        )

        # Command system
        self.command_registry = CommandRegistry()
        self._register_commands()
        
        # Load conversation history if exists
        self._load_conversation_history()

    def _init_system_monitoring(self):
        """Initialize system monitoring."""
        if self.config['system']['performance_monitoring']:
            self.monitoring_thread = threading.Thread(
                target=self._system_monitoring_loop,
                daemon=True
            )
            self.monitoring_thread.start()

    def _register_commands(self):
        """Register available commands."""
        # Core commands
        self.command_registry.register(WebSearchCommand())
        self.command_registry.register(AutomationCommand())
        self.command_registry.register(SmartHomeCommand(
            api_url=self.config['commands']['smart_home_api_url'],
            api_key=self.config['commands']['smart_home_api_key']
        ))
        self.command_registry.register(ScheduleCommand(self.command_registry))
        
        # Additional commands (you'll need to implement these)
        # self.command_registry.register(WeatherCommand(self.config['commands']['weather_api_key']))
        # self.command_registry.register(NewsCommand(self.config['commands']['news_api_key']))
        # self.command_registry.register(SystemCommand(self))

    def _load_conversation_history(self):
        """Load conversation history from file."""
        history_file = Path('conversation_history.pkl')
        if history_file.exists():
            try:
                with open(history_file, 'rb') as f:
                    history = pickle.load(f)
                    for entry in history:
                        self.context_manager.add_conversation_entry(entry)
                logger.info(f"Loaded {len(history)} conversation entries")
            except Exception as e:
                logger.error(f"Failed to load conversation history: {e}")

    def _save_conversation_history(self):
        """Save conversation history to file."""
        try:
            history_file = Path('conversation_history.pkl')
            with open(history_file, 'wb') as f:
                pickle.dump(list(self.context_manager.conversation_history), f)
            logger.info("Conversation history saved")
        except Exception as e:
            logger.error(f"Failed to save conversation history: {e}")

    async def run(self):
        """Main event-driven application loop."""
        logger.info(f"Starting Jarvis AI Assistant - Session: {self.session_id}")
        self._show_welcome()
        
        # Load user preferences
        self._load_user_preferences()

        # Start concurrent threads
        self._start_threads()
        
        # Start periodic tasks
        self._start_periodic_tasks()

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
                tasks = [
                    asyncio.wait_for(self.input_queue.get(), timeout=0.1),
                    asyncio.wait_for(self.alert_queue.get(), timeout=0.1)
                ]
                
                done, pending = await asyncio.wait(
                    tasks,
                    return_when=asyncio.FIRST_COMPLETED
                )

                for task in done:
                    try:
                        event = task.result()
                        await self._handle_event(event)
                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        logger.error(f"Error handling event: {e}")

                # Cancel pending tasks
                for task in pending:
                    task.cancel()

            except asyncio.TimeoutError:
                # No input events, continue monitoring
                continue
            except Exception as e:
                logger.error(f"Error in main event loop: {e}")

    async def _handle_event(self, event: Dict[str, Any]):
        """Handle various types of events."""
        event_type = event.get('type')

        handlers = {
            'wake_word_detected': self._handle_wake_word,
            'speech_recognized': self._handle_speech_input,
            'system_alert': self._handle_system_alert,
            'periodic_task': self._handle_periodic_task,
            'shutdown': self._handle_shutdown,
            'command_result': self._handle_command_result
        }

        handler = handlers.get(event_type)
        if handler:
            await handler(event)
        else:
            logger.warning(f"Unknown event type: {event_type}")

    async def _handle_wake_word(self, event: Optional[Dict] = None):
        """Handle wake word detection."""
        logger.info("Wake word detected - activating Jarvis")
        self.is_active = True
        
        # Get greeting based on time of day
        greeting = self._get_time_based_greeting()
        self.tts.speak(f"{greeting} How can I help you?")
        
        # Start listening for speech input
        self._start_speech_recognition()
        
        # Update context
        self.context_manager.session_context['last_activation'] = datetime.datetime.now()

    async def _handle_speech_input(self, event: Dict[str, Any]):
        """Handle recognized speech input."""
        if not self.is_active:
            return

        text = event.get('text', '').strip()
        if not text:
            return

        logger.info(f"Processing speech input: {text}")
        
        # Record start time for performance tracking
        start_time = time.time()

        # Add to conversation context
        self.context_manager.session_context['current_input'] = text
        
        # Process with emotion detection
        emotion = await self._detect_emotion(text)
        self.personality_manager.mood = emotion
        
        # Generate context-aware response
        context = self.context_manager.get_recent_context(3)
        enhanced_input = self._enhance_input_with_context(text, context)
        
        # Process through AI
        decision = await self.loop.run_in_executor(
            None, 
            self._process_with_ai, 
            enhanced_input, 
            context
        )
        
        # Record response time
        response_time = time.time() - start_time
        self.response_times.append(response_time)
        
        # Create conversation entry
        entry = ConversationEntry(
            id=str(uuid.uuid4()),
            timestamp=datetime.datetime.now(),
            user_input=text,
            ai_response=decision.get('response_text', ''),
            context=self.context_manager.session_context.copy(),
            emotion=emotion,
            command_executed=decision.get('action'),
            execution_result=None
        )
        
        self.context_manager.add_conversation_entry(entry)
        
        # Execute decision
        await self._execute_decision(decision, entry)
        
        # Check for proactive suggestions
        await self._check_proactive_suggestions()

    async def _detect_emotion(self, text: str) -> EmotionState:
        """Detect emotion from text input."""
        # Simple keyword-based emotion detection
        happy_keywords = ['good', 'great', 'awesome', 'happy', 'excited']
        concern_keywords = ['worried', 'concerned', 'problem', 'issue', 'help']
        
        text_lower = text.lower()
        
        if any(word in text_lower for word in happy_keywords):
            return EmotionState.HAPPY
        elif any(word in text_lower for word in concern_keywords):
            return EmotionState.CONCERNED
        elif '?' in text:
            return EmotionState.FOCUSED
            
        return EmotionState.NEUTRAL

    def _enhance_input_with_context(self, text: str, context: List[ConversationEntry]) -> str:
        """Enhance input with conversation context."""
        if not context:
            return text
            
        # Extract recent topics
        recent_topics = []
        for entry in context[-3:]:  # Last 3 entries
            if entry.user_input:
                recent_topics.append(entry.user_input[:50])
        
        if recent_topics:
            context_str = "Recent conversation topics: " + ", ".join(recent_topics)
            return f"{text}\n\nContext: {context_str}"
            
        return text

    def _process_with_ai(self, text: str, context: List[ConversationEntry]) -> Dict[str, Any]:
        """Process text with AI, considering context."""
        # Build context prompt
        context_prompt = ""
        if context:
            context_prompt = "\nRecent conversation:\n"
            for entry in context:
                context_prompt += f"User: {entry.user_input}\n"
                context_prompt += f"Jarvis: {entry.ai_response}\n"
        
        # Add personality traits
        personality = self.personality_manager.get_response_style()
        personality_prompt = f"\nRespond in a {personality['greeting_style']} style. "
        if personality['include_humor']:
            personality_prompt += "Include subtle humor if appropriate. "
        
        full_prompt = f"{context_prompt}\nCurrent input: {text}\n{personality_prompt}"
        
        return self.ai.decide_action(full_prompt)

    async def _execute_decision(self, decision: Dict[str, Any], entry: ConversationEntry):
        """Execute the decided action."""
        action_type = decision.get('action')
        params = decision.get('params', {})
        
        # Update entry with action
        entry.command_executed = action_type
        
        if action_type == 'command':
            result = await self._execute_command(params)
            entry.execution_result = str(result)
        elif action_type == 'respond':
            response = params.get('response_text', 'I heard you.')
            self.tts.speak(response)
            entry.ai_response = response
        elif action_type == 'search':
            result = await self._execute_search(params)
            entry.execution_result = str(result)
        elif action_type == 'schedule':
            result = await self._execute_schedule(params)
            entry.execution_result = str(result)
        elif action_type == 'system_info':
            await self._show_system_info()
        elif action_type == 'personality_adjust':
            await self._adjust_personality(params)
        
        # Deactivate after response (unless continuing conversation)
        if not decision.get('continue_conversation', False):
            self.is_active = False

    async def _execute_command(self, params: Dict[str, Any]):
        """Execute a modular command."""
        command_name = params.get('command_name', 'unknown')
        command_params = params.get('command_params', '')

        result = await self.loop.run_in_executor(
            None, self.command_registry.execute, command_name, command_params
        )

        if result:
            response = f"Command executed: {result}"
            self.tts.speak(response)
            return result
                async def _execute_search(self, params: Dict[str, Any]):
        """Execute web search."""
        query = params.get('query', '')
        result = await self.loop.run_in_executor(
            None, self.command_registry.execute, 'web_search', f'query={query}'
        )
        
        if result:
            # Summarize result for speech
            summary = result[:200] + "..." if len(result) > 200 else result
            self.tts.speak(f"Search result: {summary}")
            
        return result

    async def _execute_schedule(self, params: Dict[str, Any]):
        """Execute scheduling command."""
        task = params.get('task', '')
        time_str = params.get('time', 'now')
        result = await self.loop.run_in_executor(
            None, self.command_registry.execute, 'schedule',
            f'command=respond,command_params=response_text=Reminder: {task},time={time_str}'
        )
        
        if result:
            self.tts.speak(f"Scheduled: {result}")
            
        return result

    async def _show_system_info(self):
        """Display system information."""
        metrics = self._get_current_system_metrics()
        info = f"""
        System Status:
        - CPU Usage: {metrics.cpu_percent:.1f}%
        - Memory Usage: {metrics.memory_percent:.1f}%
        - Disk Usage: {metrics.disk_percent:.1f}%
        - Uptime: {datetime.datetime.now() - self.start_time}
        - Total Conversations: {len(self.context_manager.conversation_history)}
        """
        self.tts.speak(info)
        print(info)

    async def _adjust_personality(self, params: Dict[str, Any]):
        """Adjust Jarvis's personality."""
        trait = params.get('trait')
        adjustment = params.get('adjustment', 0.1)
        
        if trait in self.personality_manager.traits:
            self.personality_manager.adjust_trait(trait, adjustment)
            self.tts.speak(f"Adjusted {trait} by {adjustment}")
        else:
            self.tts.speak(f"Unknown trait: {trait}")

    async def _check_proactive_suggestions(self):
        """Check for proactive suggestions based on context."""
        if self.personality_manager.traits['initiative'] < 0.5:
            return
            
        # Example: Suggest weather if not checked today
        last_weather = self.context_manager.get_user_preference('last_weather_check')
        if not last_weather or (datetime.datetime.now() - last_weather).days > 0:
            if np.random.random() < 0.3:  # 30% chance
                self.tts.speak("By the way, would you like me to check today's weather?")

    async def _handle_system_alert(self, event: Dict[str, Any]):
        """Handle system alerts."""
        alert_type = event.get('alert_type')
        message = event.get('message', '')
        
        if alert_type == 'high_cpu':
            self.tts.speak(f"System alert: High CPU usage. {message}")
        elif alert_type == 'low_memory':
            self.tts.speak(f"System alert: Low memory. {message}")
        elif alert_type == 'scheduled_task':
            self.tts.speak(f"Reminder: {message}")

    async def _handle_periodic_task(self, event: Dict[str, Any]):
        """Handle periodic tasks."""
        task = event.get('task')
        
        if task == 'save_history':
            self._save_conversation_history()
        elif task == 'collect_metrics':
            await self._collect_system_metrics()
        elif task == 'backup':
            await self._create_backup()

    async def _handle_shutdown(self, event: Optional[Dict] = None):
        """Handle shutdown request."""
        self.shutdown_event.set()

    async def _handle_command_result(self, event: Dict[str, Any]):
        """Handle command execution results."""
        result = event.get('result')
        command = event.get('command')
        logger.info(f"Command '{command}' completed with result: {result}")

    def _start_periodic_tasks(self):
        """Start periodic background tasks."""
        async def periodic_saver():
            while not self.shutdown_event.is_set():
                await asyncio.sleep(self.config['system']['auto_save_interval'])
                await self.alert_queue.put({
                    'type': 'periodic_task',
                    'task': 'save_history'
                })
        
        async def metrics_collector():
            while not self.shutdown_event.is_set():
                await asyncio.sleep(60)  # Collect metrics every minute
                await self.alert_queue.put({
                    'type': 'periodic_task',
                    'task': 'collect_metrics'
                })
        
        # Start periodic tasks
        self.loop.create_task(periodic_saver())
        self.loop.create_task(metrics_collector())

    async def _collect_system_metrics(self):
        """Collect and store system metrics."""
        metrics = self._get_current_system_metrics()
        self.system_metrics_history.append(metrics)
        
        # Check for alerts
        if metrics.cpu_percent > 80:
            await self.alert_queue.put({
                'type': 'system_alert',
                'alert_type': 'high_cpu',
                'message': f'CPU at {metrics.cpu_percent:.1f}%'
            })
        if metrics.memory_percent > 85:
            await self.alert_queue.put({
                'type': 'system_alert',
                'alert_type': 'low_memory',
                'message': f'Memory at {metrics.memory_percent:.1f}%'
            })

    def _get_current_system_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        return SystemMetrics(
            timestamp=datetime.datetime.now(),
            cpu_percent=psutil.cpu_percent(interval=0.1),
            memory_percent=psutil.virtual_memory().percent,
            disk_percent=psutil.disk_usage('/').percent,
            network_bytes_sent=psutil.net_io_counters().bytes_sent,
            network_bytes_recv=psutil.net_io_counters().bytes_recv
        )

    def _system_monitoring_loop(self):
        """Background thread for system monitoring."""
        while not self.shutdown_event.is_set():
            try:
                # Monitor various system aspects
                self._monitor_performance()
                time.sleep(5)  # Check every 5 seconds
            except Exception as e:
                logger.error(f"System monitoring error: {e}")

    def _monitor_performance(self):
        """Monitor system performance."""
        # Calculate average response time
        if self.response_times:
            avg_response = sum(self.response_times) / len(self.response_times)
            if avg_response > 5.0:  # Alert if average response > 5 seconds
                asyncio.run_coroutine_threadsafe(
                    self.alert_queue.put({
                        'type': 'system_alert',
                        'alert_type': 'slow_response',
                        'message': f'Average response time: {avg_response:.1f}s'
                    }),
                    self.loop
                )

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
        # Start continuous listening
        self.speech_recognizer.start_continuous_listening(
            callback=lambda text: asyncio.run_coroutine_threadsafe(
                self.input_queue.put({'type': 'speech_recognized', 'text': text}),
                self.loop
            )
        )

    def _audio_processing_loop(self):
        """Background thread for processing audio frames."""
        while not self.shutdown_event.is_set():
            try:
                # Process audio frames
                if self.is_active and self.is_listening:
                    # Continuous listening is already handled
                    pass
                else:
                    # Wake word detection
                    frame = self.speech_recognizer.get_audio_frame()
                    if frame is not None:
                        detected = self.speech_recognizer.process_audio_frame(frame)
                        if detected and not self.is_active:
                            asyncio.run_coroutine_threadsafe(
                                self.input_queue.put({'type': 'wake_word_detected'}),
                                self.loop
                            )
                time.sleep(0.05)  # 20ms delay for ~50 FPS
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
            
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2)

        self.speech_recognizer.stop_wake_word_detection()
        self.speech_recognizer.stop_audio_stream()
        self.tts.stop()
        
        # Save final state
        self._save_conversation_history()
        self._save_user_preferences()

    def _get_time_based_greeting(self) -> str:
        """Get greeting based on time of day."""
        hour = datetime.datetime.now().hour
        
        if 5 <= hour < 12:
            return "Good morning"
        elif 12 <= hour < 17:
            return "Good afternoon"
        elif 17 <= hour < 22:
            return "Good evening"
        else:
            return "Hello"

    def _load_user_preferences(self):
        """Load user preferences from file."""
        pref_file = Path('user_preferences.json')
        if pref_file.exists():
            try:
                with open(pref_file, 'r') as f:
                    self.context_manager.user_preferences = json.load(f)
                logger.info("User preferences loaded")
            except Exception as e:
                logger.error(f"Failed to load user preferences: {e}")

    def _save_user_preferences(self):
        """Save user preferences to file."""
        try:
            pref_file = Path('user_preferences.json')
            with open(pref_file, 'w') as f:
                json.dump(self.context_manager.user_preferences, f, indent=2)
            logger.info("User preferences saved")
        except Exception as e:
            logger.error(f"Failed to save user preferences: {e}")

    async def _create_backup(self):
        """Create backup of important data."""
        if not self.config['system']['backup_enabled']:
            return
            
        try:
            backup_dir = Path('backups')
            backup_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = backup_dir / f"jarvis_backup_{timestamp}.zip"
            
            # In a real implementation, this would create a zip file
            # with conversation history, preferences, and config
            
            logger.info(f"Backup created: {backup_file}")
        except Exception as e:
            logger.error(f"Backup failed: {e}")

    def _show_welcome(self):
        """Display enhanced welcome message."""
        welcome_text = f"""
╔══════════════════════════════════════════════════════════════╗
║                    🤖 Jarvis AI Assistant                    ║
║                    Session: {self.session_id[:8]}...                    ║
╚══════════════════════════════════════════════════════════════╝

Hello! I'm Jarvis, your intelligent AI assistant. 
I'm now listening for the wake word "{self.config['speech']['wake_word']}" to activate.

📊 **Current Status:**
- System: {platform.system()} {platform.release()}
- Python: {platform.python_version()}
- Personality: {self.personality_manager.traits}
- Mood: {self.personality_manager.mood.value}

🚀 **Enhanced Features:**
• Emotion-aware responses
• Conversation memory & context
• Personality customization
• System performance monitoring
• Proactive suggestions
• Multimodal support (coming soon)

💡 **Commands you can try:**
- "Hey Jarvis, how's the system doing?"
- "Hey Jarvis, search for AI news"
- "Hey Jarvis, schedule a meeting tomorrow at 2 PM"
- "Hey Jarvis, be more casual in your responses"
- "Hey Jarvis, what were we talking about earlier?"

Say "{self.config['speech']['wake_word']}" to get started!
        """
        print(welcome_text)

    def _handle_exit(self):
        """Handle application exit."""
        logger.info("Jarvis shutting down...")
        
        # Show session summary
        duration = datetime.datetime.now() - self.start_time
        summary = f"""
╔══════════════════════════════════════════════════════════════╗
║                     Session Summary                          ║
╚══════════════════════════════════════════════════════════════╝
• Duration: {duration}
• Conversations: {len(self.context_manager.conversation_history)}
• Average Response Time: {
    sum(self.response_times)/len(self.response_times) if self.response_times else 0:.2f}s
• Final Mood: {self.personality_manager.mood.value}
        
Goodbye! Jarvis signing off.
        """
        print(summary)


def main():
    """Main entry point."""
    try:
        jarvis = Jarvis()
        asyncio.run(jarvis.run())
    except KeyboardInterrupt:
        print("\n\nShutdown initiated by user.")
    except Exception as e:
        print(f"Fatal error: {e}")
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
        return None

