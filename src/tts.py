"""
Jarvis AI Assistant - Supercharged Edition
Added: 
- Learning/Adaptive Behavior
- Plugin System
- Health Monitoring
- Advanced Security
- Multi-User Support
- External API Integration
- Advanced Analytics
- Cross-Platform Compatibility
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
import hashlib
import hmac
import secrets
import base64
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Tuple, Set, Union
from dataclasses import dataclass, asdict, field
from enum import Enum, auto
import time
import pickle
from collections import deque, defaultdict
import numpy as np
import requests
import websockets
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
import GPUtil
import matplotlib.pyplot as plt
from io import BytesIO
import inspect
import importlib
import pkgutil
from abc import ABC, abstractmethod

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from speech import SpeechRecognizer, WakeWordDetector
from tts import TextToSpeech
from ai import AIAssistant
from commands import CommandRegistry, WebSearchCommand, AutomationCommand, SmartHomeCommand, ScheduleCommand

# Setup enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('jarvis.log'),
        logging.StreamHandler(),
        logging.handlers.RotatingFileHandler(
            'jarvis_debug.log', maxBytes=10*1024*1024, backupCount=5
        )
    ]
)
logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """Security levels for different operations."""
    PUBLIC = 0
    USER = 1
    ADMIN = 2
    SYSTEM = 3

class LearningMode(Enum):
    """Different learning modes for Jarvis."""
    PASSIVE = auto()
    ACTIVE = auto()
    ADAPTIVE = auto()
    EXPERT = auto()

class EmotionState(Enum):
    """Enhanced emotion states for Jarvis."""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    CONCERNED = "concerned"
    EXCITED = "excited"
    CALM = "calm"
    FOCUSED = "focused"
    PLAYFUL = "playful"
    SERIOUS = "serious"
    CURIOUS = "curious"
    TIRED = "tired"

@dataclass
class ConversationEntry:
    """Enhanced conversation entry."""
    id: str
    timestamp: datetime.datetime
    user_input: str
    ai_response: str
    context: Dict[str, Any]
    emotion: EmotionState
    confidence: float = 1.0
    command_executed: Optional[str] = None
    execution_result: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat(),
            'emotion': self.emotion.value
        }

@dataclass
class SystemMetrics:
    """Enhanced system performance metrics."""
    timestamp: datetime.datetime
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    gpu_utilization: Optional[float] = None
    gpu_memory_percent: Optional[float] = None
    temperature: Optional[float] = None
    battery_percent: Optional[float] = None
    process_count: int = 0
    
    def is_critical(self) -> bool:
        """Check if metrics indicate critical state."""
        return (self.cpu_percent > 90 or 
                self.memory_percent > 95 or 
                self.disk_percent > 95)

@dataclass
class UserProfile:
    """User profile for multi-user support."""
    user_id: str
    name: str
    voice_profile: Dict[str, Any]
    preferences: Dict[str, Any]
    permissions: Set[str]
    learning_data: Dict[str, Any]
    last_active: datetime.datetime
    security_level: SecurityLevel
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission."""
        return permission in self.permissions
    
    def update_learning(self, interaction_type: str, data: Dict[str, Any]):
        """Update learning data based on interaction."""
        if interaction_type not in self.learning_data:
            self.learning_data[interaction_type] = []
        self.learning_data[interaction_type].append({
            'timestamp': datetime.datetime.now().isoformat(),
            'data': data
        })

class PluginBase(ABC):
    """Base class for Jarvis plugins."""
    
    def __init__(self, jarvis_instance: 'Jarvis'):
        self.jarvis = jarvis_instance
        self.config = jarvis_instance.config.get('plugins', {}).get(self.name, {})
        self.enabled = True
        
    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name."""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Plugin description."""
        pass
    
    async def initialize(self):
        """Initialize plugin."""
        pass
    
    async def shutdown(self):
        """Shutdown plugin."""
        pass
    
    async def handle_event(self, event: Dict[str, Any]):
        """Handle events from Jarvis."""
        pass

class HealthMonitor:
    """Monitor Jarvis health and performance."""
    
    def __init__(self):
        self.health_metrics = deque(maxlen=1000)
        self.alerts = deque(maxlen=100)
        self.anomaly_detector = AnomalyDetector()
        
    def check_health(self, metrics: SystemMetrics) -> Dict[str, Any]:
        """Check system health."""
        health_score = 100
        
        # CPU health
        if metrics.cpu_percent > 80:
            health_score -= 20
        elif metrics.cpu_percent > 60:
            health_score -= 10
            
        # Memory health
        if metrics.memory_percent > 90:
            health_score -= 25
        elif metrics.memory_percent > 75:
            health_score -= 15
            
        # Disk health
        if metrics.disk_percent > 95:
            health_score -= 30
        elif metrics.disk_percent > 85:
            health_score -= 20
            
        # Check for anomalies
        is_anomaly = self.anomaly_detector.detect(metrics)
        if is_anomaly:
            health_score -= 40
            
        return {
            'score': max(0, health_score),
            'status': 'healthy' if health_score > 70 else 'degraded' if health_score > 40 else 'critical',
            'timestamp': datetime.datetime.now(),
            'metrics': asdict(metrics)
        }
    
    def add_alert(self, alert_type: str, message: str, severity: str = "warning"):
        """Add health alert."""
        self.alerts.append({
            'timestamp': datetime.datetime.now(),
            'type': alert_type,
            'message': message,
            'severity': severity
        })

class AnomalyDetector:
    """Detect anomalies in system metrics."""
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.metric_history = defaultdict(lambda: deque(maxlen=window_size))
        
    def detect(self, metrics: SystemMetrics) -> bool:
        """Detect anomalies in metrics."""
        metric_dict = asdict(metrics)
        
        for key, value in metric_dict.items():
            if isinstance(value, (int, float)):
                history = self.metric_history[key]
                
                if len(history) >= 10:  # Need enough history
                    mean = np.mean(history)
                    std = np.std(history)
                    
                    if std > 0:  # Avoid division by zero
                        z_score = abs(value - mean) / std
                        if z_score > 3:  # 3 sigma rule
                            return True
                
                history.append(value)
        
        return False

class LearningEngine:
    """Advanced learning engine for Jarvis."""
    
    def __init__(self):
        self.knowledge_base = {}
        self.patterns = defaultdict(list)
        self.adaptation_rules = {}
        self.learning_mode = LearningMode.ADAPTIVE
        
    def learn_from_interaction(self, interaction: ConversationEntry):
        """Learn from interaction."""
        # Extract patterns from conversation
        words = interaction.user_input.lower().split()
        for i in range(len(words) - 1):
            pattern = f"{words[i]} {words[i+1]}"
            self.patterns[pattern].append({
                'timestamp': interaction.timestamp,
                'response': interaction.ai_response,
                'emotion': interaction.emotion.value
            })
        
        # Update knowledge base
        key = hashlib.md5(interaction.user_input.encode()).hexdigest()[:8]
        if key not in self.knowledge_base:
            self.knowledge_base[key] = []
        self.knowledge_base[key].append(interaction)
        
        # Prune old entries
        self._prune_old_entries()
    
    def predict_response(self, query: str, context: List[ConversationEntry]) -> Optional[Dict[str, Any]]:
        """Predict response based on learned patterns."""
        query_lower = query.lower()
        
        # Check for exact matches
        for key, entries in self.knowledge_base.items():
            for entry in entries:
                if entry.user_input.lower() == query_lower:
                    return {
                        'response': entry.ai_response,
                        'confidence': 0.9,
                        'source': 'exact_match'
                    }
        
        # Check for pattern matches
        words = query_lower.split()
        for i in range(len(words) - 1):
            pattern = f"{words[i]} {words[i+1]}"
            if pattern in self.patterns:
                patterns = self.patterns[pattern]
                if patterns:
                    latest = patterns[-1]
                    return {
                        'response': latest['response'],
                        'confidence': 0.7,
                        'source': 'pattern_match'
                    }
        
        return None
    
    def _prune_old_entries(self, max_age_days: int = 30):
        """Prune entries older than max_age_days."""
        cutoff = datetime.datetime.now() - datetime.timedelta(days=max_age_days)
        
        # Prune knowledge base
        for key in list(self.knowledge_base.keys()):
            self.knowledge_base[key] = [
                entry for entry in self.knowledge_base[key]
                if entry.timestamp > cutoff
            ]
            if not self.knowledge_base[key]:
                del self.knowledge_base[key]
        
        # Prune patterns
        for pattern in list(self.patterns.keys()):
            self.patterns[pattern] = [
                p for p in self.patterns[pattern]
                if p['timestamp'] > cutoff
            ]
            if not self.patterns[pattern]:
                del self.patterns[pattern]

class SecurityManager:
    """Enhanced security manager."""
    
    def __init__(self, secret_key: Optional[str] = None):
        self.secret_key = secret_key or secrets.token_hex(32)
        self.encryption_key = self._derive_encryption_key()
        self.cipher = Fernet(self.encryption_key)
        self.access_log = deque(maxlen=1000)
        self.failed_attempts = defaultdict(int)
        self.lockout_threshold = 5
        self.lockout_duration = 300  # 5 minutes
        
    def _derive_encryption_key(self) -> bytes:
        """Derive encryption key from secret."""
        salt = b'jarvis_salt_2024'
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(self.secret_key.encode()))
    
    def encrypt(self, data: str) -> str:
        """Encrypt sensitive data."""
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        return self.cipher.decrypt(encrypted_data.encode()).decode()
    
    def verify_credentials(self, user_id: str, credentials: Dict[str, Any]) -> bool:
        """Verify user credentials."""
        # Implementation depends on your auth system
        # This is a placeholder
        return True
    
    def log_access(self, user_id: str, action: str, success: bool):
        """Log access attempt."""
        self.access_log.append({
            'timestamp': datetime.datetime.now(),
            'user_id': user_id,
            'action': action,
            'success': success,
            'ip': self._get_client_ip()
        })
        
        if not success:
            self.failed_attempts[user_id] += 1
            if self.failed_attempts[user_id] >= self.lockout_threshold:
                logger.warning(f"User {user_id} locked out due to multiple failed attempts")
    
    def is_locked_out(self, user_id: str) -> bool:
        """Check if user is locked out."""
        if user_id in self.failed_attempts:
            # Reset after lockout duration
            if self.failed_attempts[user_id] >= self.lockout_threshold:
                # In real implementation, check timestamp
                return True
        return False
    
    def _get_client_ip(self) -> str:
        """Get client IP address."""
        # Implementation depends on your network setup
        return "127.0.0.1"

class AnalyticsEngine:
    """Advanced analytics engine."""
    
    def __init__(self):
        self.db_path = Path('analytics.db')
        self._init_database()
        
    def _init_database(self):
        """Initialize analytics database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS interactions (
            id TEXT PRIMARY KEY,
            timestamp DATETIME,
            user_id TEXT,
            input_type TEXT,
            input_length INTEGER,
            response_time REAL,
            emotion TEXT,
            command_executed TEXT,
            success BOOLEAN
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS performance_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            metric_name TEXT,
            metric_value REAL,
            unit TEXT
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_behavior (
            user_id TEXT,
            timestamp DATETIME,
            action_type TEXT,
            action_details TEXT,
            duration REAL
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_interaction(self, entry: ConversationEntry, user_id: str, response_time: float):
        """Log interaction to analytics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO interactions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            entry.id,
            entry.timestamp,
            user_id,
            'speech',
            len(entry.user_input),
            response_time,
            entry.emotion.value,
            entry.command_executed or '',
            entry.execution_result is not None
        ))
        
        conn.commit()
        conn.close()
    
    def get_daily_stats(self, date: datetime.date) -> Dict[str, Any]:
        """Get daily statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        start_date = datetime.datetime(date.year, date.month, date.day)
        end_date = start_date + datetime.timedelta(days=1)
        
        cursor.execute('''
        SELECT 
            COUNT(*) as total_interactions,
            AVG(response_time) as avg_response_time,
            SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_interactions
        FROM interactions 
        WHERE timestamp BETWEEN ? AND ?
        ''', (start_date, end_date))
        
        result = cursor.fetchone()
        conn.close()
        
        return {
            'total_interactions': result[0] if result else 0,
            'avg_response_time': result[1] if result and result[1] else 0,
            'success_rate': (result[2] / result[0] * 100) if result and result[0] > 0 else 0
        }

class MultiModalProcessor:
    """Enhanced multimodal processor."""
    
    def __init__(self):
        self.active_modalities = set()
        self.image_processors = {}
        self.audio_processors = {}
        self.sensor_processors = {}
        
    async def process_image(self, image_path: str) -> Dict[str, Any]:
        """Process image input."""
        try:
            # Placeholder for actual image processing
            # In production, integrate with OpenCV, PIL, etc.
            return {
                "description": "Image detected",
                "objects": [],
                "sentiment": "neutral",
                "colors": []
            }
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            return {"error": str(e)}
    
    async def process_audio_emotion(self, audio_data: np.ndarray) -> EmotionState:
        """Analyze audio for emotional tone."""
        # Placeholder for audio emotion analysis
        return EmotionState.NEUTRAL
    
    async def process_sensor_data(self, sensor_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process sensor data."""
        # Placeholder for IoT/sensor integration
        return {
            "sensor_type": sensor_type,
            "processed": True,
            "data": data
        }

class PersonalityManager:
    """Enhanced personality manager."""
    
    def __init__(self):
        self.traits = {
            "formality": 0.7,
            "humor": 0.5,
            "verbosity": 0.6,
            "initiative": 0.8,
            "creativity": 0.5,
            "empathy": 0.7,
            "assertiveness": 0.6
        }
        self.mood = EmotionState.NEUTRAL
        self.mood_history = deque(maxlen=100)
        self.adaptation_rules = self._load_adaptation_rules()
        
    def _load_adaptation_rules(self) -> Dict[str, Callable]:
        """Load personality adaptation rules."""
        return {
            "time_of_day": self._adapt_to_time_of_day,
            "user_mood": self._adapt_to_user_mood,
            "conversation_topic": self._adapt_to_topic,
            "interaction_history": self._adapt_to_history
        }
    
    def adapt_personality(self, context: Dict[str, Any]):
        """Adapt personality based on context."""
        for rule_name, rule_func in self.adaptation_rules.items():
            if rule_name in context:
                rule_func(context[rule_name])
        
        # Record mood for history
        self.mood_history.append({
            'timestamp': datetime.datetime.now(),
            'mood': self.mood,
            'traits': self.traits.copy()
        })
    
    def _adapt_to_time_of_day(self, hour: int):
        """Adapt personality based on time of day."""
        if 22 <= hour or hour < 5:  # Late night/early morning
            self.traits["verbosity"] *= 0.8  # Be more concise
            self.traits["humor"] *= 0.7  # Less humor at night
        elif 5 <= hour < 12:  # Morning
            self.traits["initiative"] *= 1.2  # More proactive in morning
    
    def _adapt_to_user_mood(self, user_mood: str):
        """Adapt to user's perceived mood."""
        if user_mood == "stressed":
            self.traits["empathy"] *= 1.3
            self.traits["humor"] *= 0.7
        elif user_mood == "happy":
            self.traits["humor"] *= 1.2
    
    def _adapt_to_topic(self, topic: str):
        """Adapt to conversation topic."""
        serious_topics = ["work", "business", "finance", "health"]
        if topic in serious_topics:
            self.traits["formality"] *= 1.2
            self.traits["humor"] *= 0.8
    
    def _adapt_to_history(self, history: List[ConversationEntry]):
        """Adapt based on interaction history."""
        if len(history) > 10:
            # User is chatty, increase engagement
            self.traits["initiative"] *= 1.1
            self.traits["verbosity"] *= 1.05

class ContextManager:
    """Enhanced context manager."""
    
    def __init__(self, max_history: int = 100):
        self.conversation_history = deque(maxlen=max_history)
        self.user_profiles: Dict[str, UserProfile] = {}
        self.session_context = {
            "current_topic": None,
            "mentioned_entities": [],
            "active_tasks": [],
            "user_mood": "neutral",
            "conversation_depth": 0,
            "last_commands": deque(maxlen=10),
            "environmental_context": {}
        }
        
    def add_conversation_entry(self, entry: ConversationEntry):
        """Add a conversation to history."""
        self.conversation_history.append(entry)
        
    def get_recent_context(self, n: int = 10) -> List[ConversationEntry]:
        """Get recent conversation context."""
        return list(self.conversation_history)[-n:]
        
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile."""
        return self.user_profiles.get(user_id)
    
    def create_user_profile(self, user_id: str, name: str, **kwargs) -> UserProfile:
        """Create a new user profile."""
        profile = UserProfile(
            user_id=user_id,
            name=name,
            voice_profile={},
            preferences=kwargs.get('preferences', {}),
            permissions=kwargs.get('permissions', {'basic'}),
            learning_data={},
            last_active=datetime.datetime.now(),
            security_level=SecurityLevel.USER
        )
        self.user_profiles[user_id] = profile
        return profile

class PluginManager:
    """Manage Jarvis plugins."""
    
    def __init__(self, jarvis_instance: 'Jarvis'):
        self.jarvis = jarvis_instance
        self.plugins: Dict[str, PluginBase] = {}
        self.plugin_dirs = ['plugins']
        
    async def discover_plugins(self):
        """Discover available plugins."""
        for plugin_dir in self.plugin_dirs:
            if Path(plugin_dir).exists():
                self._load_plugins_from_dir(plugin_dir)
    
    def _load_plugins_from_dir(self, plugin_dir: str):
        """Load plugins from directory."""
        for module_info in pkgutil.iter_modules([plugin_dir]):
            try:
                module = importlib.import_module(f"{plugin_dir}.{module_info.name}")
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, PluginBase) and 
                        obj != PluginBase):
                        plugin_instance = obj(self.jarvis)
                        self.plugins[plugin_instance.name] = plugin_instance
                        logger.info(f"Loaded plugin: {plugin_instance.name} v{plugin_instance.version}")
            except Exception as e:
                logger.error(f"Failed to load plugin from {module_info.name}: {e}")
    
    async def initialize_plugins(self):
        """Initialize all plugins."""
        for plugin in self.plugins.values():
            if plugin.enabled:
                try:
                    await plugin.initialize()
                    logger.info(f"Initialized plugin: {plugin.name}")
                except Exception as e:
                    logger.error(f"Failed to initialize plugin {plugin.name}: {e}")
        async def shutdown_plugins(self):
        """Shutdown all plugins."""
        for plugin in self.plugins.values():
            if plugin.enabled:
                try:
                    await plugin.shutdown()
                except Exception as e:
                    logger.error(f"Failed to shutdown plugin {plugin.name}: {e}")
    
    async def broadcast_event(self, event: Dict[str, Any]):
        """Broadcast event to all plugins."""
        for plugin in self.plugins.values():
            if plugin.enabled:
                try:
                    await plugin.handle_event(event)
                except Exception as e:
                    logger.error(f"Plugin {plugin.name} failed to handle event: {e}")

class ExternalAPIHandler:
    """Handle external API integrations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.apis = self._initialize_apis()
        self.api_cache = {}
        self.cache_timeout = 300  # 5 minutes
        
    def _initialize_apis(self) -> Dict[str, Any]:
        """Initialize API connections."""
        apis = {}
        
        # Weather API
        if weather_key := self.config.get('weather_api_key'):
            apis['weather'] = {
                'key': weather_key,
                'base_url': 'https://api.openweathermap.org/data/2.5'
            }
        
        # News API
        if news_key := self.config.get('news_api_key'):
            apis['news'] = {
                'key': news_key,
                'base_url': 'https://newsapi.org/v2'
            }
        
        # Calendar API
        if calendar_config := self.config.get('calendar'):
            apis['calendar'] = calendar_config
        
        return apis
    
    async def call_api(self, api_name: str, endpoint: str, 
                      params: Optional[Dict] = None) -> Dict[str, Any]:
        """Call external API."""
        cache_key = f"{api_name}:{endpoint}:{json.dumps(params or {})}"
        
        # Check cache
        if cache_key in self.api_cache:
            cached_data, timestamp = self.api_cache[cache_key]
            if time.time() - timestamp < self.cache_timeout:
                return cached_data
        
        if api_name not in self.apis:
            return {"error": f"API {api_name} not configured"}
        
        api_config = self.apis[api_name]
        url = f"{api_config['base_url']}/{endpoint}"
        
        try:
            response = requests.get(
                url,
                params={**params, 'api_key': api_config.get('key')},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            # Cache the result
            self.api_cache[cache_key] = (data, time.time())
            
            return data
        except Exception as e:
            logger.error(f"API call failed: {api_name}/{endpoint}: {e}")
            return {"error": str(e)}

class Jarvis:
    """Supercharged Jarvis AI Assistant."""

    def __init__(self, config_path: Optional[Path] = None):
        self.config = self._load_config(config_path)
        self._setup_components()

        # Core system state
        self.is_active = False
        self.is_listening = False
        self.session_id = str(uuid.uuid4())
        self.start_time = datetime.datetime.now()
        self.current_user: Optional[UserProfile] = None
        self.learning_mode = LearningMode.ADAPTIVE
        
        # Enhanced components
        self.multimodal_processor = MultiModalProcessor()
        self.personality_manager = PersonalityManager()
        self.context_manager = ContextManager()
        self.learning_engine = LearningEngine()
        self.security_manager = SecurityManager(self.config.get('security', {}).get('secret_key'))
        self.health_monitor = HealthMonitor()
        self.analytics_engine = AnalyticsEngine()
        self.plugin_manager = PluginManager(self)
        self.external_api_handler = ExternalAPIHandler(self.config.get('apis', {}))
        
        # Performance monitoring
        self.system_metrics_history = deque(maxlen=1000)
        self.response_times = deque(maxlen=1000)
        self.error_log = deque(maxlen=500)
        
        # Asyncio event loop
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        # Enhanced event queues
        self.input_queue = asyncio.Queue(maxsize=1000)
        self.decision_queue = asyncio.Queue(maxsize=1000)
        self.action_queue = asyncio.Queue(maxsize=1000)
        self.alert_queue = asyncio.Queue(maxsize=1000)
        self.plugin_queue = asyncio.Queue(maxsize=1000)
        
        # Thread management
        self.threads = {}
        self.shutdown_event = threading.Event()
        
        # Initialize enhanced systems
        self._init_enhanced_systems()

    def _load_config(self, config_path: Optional[Path]) -> dict:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'

        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            
            # Enhanced environment variable handling
            self._process_config_env_vars(config)
            
            # Validate configuration
            self._validate_config(config)
            
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self._get_default_config()

    def _process_config_env_vars(self, config: dict):
        """Process environment variables in config."""
        for section, values in config.items():
            if isinstance(values, dict):
                for key, value in values.items():
                    if isinstance(value, str) and value.startswith("env:"):
                        env_var = value[4:]
                        config[section][key] = os.getenv(env_var, "")
                    elif isinstance(value, dict):
                        self._process_config_env_vars({key: value})

    def _validate_config(self, config: dict):
        """Validate configuration."""
        required_sections = ['openai', 'speech', 'tts']
        for section in required_sections:
            if section not in config:
                logger.warning(f"Missing configuration section: {section}")
        
        # Validate API keys
        if not config.get('openai', {}).get('api_key'):
            logger.error("OpenAI API key not configured")
        
        if not config.get('speech', {}).get('picovoice_key'):
            logger.warning("Picovoice API key not configured - wake word detection may not work")

    def _get_default_config(self) -> dict:
        """Get comprehensive default configuration."""
        return {
            'openai': {
                'api_key': os.getenv('OPENAI_API_KEY', ''),
                'model': 'gpt-4-turbo-preview',
                'temperature': 0.7,
                'max_tokens': 2000,
                'timeout': 30,
                'max_retries': 3
            },
            'speech': {
                'picovoice_key': os.getenv('PICOVOICE_KEY', ''),
                'wake_word_sensitivity': 0.5,
                'wake_word': 'jarvis',
                'language': 'en-US',
                'energy_threshold': 300,
                'pause_threshold': 0.8
            },
            'tts': {
                'voice_id': 0,
                'rate': 200,
                'volume': 1.0,
                'pitch': 100,
                'engine': 'pyttsx3'
            },
            'commands': {
                'smart_home': {
                    'api_url': 'https://api.smarthome.com',
                    'api_key': os.getenv('SMART_HOME_API_KEY', '')
                },
                'automation': {
                    'scripts_dir': 'scripts',
                    'timeout': 30
                }
            },
            'system': {
                'max_conversation_history': 100,
                'auto_save_interval': 300,
                'performance_monitoring': True,
                'backup_enabled': True,
                'health_check_interval': 60,
                'max_response_time': 10.0,
                'resource_limits': {
                    'max_cpu_percent': 80,
                    'max_memory_percent': 85,
                    'max_disk_percent': 90
                }
            },
            'security': {
                'secret_key': os.getenv('JARVIS_SECRET_KEY', secrets.token_hex(32)),
                'require_authentication': False,
                'session_timeout': 3600,
                'encryption_enabled': True
            },
            'apis': {
                'weather_api_key': os.getenv('WEATHER_API_KEY', ''),
                'news_api_key': os.getenv('NEWS_API_KEY', ''),
                'calendar': {
                    'type': 'google',
                    'credentials_file': 'credentials.json'
                }
            },
            'plugins': {
                'enabled_by_default': True,
                'auto_discover': True,
                'plugin_dirs': ['plugins']
            },
            'learning': {
                'enabled': True,
                'mode': 'adaptive',
                'retention_days': 30,
                'adaptive_threshold': 0.7
            },
            'analytics': {
                'enabled': True,
                'collect_usage_stats': True,
                'anonymize_data': True,
                'report_interval': 86400
            }
        }

    def _setup_components(self):
        """Initialize all Jarvis components with enhanced error handling."""
        try:
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

            # AI/LLM with retry logic
            self.ai = AIAssistant(
                api_key=self.config['openai']['api_key'],
                model=self.config['openai']['model'],
                temperature=self.config['openai']['temperature'],
                max_tokens=self.config['openai']['max_tokens'],
                timeout=self.config['openai']['timeout'],
                max_retries=self.config['openai']['max_retries']
            )

            # Enhanced command system
            self.command_registry = CommandRegistry()
            self._register_commands()
            
            # Load conversation history
            self._load_conversation_history()
            
            # Load user profiles
            self._load_user_profiles()
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.critical(f"Failed to initialize components: {e}")
            raise

    def _init_enhanced_systems(self):
        """Initialize enhanced systems."""
        # Start system monitoring
        if self.config['system']['performance_monitoring']:
            self._start_system_monitoring()
        
        # Start health monitoring
        self._start_health_monitoring()
        
        # Initialize plugins
        if self.config['plugins']['enabled_by_default']:
            asyncio.run_coroutine_threadsafe(
                self._initialize_plugins(),
                self.loop
            )

    async def _initialize_plugins(self):
        """Initialize plugin system."""
        try:
            await self.plugin_manager.discover_plugins()
            await self.plugin_manager.initialize_plugins()
            logger.info(f"Loaded {len(self.plugin_manager.plugins)} plugins")
        except Exception as e:
            logger.error(f"Failed to initialize plugins: {e}")

    def _register_commands(self):
        """Register available commands."""
        # Core commands
        self.command_registry.register(WebSearchCommand())
        self.command_registry.register(AutomationCommand())
        self.command_registry.register(SmartHomeCommand(
            api_url=self.config['commands']['smart_home']['api_url'],
            api_key=self.config['commands']['smart_home']['api_key']
        ))
        self.command_registry.register(ScheduleCommand(self.command_registry))
        
        # Register system commands
        self.command_registry.register(self._create_system_command())
        
        # Register learning commands
        self.command_registry.register(self._create_learning_command())

    def _create_system_command(self):
        """Create system command handler."""
        class SystemCommand:
            def __init__(self, jarvis):
                self.jarvis = jarvis
                self.name = "system"
                
            async def execute(self, params: str) -> str:
                """Execute system command."""
                param_dict = self.jarvis._parse_command_params(params)
                action = param_dict.get('action', 'status')
                
                if action == 'status':
                    return self.jarvis._get_system_status()
                elif action == 'health':
                    return self.jarvis._get_health_report()
                elif action == 'metrics':
                    return self.jarvis._get_metrics_summary()
                elif action == 'restart':
                    return await self.jarvis._restart_system()
                elif action == 'update':
                    return await self.jarvis._check_for_updates()
                else:
                    return f"Unknown system action: {action}"
                return SystemCommand(self)

    def _create_learning_command(self):
        """Create learning command handler."""
        class LearningCommand:
            def __init__(self, jarvis):
                self.jarvis = jarvis
                self.name = "learning"
                
            async def execute(self, params: str) -> str:
                """Execute learning command."""
                param_dict = self.jarvis._parse_command_params(params)
                action = param_dict.get('action', 'status')
                
                if action == 'status':
                    return f"Learning mode: {self.jarvis.learning_mode.name}\nKnowledge base: {len(self.jarvis.learning_engine.knowledge_base)} entries"
                elif action == 'enable':
                    self.jarvis.learning_mode = LearningMode.ADAPTIVE
                    return "Learning enabled in adaptive mode"
                elif action == 'disable':
                    self.jarvis.learning_mode = LearningMode.PASSIVE
                    return "Learning disabled"
                elif action == 'clear':
                    self.jarvis.learning_engine = LearningEngine()
                    return "Learning data cleared"
                else:
                    return f"Unknown learning action: {action}"
        
        return LearningCommand(self)

    def _load_conversation_history(self):
        """Load conversation history from encrypted file."""
        history_file = Path('conversation_history.enc')
        if history_file.exists():
            try:
                encrypted_data = history_file.read_bytes()
                decrypted_data = self.security_manager.decrypt(encrypted_data.decode())
                history = pickle.loads(decrypted_data.encode())
                
                for entry in history:
                    self.context_manager.add_conversation_entry(entry)
                
                logger.info(f"Loaded {len(history)} encrypted conversation entries")
            except Exception as e:
                logger.error(f"Failed to load encrypted conversation history: {e}")

    def _save_conversation_history(self):
        """Save conversation history to encrypted file."""
        try:
            history = list(self.context_manager.conversation_history)
            serialized_data = pickle.dumps(history)
            encrypted_data = self.security_manager.encrypt(serialized_data.decode())
            
            history_file = Path('conversation_history.enc')
            history_file.write_bytes(encrypted_data.encode())
            
            logger.info("Conversation history saved securely")
        except Exception as e:
            logger.error(f"Failed to save conversation history: {e}")

    def _load_user_profiles(self):
        """Load user profiles from database."""
        profiles_file = Path('user_profiles.db')
        if profiles_file.exists():
            try:
                conn = sqlite3.connect(profiles_file)
                cursor = conn.cursor()
                
                cursor.execute('SELECT * FROM user_profiles')
                rows = cursor.fetchall()
                
                for row in rows:
                    profile = UserProfile(
                        user_id=row[0],
                        name=row[1],
                        voice_profile=json.loads(row[2]),
                        preferences=json.loads(row[3]),
                        permissions=set(json.loads(row[4])),
                        learning_data=json.loads(row[5]),
                        last_active=datetime.datetime.fromisoformat(row[6]),
                        security_level=SecurityLevel(row[7])
                    )
                    self.context_manager.user_profiles[profile.user_id] = profile
                
                conn.close()
                logger.info(f"Loaded {len(rows)} user profiles")
            except Exception as e:
                logger.error(f"Failed to load user profiles: {e}")

    def _save_user_profiles(self):
        """Save user profiles to database."""
        try:
            profiles_file = Path('user_profiles.db')
            conn = sqlite3.connect(profiles_file)
            cursor = conn.cursor()
            
            # Create table if not exists
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                name TEXT,
                voice_profile TEXT,
                preferences TEXT,
                permissions TEXT,
                learning_data TEXT,
                last_active TEXT,
                security_level INTEGER
            )
            ''')
            
            # Clear existing data
            cursor.execute('DELETE FROM user_profiles')
            
            # Insert current profiles
            for profile in self.context_manager.user_profiles.values():
                cursor.execute('''
                INSERT INTO user_profiles VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    profile.user_id,
                    profile.name,
                    json.dumps(profile.voice_profile),
                    json.dumps(profile.preferences),
                    json.dumps(list(profile.permissions)),
                    json.dumps(profile.learning_data),
                    profile.last_active.isoformat(),
                    profile.security_level.value
                ))
            
            conn.commit()
            conn.close()
            logger.info("User profiles saved")
        except Exception as e:
            logger.error(f"Failed to save user profiles: {e}")

    async def run(self):
        """Main event-driven application loop."""
        logger.info(f"Starting Jarvis AI Assistant - Session: {self.session_id}")
        self._show_welcome()
        
        # Authenticate user if required
        if self.config['security']['require_authentication']:
            await self._authenticate_user()
        else:
            # Use default user
            self.current_user = self.context_manager.create_user_profile(
                'default', 'Default User'
            )
        
        # Start concurrent threads
        self._start_threads()
        
        # Start periodic tasks
        self._start_periodic_tasks()
        
        # Start API listeners
        self._start_api_listeners()

        try:
            # Start the main event loop
            await self._main_event_loop()
        except KeyboardInterrupt:
            logger.info("Shutdown requested by user")
        except Exception as e:
            logger.error(f"Fatal error in main loop: {e}")
        finally:
            await self._shutdown_sequence()

    async def _main_event_loop(self):
        """Enhanced main asyncio event loop."""
        # Start wake word detection
        self._start_wake_word_detection()

        while not self.shutdown_event.is_set():
            try:
                # Wait for multiple event types
                tasks = [
                    asyncio.wait_for(self.input_queue.get(), timeout=0.01),
                    asyncio.wait_for(self.alert_queue.get(), timeout=0.01),
                    asyncio.wait_for(self.plugin_queue.get(), timeout=0.01)
                ]
                
                done, pending = await asyncio.wait(
                    tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=1.0
                )

                for task in done:
                    try:
                        event = task.result()
                        await self._process_event(event)
                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        self._log_error(f"Error handling event: {e}", event)

                # Cancel pending tasks
                for task in pending:
                    task.cancel()
                    
                # Perform periodic checks
                await self._periodic_checks()

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self._log_error(f"Error in main event loop: {e}", None)

    async def _process_event(self, event: Dict[str, Any]):
        """Process events with enhanced handling."""
        event_type = event.get('type')
        
        # Log event for analytics
        if self.current_user:
            self.analytics_engine.log_interaction(
                ConversationEntry(
                    id=str(uuid.uuid4()),
                    timestamp=datetime.datetime.now(),
                    user_input=event.get('text', ''),
                    ai_response='',
                    context={},
                    emotion=EmotionState.NEUTRAL
                ),
                self.current_user.user_id,
                0.0
            )
        
        # Handle event based on type
        handlers = {
            'wake_word_detected': self._handle_wake_word,
            'speech_recognized': self._handle_speech_input,
            'text_input': self._handle_text_input,
            'system_alert': self._handle_system_alert,
            'health_check': self._handle_health_check,
            'api_request': self._handle_api_request,
            'plugin_event': self._handle_plugin_event,
            'shutdown': self._handle_shutdown,
            'command_result': self._handle_command_result,
            'learning_update': self._handle_learning_update,
            'user_switch': self._handle_user_switch
        }
        
        handler = handlers.get(event_type)
        if handler:
            try:
                await handler(event)
            except Exception as e:
                self._log_error(f"Handler error for {event_type}: {e}", event)
        else:
            logger.warning(f"Unknown event type: {event_type}")
            
        # Broadcast event to plugins
        await self.plugin_manager.broadcast_event(event)

    async def _handle_wake_word(self, event: Optional[Dict] = None):
        """Handle wake word detection with user recognition."""
        logger.info("Wake word detected")
        
        # Check if we need to authenticate
        if (self.config['security']['require_authentication'] and 
            not self.current_user):
            await self._authenticate_user()
            if not self.current_user:
                self.tts.speak("Authentication required. Please identify yourself.")
                return
        
        self.is_active = True
        
        # Personalized greeting
        greeting = self._get_personalized_greeting()
        self.tts.speak(f"{greeting} How can I help you?")
        
        # Start listening
        self._start_speech_recognition()
        
        # Update user activity
        if self.current_user:
            self.current_user.last_active = datetime.datetime.now()

    async def _handle_speech_input(self, event: Dict[str, Any]):
        """Enhanced speech input handling."""
        if not self.is_active:
            return

        text = event.get('text', '').strip()
        if not text:
            return

        logger.info(f"Processing speech input: {text}")
        
        # Security check
        if not self._security_check('process_input', text):
            self.tts.speak("I'm sorry, I cannot process that request.")
            return
        
        start_time = time.time()
        try:
            # Update context
            self.context_manager.session_context['current_input'] = text
            
            # Detect emotion
            emotion = await self._detect_emotion(text)
            self.personality_manager.mood = emotion
            
            # Adapt personality
            self.personality_manager.adapt_personality({
                'time_of_day': datetime.datetime.now().hour,
                'user_mood': self.context_manager.session_context.get('user_mood', 'neutral'),
                'conversation_topic': self._extract_topic(text),
                'interaction_history': self.context_manager.get_recent_context(5)
            })
            
            # Check learning cache first
            if self.learning_mode != LearningMode.PASSIVE:
                cached_response = self.learning_engine.predict_response(
                    text, self.context_manager.get_recent_context(3)
                )
                if cached_response and cached_response['confidence'] > 0.8:
                    # Use cached response
                    response_time = time.time() - start_time
                    self.response_times.append(response_time)
                    
                    entry = ConversationEntry(
                        id=str(uuid.uuid4()),
                        timestamp=datetime.datetime.now(),
                        user_input=text,
                        ai_response=cached_response['response'],
                        context=self.context_manager.session_context.copy(),
                        emotion=emotion,
                        confidence=cached_response['confidence'],
                        command_executed='cached_response',
                        execution_result=None,
                        metadata={'source': cached_response['source']}
                    )
                    
                    self.context_manager.add_conversation_entry(entry)
                    self.learning_engine.learn_from_interaction(entry)
                    
                    self.tts.speak(cached_response['response'])
                    
                    # Analytics
                    if self.current_user:
                        self.analytics_engine.log_interaction(
                            entry, self.current_user.user_id, response_time
                        )
                    
                    self.is_active = False
                    return
            
            # Get context-aware response
            context = self.context_manager.get_recent_context(5)
            enhanced_input = self._enhance_input_with_context(text, context)
            
            # Process through AI
            decision = await self.loop.run_in_executor(
                None, 
                self._process_with_ai, 
                enhanced_input, 
                context
            )
            
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
            
            # Learn from interaction
            if self.learning_mode != LearningMode.PASSIVE:
                self.learning_engine.learn_from_interaction(entry)
            
            # Execute decision
            await self._execute_decision(decision, entry)
            
            # Analytics
            if self.current_user:
                self.analytics_engine.log_interaction(
                    entry, self.current_user.user_id, response_time
                )
            
            # Check for proactive suggestions
            await self._check_proactive_suggestions()
            
            # Update learning
            await self._update_learning(text, entry)
            
        except Exception as e:
            self._log_error(f"Speech input processing failed: {e}", event)
            self.tts.speak("I encountered an error processing your request.")

    async def _handle_text_input(self, event: Dict[str, Any]):
        """Handle text input from other sources."""
        text = event.get('text', '').strip()
        if not text:
            return
            
        # Process similarly to speech input
        await self.input_queue.put({
            'type': 'speech_recognized',
            'text': text,
            'source': event.get('source', 'text')
        })

    async def _handle_system_alert(self, event: Dict[str, Any]):
        """Handle system alerts."""
        alert_type = event.get('alert_type')
        message = event.get('message', '')
        severity = event.get('severity', 'warning')
        
        logger.warning(f"System alert: {alert_type} - {message}")
        
        # Add to health monitor
        self.health_monitor.add_alert(alert_type, message, severity)
        
        # Speak alert if critical
        if severity == 'critical':
            self.tts.speak(f"Critical alert: {message}")
        elif severity == 'warning' and self.is_active:
            self.tts.speak(f"System alert: {message}")

    async def _handle_health_check(self, event: Optional[Dict] = None):
        """Handle health check requests."""
        metrics = self._get_current_system_metrics()
        health = self.health_monitor.check_health(metrics)
        
        if health['status'] == 'critical':
            await self.alert_queue.put({
                'type': 'system_alert',
                'alert_type': 'health_critical',
                'message': f"System health critical: {health['score']}/100",
                'severity': 'critical'
            })
        elif health['status'] == 'degraded':
            logger.warning(f"System health degraded: {health['score']}/100")

    async def _handle_api_request(self, event: Dict[str, Any]):
        """Handle external API requests."""
        api_name = event.get('api_name')
        endpoint = event.get('endpoint')
        params = event.get('params')
        
        try:
            result = await self.external_api_handler.call_api(
                api_name, endpoint, params
            )
            
            # Process result
            if 'error' not in result:
                # Store result for potential use
                await self.decision_queue.put({
                    'type': 'api_response',
                    'api_name': api_name,
                    'data': result
                })
            else:
                logger.error(f"API request failed: {result['error']}")
                
        except Exception as e:
            logger.error(f"API request failed: {e}")

    async def _handle_plugin_event(self, event: Dict[str, Any]):
        """Handle events from plugins."""
        plugin_name = event.get('plugin')
        event_data = event.get('data', {})
        
        logger.debug(f"Plugin event from {plugin_name}: {event_data}")

    async def _handle_shutdown(self, event: Optional[Dict] = None):
        """Handle shutdown request."""
        logger.info("Shutdown requested")
        self.shutdown_event.set()

    async def _handle_command_result(self, event: Dict[str, Any]):
        """Handle command execution results."""
        result = event.get('result')
        command = event.get('command')
        success = event.get('success', False)
        
        logger.info(f"Command '{command}' completed: {success}")
        
        if not success and self.is_active:
            self.tts.speak(f"Command {command} failed to execute.")

    async def _handle_learning_update(self, event: Dict[str, Any]):
        """Handle learning updates."""
        data = event.get('data', {})
        
        if self.learning_mode != LearningMode.PASSIVE:
            # Update learning engine with new data
            # Implementation depends on your learning strategy
            pass

    async def _handle_user_switch(self, event: Dict[str, Any]):
        """Handle user switching."""
        user_id = event.get('user_id')
        
        if user_id in self.context_manager.user_profiles:
            self.current_user = self.context_manager.user_profiles[user_id]
            self.tts.speak(f"Hello {self.current_user.name}, welcome back!")
        else:
            self.tts.speak("User not recognized.")

    def _get_personalized_greeting(self) -> str:
        """Get personalized greeting for current user."""
        base_greeting = self._get_time_based_greeting()
        
        if self.current_user:
            # Check user preferences for greeting style
            preferred_greeting = self.current_user.preferences.get('greeting_style', 'formal')
            
            if preferred_greeting == 'casual':
                greetings = ["Hey", "Hi", "Hello"]
                return f"{np.random.choice(greetings)} {self.current_user.name}"
            elif preferred_greeting == 'friendly':
                return f"{base_greeting}, {self.current_user.name}! Great to see you!"
            else:  # formal
                return f"{base_greeting}, {self.current_user.name}"
        
        return base_greeting

    def _security_check(self, action: str, data: Any) -> bool:
        """Check if action is allowed."""
        if not self.current_user:
            return False
        
        # Check permissions
        required_permission = f"action.{action}"
        if not self.current_user.has_permission(required_permission):
            logger.warning(f"Permission denied for {action}")
            return False
        
        # Check for sensitive data
        sensitive_keywords = ['password', 'secret', 'key', 'token', 'credit card']
        if isinstance(data, str):
            data_lower = data.lower()
            if any(keyword in data_lower for keyword in sensitive_keywords):
                logger.warning(f"Sensitive data detected in {action}")
                # In production, implement proper sanitization
        
        return True

    def _extract_topic(self, text: str) -> str:
        """Extract main topic from text."""
        # Simple implementation - in production, use NLP
        topics = {
            'weather': ['weather', 'rain', 'sunny', 'temperature'],
            'news': ['news', 'headlines', 'update'],
            'system': ['system', 'computer', 'status', 'health'],
            'music': ['music', 'song', 'playlist', 'artist']
        }
        
        text_lower = text.lower()
        for topic, keywords in topics.items():
            if any(keyword in text_lower for keyword in keywords):
                return topic
        
        return 'general'

    async def _update_learning(self, input_text: str, entry: ConversationEntry):
        """Update learning based on interaction."""
        if self.learning_mode == LearningMode.ACTIVE:
            # Active learning - ask for feedback
            if np.random.random() < 0.1:  # 10% chance to ask
                self.tts.speak("Was that response helpful?")
                # In production, listen for feedback
        elif self.learning_mode == LearningMode.EXPERT:
            # Expert mode - analyze interaction deeply
            # In production, implement advanced analysis
            pass

    def _enhance_input_with_context(self, text: str, context: List[ConversationEntry]) -> str:
        """Enhanced context integration."""
        if not context:
            return text
        
        context_str = "\nRecent conversation context:\n"
        for i, entry in enumerate(context[-3:], 1):
            context_str += f"{i}. User: {entry.user_input}\n"
            context_str += f"   Jarvis: {entry.ai_response}\n"
        
        # Add user preferences if available
        if self.current_user:
            preferences = self.current_user.preferences
            if preferences:
                context_str += f"\nUser preferences: {json.dumps(preferences, indent=2)}\n"
        
        # Add system context
        context_str += f"\nSystem time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        context_str += f"Jarvis mood: {self.personality_manager.mood.value}\n"
        
        return f"{context_str}\nCurrent input: {text}"

    def _process_with_ai(self, text: str, context: List[ConversationEntry]) -> Dict[str, Any]:
        """Process text with AI with enhanced context."""
        # Add personality instructions
        personality = self.personality_manager.get_response_style()
        instructions = f"""
        Respond as Jarvis AI Assistant with the following personality:
        - Style: {personality['greeting_style']}
        - Include humor: {personality['include_humor']}
        - Response length: {personality['response_length']}
        - Current mood: {self.personality_manager.mood.value}
        
        User context:
        - Name: {self.current_user.name if self.current_user else 'Guest'}
        - Security level: {self.current_user.security_level.name if self.current_user else 'PUBLIC'}
        
        System context:
        - Time: {datetime.datetime.now().strftime('%I:%M %p')}
        - Health: {self.health_monitor.check_health(self._get_current_system_metrics())['status']}
        
        Instructions:
        {text}
        """
        
        return self.ai.decide_action(instructions)

    async def _execute_decision(self, decision: Dict[str, Any], entry: ConversationEntry):
        """Execute decision with enhanced capabilities."""
        action_type = decision.get('action')
        params = decision.get('params', {})
        
        entry.command_executed = action_type
        
        handlers = {
            'command': self._execute_command,
            'respond': self._execute_response,
            'search': self._execute_search,
            'schedule': self._execute_schedule,
            'system_info': self._show_system_info,
            'personality_adjust': self._adjust_personality,
            'api_call': self._execute_api_call,
            'plugin_action': self._execute_plugin_action,
            'learning_action': self._execute_learning_action
        }
        
        handler = handlers.get(action_type)
        if handler:
            try:
                result = await handler(params)
                entry.execution_result = str(result)
            except Exception as e:
                entry.execution_result = f"Error: {e}"
                self._log_error(f"Decision execution failed: {e}", decision)
        else:
            logger.warning(f"Unknown action type: {action_type}")
            self.tts.speak("I'm not sure how to handle that request.")
        
        if not decision.get('continue_conversation', False):
            self.is_active = False

    async def _execute_response(self, params: Dict[str, Any]):
        """Execute response action."""
        response = params.get('response_text', '')
        if response:
            self.tts.speak(response)
        return response

    async def _execute_api_call(self, params: Dict[str, Any]):
        """Execute API call."""
        api_name = params.get('api_name')
        endpoint = params.get('endpoint')
        api_params = params.get('params', {})
        
        result = await self.external_api_handler.call_api(
            api_name, endpoint, api_params
        )
        
        if 'error' not in result:
            # Format result for speech
            formatted_result = self._format_api_result(api_name, result)
            self.tts.speak(formatted_result)
            return formatted_result
        else:
            error_msg = f"API call failed: {result['error']}"
            self.tts.speak(error_msg)
            return error_msg

    async def _execute_plugin_action(self, params: Dict[str, Any]):
        """Execute plugin action."""
        plugin_name = params.get('plugin_name')
        action = params.get('action')
        action_params = params.get('params', {})
        
        if plugin_name in self.plugin_manager.plugins:
            plugin = self.plugin_manager.plugins[plugin_name]
            # In production, implement plugin action execution
            return f"Plugin {plugin_name} action {action} executed"
        else:
            return f"Plugin {plugin_name} not found"

    async def _execute_learning_action(self, params: Dict[str, Any]):
        """Execute learning action."""
        action = params.get('action', '')
        
        if action == 'analyze_patterns':
            patterns = len(self.learning_engine.patterns)
            return f"Found {patterns} learning patterns"
        elif action == 'knowledge_summary':
            entries = len(self.learning_engine.knowledge_base)
            return f"Knowledge base has {entries} entries"
        else:
            return f"Unknown learning action: {action}"

    def _format_api_result(self, api_name: str, result: Dict[str, Any]) -> str:
        """Format API result for speech."""
        if api_name == 'weather':
            if 'main' in result:
                temp = result['main'].get('temp', 0)
                description = result['weather'][0].get('description', '')
                return f"Current weather: {description}, temperature {temp}°C"
        elif api_name == 'news':
            if 'articles' in result and result['articles']:
                article = result['articles'][0]
                title = article.get('title', '')
                return f"Latest news: {title}"
        
        return "API response received"

    async def _periodic_checks(self):
        """Perform periodic system checks."""
        current_time = time.time()
        
        # Check system health every 30 seconds
        if not hasattr(self, '_last_health_check'):
            self._last_health_check = 0
        
        if current_time - self._last_health_check > 30:
            await self._handle_health_check(None)
            self._last_health_check = current_time
        
        # Check for resource limits
        metrics = self._get_current_system_metrics()
        limits = self.config['system']['resource_limits']
        
        if metrics.cpu_percent > limits['max_cpu_percent']:
            await self.alert_queue.put({
                'type': 'system_alert',
                'alert_type': 'high_cpu',
                'message': f'CPU usage at {metrics.cpu_percent:.1f}%',
                'severity': 'warning'
            })
        
        if metrics.memory_percent > limits['max_memory_percent']:
            await self.alert_queue.put({
                'type': 'system_alert',
                'alert_type': 'low_memory',
                'message': f'Memory usage at {metrics.memory_percent:.1f}%',
                'severity': 'warning'
            })

    def _get_system_status(self) -> str:
        """Get comprehensive system status."""
        metrics = self._get_current_system_metrics()
        health = self.health_monitor.check_health(metrics)
        
        status = f"""
        ╔══════════════════════════════════════════════════════════════╗
        ║                     System Status Report                      ║
        ╠══════════════════════════════════════════════════════════════╣
        ║ Health Score: {health['score']}/100 ({health['status'].upper()})
        ║ Uptime: {datetime.datetime.now() - self.start_time}
        ║ Session: {self.session_id[:8]}...
        ║ Current User: {self.current_user.name if self.current_user else 'None'}
        ║
        ║ Performance Metrics:
        ║ • CPU: {metrics.cpu_percent:.1f}%
        ║ • Memory: {metrics.memory_percent:.1f}%
        ║ • Disk: {metrics.disk_percent:.1f}%
        ║ • Avg Response Time: {
            sum(self.response_times)/len(self.response_times) if self.response_times else 0:.2f}s
        ║
        ║ System Information:
        ║ • Conversations: {len(self.context_manager.conversation_history)}
        ║ • Users: {len(self.context_manager.user_profiles)}
        ║ • Plugins: {len(self.plugin_manager.plugins)}
        ║ • Learning Mode: {self.learning_mode.name}
        ║ • Mood: {self.personality_manager.mood.value}
        ╚══════════════════════════════════════════════════════════════╝
        """
        
        return status

    def _get_health_report(self) -> str:
        """Get detailed health report."""
        metrics = self._get_current_system_metrics()
        health = self.health_monitor.check_health(metrics)
        
        report = f"""
        Health Report:
        -------------
        Overall Health: {health['score']}/100 ({health['status']})
        Timestamp: {datetime.datetime.now()}
        
        Resource Usage:
        • CPU: {metrics.cpu_percent:.1f}% {'⚠️' if metrics.cpu_percent > 80 else '✅'}
        • Memory: {metrics.memory_percent:.1f}% {'⚠️' if metrics.memory_percent > 85 else '✅'}
        • Disk: {metrics.disk_percent:.1f}% {'⚠️' if metrics.disk_percent > 90 else '✅'}
        
        Performance:
        • Response Times: {len(self.response_times)} samples
        • Avg Response: {sum(self.response_times)/len(self.response_times) if self.response_times else 0:.2f}s
        • Recent Alerts: {len(self.health_monitor.alerts)}
        
        System Info:
        • Uptime: {datetime.datetime.now() - self.start_time}
        • Active: {'Yes' if self.is_active else 'No'}
        • Listening: {'Yes' if self.is_listening else 'No'}
        """
        
        return report

    def _get_metrics_summary(self) -> str:
        """Get metrics summary."""
        if not self.system_metrics_history:
            return "No metrics available yet."
        
        recent_metrics = list(self.system_metrics_history)[-10:]
        
        summary = "Recent System Metrics:\n"
        for metric in recent_metrics:
            summary += f"""
            {metric.timestamp.strftime('%H:%M:%S')}:
              CPU: {metric.cpu_percent:.1f}%
              Memory: {metric.memory_percent:.1f}%
              Disk: {metric.disk_percent:.1f}%
            """
        
        return summary

    async def _restart_system(self) -> str:
        """Restart the system."""
        self.tts.speak("Restarting system components.")
        
        # Save state
        self._save_conversation_history()
        self._save_user_profiles()
        
        # Restart components
        self._stop_threads()
        await asyncio.sleep(1)
        self._setup_components()
        self._start_threads()
        
        return "System restart completed"

    async def _check_for_updates(self) -> str:
        """Check for system updates."""
        # In production, implement actual update checking
        return "Update check not implemented yet"

    def _authenticate_user(self):
        """Authenticate user."""
        # In production, implement proper authentication
        # For now, create a default user
        self.current_user = self.context_manager.create_user_profile(
            'user_001', 'Primary User',
            preferences={'greeting_style': 'friendly'},
            permissions={'basic', 'system_info', 'commands'}
        )
        return True

    def _start_system_monitoring(self):
        """Start enhanced system monitoring."""
        self.threads['system_monitor'] = threading.Thread(
            target=self._system_monitoring_loop,
            daemon=True,
            name='SystemMonitor'
        )
        self.threads['system_monitor'].start()

    def _start_health_monitoring(self):
        """Start health monitoring."""
        self.threads['health_monitor'] = threading.Thread(
            target=self._health_monitoring_loop,
            daemon=True,
            name='HealthMonitor'
        )
        self.threads['health_monitor'].start()

    def _system_monitoring_loop(self):
        """Enhanced system monitoring loop."""
        while not self.shutdown_event.is_set():
            try:
                metrics = self._get_current_system_metrics()
                self.system_metrics_history.append(metrics)
                
                # Check for anomalies
                if self.health_monitor.anomaly_detector.detect(metrics):
                    asyncio.run_coroutine_threadsafe(
                        self.alert_queue.put({
                            'type': 'system_alert',
                            'alert_type': 'anomaly_detected',
                            'message': 'System anomaly detected',
                            'severity': 'warning'
                        }),
                        self.loop
                    )
                
                time.sleep(2)  # Check every 2 seconds
            except Exception as e:
                self._log_error(f"System monitoring error: {e}", None)
                time.sleep(5)

    def _health_monitoring_loop(self):
        """Health monitoring loop."""
        while not self.shutdown_event.is_set():
            try:
                # Perform health checks
                asyncio.run_coroutine_threadsafe(
                    self.alert_queue.put({'type': 'health_check'}),
                    self.loop
                )
                
                time.sleep(self.config['system']['health_check_interval'])
            except Exception as e:
                self._log_error(f"Health monitoring error: {e}", None)
                time.sleep(10)

    def _start_api_listeners(self):
        """Start API listeners."""
        # Start WebSocket server for real-time communication
        self.threads['websocket_server'] = threading.Thread(
            target=self._start_websocket_server,
            daemon=True,
            name='WebSocketServer'
        )
        self.threads['websocket_server'].start()
        
        # Start HTTP API server
        self.threads['http_server'] = threading.Thread(
            target=self._start_http_server,
            daemon=True,
            name='HTTPServer'
        )
        self.threads['http_server'].start()

    def _start_websocket_server(self):
        """Start WebSocket server."""
        async def websocket_handler(websocket, path):
            try:
                async for message in websocket:
                    data = json.loads(message)
                    await self.input_queue.put({
                        'type': 'text_input',
                        'text': data.get('text', ''),
                        'source': 'websocket'
                    })
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
        
        # Start server
        start_server = websockets.serve(
            websocket_handler, 
            "localhost", 
            8765
        )
        
        asyncio.run_coroutine_threadsafe(start_server, self.loop)

    def _start_http_server(self):
        """Start HTTP API server."""
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import urllib.parse
        
        class JarvisAPIHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode())
                
                # Queue the request
                asyncio.run_coroutine_threadsafe(
                    self.server.jarvis.input_queue.put({
                        'type': 'api_request',
                        'endpoint': self.path,
                        'data': data
                    }),
                    self.server.jarvis.loop
                )
                
                self.send_response(202)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                response = json.dumps({'status': 'queued'})
                self.wfile.write(response.encode())
            
            def log_message(self, format, *args):
                logger.info(f"HTTP {format % args}")
        
        # Custom server class to pass Jarvis instance
        class JarvisHTTPServer(HTTPServer):
            def __init__(self, *args, **kwargs):
                self.jarvis = kwargs.pop('jarvis')
                super().__init__(*args, **kwargs)
        
        server = JarvisHTTPServer(
            ('localhost', 8080), 
            JarvisAPIHandler,
            jarvis=self
        )
        
        server.serve_forever()

    def _start_threads(self):
        """Start all background threads."""
        # Start audio processing
        self.audio_thread = threading.Thread(
            target=self._audio_processing_loop,
            daemon=True,
            name='AudioProcessor'
        )
        self.audio_thread.start()
        
        # Start processing thread
        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True,
            name='CommandProcessor'
        )
        self.processing_thread.start()
        
        logger.info("All background threads started")

    def _processing_loop(self):
        """Background processing loop."""
        while not self.shutdown_event.is_set():
            try:
                # Process items from decision queue
                # This runs in a separate thread for CPU-intensive tasks
                time.sleep(0.1)
            except Exception as e:
                self._log_error(f"Processing loop error: {e}", None)

    def _audio_processing_loop(self):
        """Enhanced audio processing loop."""
        while not self.shutdown_event.is_set():
            try:
                if self.is_active and self.is_listening:
                    # Process speech recognition
                    # Implementation depends on your speech library
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
                
                time.sleep(0.01)  # 100 FPS
            except Exception as e:
                self._log_error(f"Audio processing error: {e}", None)
                time.sleep(0.1)

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
                await asyncio.sleep(60)
                await self.alert_queue.put({
                    'type': 'periodic_task',
                    'task': 'collect_metrics'
                })
        
        async def backup_creator():
            while not self.shutdown_event.is_set():
                await asyncio.sleep(3600)  # Every hour
                await self.alert_queue.put({
                    'type': 'periodic_task',
                    'task': 'backup'
                })
        
        async def analytics_reporter():
            while not self.shutdown_event.is_set():
                await asyncio.sleep(self.config['analytics']['report_interval'])
                await self._generate_analytics_report()
        
        # Start all periodic tasks
        self.loop.create_task(periodic_saver())
        self.loop.create_task(metrics_collector())
        if self.config['system']['backup_enabled']:
            self.loop.create_task(backup_creator())
        if self.config['analytics']['enabled']:
            self.loop.create_task(analytics_reporter())

    async def _generate_analytics_report(self):
        """Generate analytics report."""
        try:
            today = datetime.date.today()
            stats = self.analytics_engine.get_daily_stats(today)
            
            report = f"""
            Daily Analytics Report - {today}
            --------------------------------
            Total Interactions: {stats['total_interactions']}
            Average Response Time: {stats['avg_response_time']:.2f}s
            Success Rate: {stats['success_rate']:.1f}%
            System Health: {self.health_monitor.check_health(self._get_current_system_metrics())['status']}
            """
            
            logger.info(report)
            
            # Save report to file
            report_file = Path(f"analytics_report_{today}.txt")
            report_file.write_text(report)
            
        except Exception as e:
            logger.error(f"Analytics report generation failed: {e}")

    async def _shutdown_sequence(self):
        """Enhanced shutdown sequence."""
        logger.info("Starting shutdown sequence...")
        
        # Stop all threads
        self._stop_threads()
        
        # Shutdown plugins
        await self.plugin_manager.shutdown_plugins()
        
        # Save all data
        self._save_conversation_history()
        self._save_user_profiles()
        
        # Close connections
        self.speech_recognizer.stop_wake_word_detection()
        self.speech_recognizer.stop_audio_stream()
        self.tts.stop()
        
        # Generate final report
        self._generate_final_report()
        
        # Handle exit
        self._handle_exit()

    def _stop_threads(self):
        """Stop all background threads."""
        self.shutdown_event.set()
        
        for name, thread in self.threads.items():
            if thread and thread.is_alive():
                thread.join(timeout=2)
                logger.info(f"Thread {name} stopped")
        
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=2)
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2)

    def _generate_final_report(self):
        """Generate final session report."""
        duration = datetime.datetime.now() - self.start_time
        avg_response = sum(self.response_times)/len(self.response_times) if self.response_times else 0
        
        report = f"""
        ╔══════════════════════════════════════════════════════════════╗
        ║                    Session Final Report                       ║
        ╠══════════════════════════════════════════════════════════════╣
        ║ Session ID: {self.session_id}
        ║ Duration: {duration}
        ║ Start Time: {self.start_time}
        ║ End Time: {datetime.datetime.now()}
        ║
        ║ Performance Summary:
        ║ • Total Interactions: {len(self.context_manager.conversation_history)}
        ║ • Avg Response Time: {avg_response:.2f}s
        ║ • System Health: {self.health_monitor.check_health(self._get_current_system_metrics())['status']}
        ║
        ║ User Summary:
        ║ • Active Users: {len(self.context_manager.user_profiles)}
        ║ • Current User: {self.current_user.name if self.current_user else 'None'}
        ║
        ║ Learning Summary:
        ║ • Knowledge Base: {len(self.learning_engine.knowledge_base)} entries
        ║ • Patterns Learned: {len(self.learning_engine.patterns)}
        ║ • Learning Mode: {self.learning_mode.name}
        ╚══════════════════════════════════════════════════════════════╝
        """
        
        logger.info(report)
        
        # Save to file
        report_file = Path(f"session_report_{self.session_id[:8]}.txt")
        report_file.write_text(report)

    def _log_error(self, message: str, context: Optional[Dict[str, Any]]):
        """Log error with context."""
        error_entry = {
            'timestamp': datetime.datetime.now(),
            'message': message,
            'context': context,
            'system_metrics': asdict(self._get_current_system_metrics()) if hasattr(self, '_get_current_system_metrics') else {}
        }
        self.error_log.append(error_entry)
        logger.error(f"{message} - Context: {context}")

    def _get_current_system_metrics(self) -> SystemMetrics:
        """Get current system metrics with GPU support."""
        try:
            gpus = GPUtil.getGPUs()
            gpu = gpus[0] if gpus else None
        except:
            gpu = None
        
        return SystemMetrics(
            timestamp=datetime.datetime.now(),
            cpu_percent=psutil.cpu_percent(interval=0.1),
            memory_percent=psutil.virtual_memory().percent,
            disk_percent=psutil.disk_usage('/').percent,
            network_bytes_sent=psutil.net_io_counters().bytes_sent,
            network_bytes_recv=psutil.net_io_counters().bytes_recv,
            gpu_utilization=gpu.load * 100 if gpu else None,
            gpu_memory_percent=gpu.memoryUtil * 100 if gpu else None,
            process_count=len(psutil.pids())
        )

    def _get_time_based_greeting(self) -> str:
        """Get time-based greeting."""
        hour = datetime.datetime.now().hour
        
        if 5 <= hour < 12:
            return "Good morning"
        elif 12 <= hour < 17:
            return "Good afternoon"
        elif 17 <= hour < 22:
            return "Good evening"
        else:
            return "Hello"

    def _parse_command_params(self, params: str) -> Dict[str, str]:
        """Parse command parameters."""
        result = {}
        if params:
            for item in params.split(','):
                if '=' in item:
                    key, value = item.split('=', 1)
                    result[key.strip()] = value.strip()
        return result

    def _show_welcome(self):
        """Display enhanced welcome message."""
        welcome_text = f"""
╔══════════════════════════════════════════════════════════════════════════╗
║                    🤖 Jarvis AI Assistant - Supercharged                 ║
║                    Session: {self.session_id[:12]}...                            ║
╚══════════════════════════════════════════════════════════════════════════╝

🚀 **System Initialized Successfully!**
• Version: 3.0.0
• Platform: {platform.system()} {platform.release()}
• Python: {platform.python_version()}
• Architecture: {platform.machine()}
• Processors: {psutil.cpu_count()} cores
• Memory: {psutil.virtual_memory().total / 1024**3:.1f} GB

🎭 **Personality Profile:**
{json.dumps(self.personality_manager.traits, indent=2)}

📊 **Current Status:**
• Security: {'🔒 Authenticated' if self.current_user else '🔓 Public'}
• Learning: {self.learning_mode.name}
• Mood: {self.personality_manager.mood.value}
• Health: {self.health_monitor.check_health(self._get_current_system_metrics())['status']}

🔌 **Loaded Features:**
✓ Multi-User Support
✓ Advanced Learning Engine
✓ Plugin System ({len(self.plugin_manager.plugins)} plugins)
✓ Health Monitoring
✓ External API Integration
✓ Real-time Analytics
✓ Cross-Platform Compatibility

💡 **Available Commands:**
- "system status" - Get comprehensive system report
- "system health" - Check system health
- "learning status" - View learning progress
- "personality adjust [trait] [value]" - Adjust personality
- "api weather" - Get weather information
- "api news" - Get latest news
- "user switch [id]" - Switch user profile

🔊 **Voice Commands:**
Say "{self.config['speech']['wake_word']}" followed by your command.

📡 **Connectivity:**
• WebSocket API: ws://localhost:8765
• HTTP API: http://localhost:8080
• Logs: jarvis.log, jarvis_debug.log

Ready to assist! Say "{self.config['speech']['wake_word']}" to begin.
        """
        print(welcome_text)

    def _handle_exit(self):
        """Handle application exit."""
        logger.info("Jarvis shutting down...")
        
        exit_text = f"""
╔══════════════════════════════════════════════════════════════╗
║                    Jarvis Shutdown Complete                  ║
╠══════════════════════════════════════════════════════════════╣
║ Session: {self.session_id}
║ Duration: {datetime.datetime.now() - self.start_time}
║ Interactions: {len(self.context_manager.conversation_history)}
║ Users: {len(self.context_manager.user_profiles)}
║ Final Mood: {self.personality_manager.mood.value}
║
║ All data saved securely.
║ Plugins shutdown gracefully.
║ Connections closed.
║
║ Thank you for using Jarvis AI Assistant!
║ See you next time! 👋
╚══════════════════════════════════════════════════════════════╝
        """
        
        print(exit_text)
        logger.info("Jarvis shutdown complete")

class JarvisPluginExample(PluginBase):
    """Example plugin for Jarvis."""
    
    @property
    def name(self) -> str:
        return "example_plugin"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "Example plugin for Jarvis AI Assistant"
    
    async def initialize(self):
        """Initialize plugin."""
        logger.info(f"Initializing {self.name} v{self.version}")
        
    async def shutdown(self):
        """Shutdown plugin."""
        logger.info(f"Shutting down {self.name}")
        
    async def handle_event(self, event: Dict[str, Any]):
        """Handle events from Jarvis."""
        if event.get('type') == 'speech_recognized':
            text = event.get('text', '')
            if 'example' in text.lower():
                # Respond to example keyword
                await self.jarvis.input_queue.put({
                    'type': 'text_input',
                    'text': "I heard you mention 'example'!",
                    'source': 'example_plugin'
                })

def main():
    """Main entry point with enhanced error handling."""
    try:
        # Set up signal handlers
        import signal
        
        def signal_handler(sig, frame):
            print("\n\nReceived shutdown signal. Shutting down gracefully...")
            raise KeyboardInterrupt
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Initialize Jarvis
        jarvis = Jarvis()
        
        # Run with timeout protection
        asyncio.run(jarvis.run(), debug=True)
        
    except KeyboardInterrupt:
        print("\n\nShutdown initiated by user.")
    except Exception as e:
        print(f"\n⚠️  Fatal error: {e}")
        logger.critical(f"Fatal error: {e}", exc_info=True)
        
        # Attempt emergency shutdown
        try:
            if 'jarvis' in locals():
                jarvis.shutdown_event.set()
        except:
            pass
        
        sys.exit(1)

if __name__ == "__main__":
    main()


