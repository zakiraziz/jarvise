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
    
