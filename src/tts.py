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
    
