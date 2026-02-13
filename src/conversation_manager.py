"""
Conversation Manager
Handles conversation history and context for interactive debugging sessions with enhanced features.
"""

import json
import logging
from typing import List, Dict, Optional, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import sqlite3
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from collections import defaultdict
import re
import pickle
import zlib
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages in conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    DEBUG = "debug"
    CODE = "code"
    SOLUTION = "solution"
    PROBLEM = "problem"
    FEEDBACK = "feedback"
    COMMAND = "command"
    RESULT = "result"


class ConversationState(Enum):
    """States of a conversation."""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"
    ERROR = "error"


class SentimentType(Enum):
    """Sentiment analysis results."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    FRUSTRATED = "frustrated"
    SATISFIED = "satisfied"
    CONFUSED = "confused"


@dataclass
class Message:
    """Enhanced message structure."""
    id: str
    timestamp: datetime
    role: MessageType
    content: str
    metadata: Dict[str, Any] = None
    tokens: Optional[int] = None
    sentiment: Optional[str] = None
    topics: List[str] = None
    references: List[str] = None
    attachments: List[str] = None
    edited: bool = False
    edit_history: List[Dict] = None


@dataclass
class ConversationSummary:
    """Summary of a conversation."""
    id: str
    start_time: datetime
    end_time: Optional[datetime]
    state: ConversationState
    total_messages: int
    total_problems: int
    total_solutions: int
    total_tokens: int
    duration: Optional[timedelta]
    participants: List[str]
    topics: List[str]
    sentiment_trend: List[Dict]
    tags: List[str]
    rating: Optional[int]
    feedback: Optional[str]


class ConversationManager:
    """Enhanced conversation manager with advanced features."""

    def __init__(
        self,
        max_history: int = 100,
        storage_path: Optional[Path] = None,
        db_path: Optional[Path] = None,
        enable_compression: bool = True,
        enable_search: bool = True,
        enable_sentiment: bool = True,
        enable_topics: bool = True,
        auto_save: bool = True,
        auto_summarize: bool = True
    ):
        self.max_history = max_history
        self.storage_path = storage_path or Path("conversations")
        self.storage_path.mkdir(exist_ok=True, parents=True)
        
        # Database for advanced features
        self.db_path = db_path or self.storage_path / "conversations.db"
        self.enable_compression = enable_compression
        self.enable_search = enable_search
        self.enable_sentiment = enable_sentiment
        self.enable_topics = enable_topics
        self.auto_save = auto_save
        self.auto_summarize = auto_summarize
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Conversations cache
        self.conversations: Dict[str, Dict] = {}
        self.current_conversation_id: Optional[str] = None
        
        # Search index
        self.search_index = defaultdict(set)
        
        # Sentiment analysis patterns (simple implementation)
        self.sentiment_patterns = {
            SentimentType.POSITIVE: [
                r'\b(great|awesome|excellent|perfect|thanks|good|love|amazing)\b',
                r'✓|✅|👍|🎉'
            ],
            SentimentType.NEGATIVE: [
                r'\b(bad|wrong|error|bug|issue|problem|fail|broken|terrible|awful)\b',
                r'✗|❌|👎|😞'
            ],
            SentimentType.FRUSTRATED: [
                r'\b(frustrat|annoying|hate|stupid|useless|waste|confusing)\b',
                r'😠|😡|🤬'
            ],
            SentimentType.SATISFIED: [
                r'\b(works|solved|fixed|perfect|excellent|great job|thank you)\b',
                r'🎯|⭐|🌟'
            ],
            SentimentType.CONFUSED: [
                r'\b(confus|unclear|don\'t understand|what|how|why|help|explain)\b',
                r'🤔|😕|❓'
            ]
        }
        
        # Initialize database
        self._init_database()
        
        # Start new conversation
        self.new_conversation()
        
        logger.info(f"Conversation manager initialized with max_history={max_history}")

    def _init_database(self):
        """Initialize SQLite database for conversation storage."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Conversations table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS conversations (
                        id TEXT PRIMARY KEY,
                        start_time TIMESTAMP,
                        end_time TIMESTAMP,
                        state TEXT,
                        summary TEXT,
                        metadata TEXT,
                        tags TEXT,
                        rating INTEGER,
                        feedback TEXT
                    )
                ''')
                
                # Messages table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS messages (
                        id TEXT PRIMARY KEY,
                        conversation_id TEXT,
                        timestamp TIMESTAMP,
                        role TEXT,
                        content TEXT,
                        metadata TEXT,
                        tokens INTEGER,
                        sentiment TEXT,
                        topics TEXT,
                        references TEXT,
                        attachments TEXT,
                        FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                    )
                ''')
                
                # Search index table (for full-text search)
                if self.enable_search:
                    cursor.execute('''
                        CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts 
                        USING fts5(content, conversation_id, message_id)
                    ''')
                
                # Topics table
                if self.enable_topics:
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS topics (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            name TEXT UNIQUE,
                            count INTEGER DEFAULT 1,
                            last_seen TIMESTAMP
                        )
                    ''')
                    
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS message_topics (
                            message_id TEXT,
                            topic_id INTEGER,
                            confidence REAL,
                            FOREIGN KEY (message_id) REFERENCES messages(id),
                            FOREIGN KEY (topic_id) REFERENCES topics(id)
                        )
                    ''')
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")

    @contextmanager
    def _get_db_connection(self):
        """Get database connection with context manager."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _generate_id(self, prefix: str = "msg") -> str:
        """Generate a unique ID."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        random_part = hashlib.md5(str(timestamp).encode()).hexdigest()[:8]
        return f"{prefix}_{timestamp}_{random_part}"

    def _analyze_sentiment(self, text: str) -> Optional[str]:
        """Analyze sentiment of text."""
        if not self.enable_sentiment:
            return None
            
        text_lower = text.lower()
        
        for sentiment, patterns in self.sentiment_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    return sentiment.value
        
        return SentimentType.NEUTRAL.value

    def _extract_topics(self, text: str) -> List[str]:
        """Extract topics from text."""
        if not self.enable_topics:
            return []
            
        # Simple topic extraction (can be enhanced with NLP)
        topics = []
        
        # Programming language detection
        languages = ['python', 'javascript', 'java', 'cpp', 'ruby', 'go', 'rust']
        for lang in languages:
            if lang in text.lower():
                topics.append(f"language:{lang}")
        
        # Problem type detection
        problem_types = ['bug', 'error', 'optimization', 'refactor', 'test', 'documentation']
        for ptype in problem_types:
            if ptype in text.lower():
                topics.append(f"type:{ptype}")
        
        # Framework detection
        frameworks = ['django', 'flask', 'react', 'vue', 'angular', 'spring', 'pytorch']
        for framework in frameworks:
            if framework in text.lower():
                topics.append(f"framework:{framework}")
        
        return list(set(topics))  # Remove duplicates

    def _compress_content(self, content: str) -> bytes:
        """Compress content for storage."""
        if not self.enable_compression:
            return content.encode()
        
        compressed = zlib.compress(content.encode(), level=6)
        return compressed

    def _decompress_content(self, compressed: bytes) -> str:
        """Decompress stored content."""
        if not self.enable_compression:
            return compressed.decode() if isinstance(compressed, bytes) else compressed
        
        try:
            decompressed = zlib.decompress(compressed)
            return decompressed.decode()
        except:
            return str(compressed)

    def new_conversation(
        self,
        metadata: Optional[Dict] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """Start a new conversation with enhanced metadata."""
        with self._lock:
            # Save current conversation if it has content
            if self.current_conversation_id and self.auto_save:
                self.save_conversation(self.current_conversation_id)
            
            # Create new conversation
            conversation_id = self._generate_id("conv")
            
            self.conversations[conversation_id] = {
                'id': conversation_id,
                'start_time': datetime.now(),
                'end_time': None,
                'state': ConversationState.ACTIVE,
                'messages': [],
                'problems': [],
                'solutions': [],
                'metadata': metadata or {},
                'tags': tags or [],
                'participants': ['user', 'assistant'],
                'summary': None,
                'total_tokens': 0
            }
            
            self.current_conversation_id = conversation_id
            
            # Add system message
            self.add_message(
                "Conversation started",
                MessageType.SYSTEM,
                metadata={'event': 'conversation_start'}
            )
            
            logger.info(f"Started new conversation {conversation_id}")
            return conversation_id

    def add_message(
        self,
        content: str,
        role: Union[MessageType, str],
        metadata: Optional[Dict] = None,
        tokens: Optional[int] = None,
        references: Optional[List[str]] = None,
        attachments: Optional[List[str]] = None,
        conversation_id: Optional[str] = None
    ) -> str:
        """Add a message to the conversation with enhanced features."""
        with self._lock:
            conv_id = conversation_id or self.current_conversation_id
            if not conv_id or conv_id not in self.conversations:
                conv_id = self.new_conversation()
            
            conversation = self.conversations[conv_id]
            
            # Convert role to enum if string
            if isinstance(role, str):
                try:
                    role = MessageType[role.upper()]
                except KeyError:
                    role = MessageType.USER
            
