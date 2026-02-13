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
                        # Create message
            message_id = self._generate_id()
            timestamp = datetime.now()
            
            # Analyze sentiment
            sentiment = self._analyze_sentiment(content) if self.enable_sentiment else None
            
            # Extract topics
            topics = self._extract_topics(content) if self.enable_topics else []
            
            message = Message(
                id=message_id,
                timestamp=timestamp,
                role=role,
                content=content,
                metadata=metadata or {},
                tokens=tokens,
                sentiment=sentiment,
                topics=topics,
                references=references or [],
                attachments=attachments or [],
                edited=False,
                edit_history=[]
            )
            
            # Convert to dict for storage
            message_dict = {
                'id': message.id,
                'timestamp': message.timestamp.isoformat(),
                'role': message.role.value,
                'content': self._compress_content(message.content),
                'metadata': json.dumps(message.metadata),
                'tokens': message.tokens,
                'sentiment': message.sentiment,
                'topics': json.dumps(message.topics),
                'references': json.dumps(message.references),
                'attachments': json.dumps(message.attachments),
                'edited': message.edited,
                'edit_history': json.dumps(message.edit_history)
            }
            
            conversation['messages'].append(message_dict)
            conversation['total_tokens'] += (tokens or 0)
            
            # Update topics in database
            if self.enable_topics and topics:
                self._update_topics(conv_id, message_id, topics)
            
            # Update search index
            if self.enable_search:
                self._update_search_index(conv_id, message_id, content)
            
            # Maintain max history
            if len(conversation['messages']) > self.max_history:
                conversation['messages'] = conversation['messages'][-self.max_history:]
            
            # Auto-save if enabled
            if self.auto_save and len(conversation['messages']) % 10 == 0:
                self.save_conversation(conv_id)
            
            # Auto-summarize if enabled
            if self.auto_summarize and len(conversation['messages']) % 20 == 0:
                self.generate_summary(conv_id)
            
            logger.debug(f"Added message {message_id} to conversation {conv_id}")
            return message_id

    def add_problem(
        self,
        problem_text: str,
        analysis: Dict,
        metadata: Optional[Dict] = None
    ) -> str:
        """Add a problem to the current conversation with enhanced metadata."""
        problem_id = self._generate_id("prob")
        
        problem_entry = {
            'id': problem_id,
            'timestamp': datetime.now().isoformat(),
            'text': problem_text,
            'analysis': analysis,
            'status': 'open',
            'attempts': 0,
            'solutions_tried': [],
            'metadata': metadata or {}
        }

        with self._lock:
            conv_id = self.current_conversation_id
            if conv_id:
                self.conversations[conv_id]['problems'].append(problem_entry)

        # Add as message
        self.add_message(
            f"Problem: {problem_text[:200]}...",
            MessageType.PROBLEM,
            metadata={'problem_id': problem_id, 'analysis': analysis}
        )

        logger.info(f"Added problem {problem_id} to conversation {conv_id}")
        return problem_id

    def add_solution(
        self,
        solution: Dict,
        problem_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """Add a solution to the current conversation with enhanced tracking."""
        solution_id = self._generate_id("sol")
        
        solution_entry = {
            'id': solution_id,
            'timestamp': datetime.now().isoformat(),
            'solution': solution,
            'problem_id': problem_id,
            'feedback': None,
            'rating': None,
            'metadata': metadata or {}
        }

        with self._lock:
            conv_id = self.current_conversation_id
            if conv_id:
                self.conversations[conv_id]['solutions'].append(solution_entry)
                
                # Update problem status if problem_id provided
                if problem_id:
                    for problem in self.conversations[conv_id]['problems']:
                        if problem['id'] == problem_id:
                            problem['solutions_tried'].append(solution_id)
                            problem['attempts'] += 1
                            break

        # Add as message
        self.add_message(
            f"Solution: {solution.get('explanation', 'N/A')[:200]}...",
            MessageType.SOLUTION,
            metadata={'solution_id': solution_id, 'problem_id': problem_id}
        )

        logger.info(f"Added solution {solution_id} to conversation {conv_id}")
        return solution_id

    def add_feedback(
        self,
        message_id: str,
        feedback: str,
        rating: Optional[int] = None,
        metadata: Optional[Dict] = None
    ) -> bool:
        """Add feedback for a specific message."""
        try:
            with self._lock:
                conv_id = self.current_conversation_id
                if not conv_id:
                    return False
                
                for msg in self.conversations[conv_id]['messages']:
                    if msg['id'] == message_id:
                        msg['feedback'] = {
                            'timestamp': datetime.now().isoformat(),
                            'text': feedback,
                            'rating': rating,
                            'metadata': metadata or {}
                        }
                        
                        self.add_message(
                            f"Feedback provided: {feedback[:100]}...",
                            MessageType.FEEDBACK,
                            metadata={'message_id': message_id, 'rating': rating}
                        )
                        
                        logger.info(f"Added feedback for message {message_id}")
                        return True
                
                return False
                
        except Exception as e:
            logger.error(f"Failed to add feedback: {e}")
            return False

    def edit_message(
        self,
        message_id: str,
        new_content: str,
        reason: Optional[str] = None
    ) -> bool:
        """Edit an existing message."""
        try:
            with self._lock:
                conv_id = self.current_conversation_id
                if not conv_id:
                    return False
                
                for msg in self.conversations[conv_id]['messages']:
                    if msg['id'] == message_id:
                        # Store edit history
                        edit_entry = {
                            'timestamp': datetime.now().isoformat(),
                            'old_content': self._decompress_content(msg['content']),
                            'reason': reason
                        }
                        
                        if 'edit_history' not in msg:
                            msg['edit_history'] = []
                        
                        edit_history = json.loads(msg.get('edit_history', '[]'))
                        edit_history.append(edit_entry)
                        
                        # Update message
                        msg['content'] = self._compress_content(new_content)
                        msg['edited'] = True
                        msg['edit_history'] = json.dumps(edit_history)
                        msg['timestamp'] = datetime.now().isoformat()
                        
                        logger.info(f"Edited message {message_id}")
                        return True
                
                return False
                
        except Exception as e:
            logger.error(f"Failed to edit message: {e}")
            return False

    def _update_topics(self, conversation_id: str, message_id: str, topics: List[str]):
        """Update topics in database."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                for topic in topics:
                    # Insert or update topic
                    cursor.execute('''
                        INSERT INTO topics (name, last_seen)
                        VALUES (?, ?)
                        ON CONFLICT(name) DO UPDATE SET
                            count = count + 1,
                            last_seen = ?
                    ''', (topic, datetime.now(), datetime.now()))
                    
                    # Get topic ID
                    cursor.execute('SELECT id FROM topics WHERE name = ?', (topic,))
                    topic_id = cursor.fetchone()['id']
                    
                    # Link message to topic
                    cursor.execute('''
                        INSERT INTO message_topics (message_id, topic_id, confidence)
                        VALUES (?, ?, ?)
                    ''', (message_id, topic_id, 1.0))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to update topics: {e}")

    def _update_search_index(self, conversation_id: str, message_id: str, content: str):
        """Update search index."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO messages_fts (content, conversation_id, message_id)
                    VALUES (?, ?, ?)
                ''', (content, conversation_id, message_id))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to update search index: {e}")

    def search_messages(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict]:
        """Search messages using full-text search."""
        if not self.enable_search:
            logger.warning("Search is disabled")
            return []
        
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                if conversation_id:
                    cursor.execute('''
                        SELECT m.*, fts.conversation_id
                        FROM messages_fts fts
                        JOIN messages m ON m.id = fts.message_id
                        WHERE fts.content MATCH ? AND fts.conversation_id = ?
                        ORDER BY rank
                        LIMIT ?
                    ''', (query, conversation_id, limit))
                else:
                    cursor.execute('''
                        SELECT m.*, fts.conversation_id
                        FROM messages_fts fts
                        JOIN messages m ON m.id = fts.message_id
                        WHERE fts.content MATCH ?
                        ORDER BY rank
                        LIMIT ?
                    ''', (query, limit))
                
                results = []
                for row in cursor.fetchall():
                    result = dict(row)
                    result['content'] = self._decompress_content(result['content'])
                    results.append(result)
                
                return results
                
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def get_conversation_history(
        self,
        conversation_id: Optional[str] = None,
        include_decompressed: bool = True
    ) -> List[Dict]:
        """Get conversation history with decompression option."""
        conv_id = conversation_id or self.current_conversation_id
        if not conv_id or conv_id not in self.conversations:
            return []
        
        messages = self.conversations[conv_id]['messages']
        
        if include_decompressed:
            decompressed = []
            for msg in messages:
                msg_copy = msg.copy()
                msg_copy['content'] = self._decompress_content(msg['content'])
                decompressed.append(msg_copy)
            return decompressed
        
        return messages

    def get_recent_context(
        self,
        max_messages: int = 10,
        conversation_id: Optional[str] = None,
        include_system: bool = False
    ) -> List[Dict]:
        """Get recent conversation context with filtering."""
        conv_id = conversation_id or self.current_conversation_id
        if not conv_id or conv_id not in self.conversations:
            return []
        
        messages = self.conversations[conv_id]['messages'][-max_messages:]
        
        context = []
        for msg in messages:
            msg_copy = {
                'id': msg['id'],
                'timestamp': msg['timestamp'],
                'role': msg['role'],
                'content': self._decompress_content(msg['content']),
                'sentiment': msg.get('sentiment'),
                'tokens': msg.get('tokens')
            }
            
            if include_system or msg['role'] not in ['system', 'debug']:
                context.append(msg_copy)
        
        return context

    def get_current_problem(self, conversation_id: Optional[str] = None) -> Optional[Dict]:
        """Get the current active problem."""
        conv_id = conversation_id or self.current_conversation_id
        if not conv_id or conv_id not in self.conversations:
            return None
        
        problems = self.conversations[conv_id]['problems']
        if problems:
            # Return the most recent open problem
            for problem in reversed(problems):
                if problem.get('status') == 'open':
                    return problem
            return problems[-1]
        
        return None

    def get_current_solution(self, conversation_id: Optional[str] = None) -> Optional[Dict]:
        """Get the current active solution."""
        conv_id = conversation_id or self.current_conversation_id
        if not conv_id or conv_id not in self.conversations:
            return None
        
        solutions = self.conversations[conv_id]['solutions']
        return solutions[-1] if solutions else None

    def generate_summary(self, conversation_id: Optional[str] = None) -> Optional[ConversationSummary]:
        """Generate a comprehensive summary of the conversation."""
        conv_id = conversation_id or self.current_conversation_id
        if not conv_id or conv_id not in self.conversations:
            return None
        
        conv = self.conversations[conv_id]
        
        # Calculate statistics
        total_messages = len(conv['messages'])
        total_problems = len(conv['problems'])
        total_solutions = len(conv['solutions'])
        total_tokens = conv['total_tokens']
                # Calculate duration
        start_time = datetime.fromisoformat(conv['start_time']) if isinstance(conv['start_time'], str) else conv['start_time']
        end_time = None
        if conv['end_time']:
            end_time = datetime.fromisoformat(conv['end_time']) if isinstance(conv['end_time'], str) else conv['end_time']
        
        duration = end_time - start_time if end_time else datetime.now() - start_time
        
        # Analyze sentiment trend
        sentiment_trend = []
        for msg in conv['messages'][-20:]:  # Last 20 messages
            if msg.get('sentiment'):
                sentiment_trend.append({
                    'timestamp': msg['timestamp'],
                    'sentiment': msg['sentiment']
                })
        
        # Extract all topics
        all_topics = set()
        for msg in conv['messages']:
            if msg.get('topics'):
                topics = json.loads(msg['topics'])
                all_topics.update(topics)
        
        # Create summary
        summary = ConversationSummary(
            id=conv_id,
            start_time=start_time,
            end_time=end_time,
            state=ConversationState(conv['state']) if isinstance(conv['state'], str) else conv['state'],
            total_messages=total_messages,
            total_problems=total_problems,
            total_solutions=total_solutions,
            total_tokens=total_tokens,
            duration=duration,
            participants=conv['participants'],
            topics=list(all_topics),
            sentiment_trend=sentiment_trend,
            tags=conv['tags'],
            rating=None,
            feedback=None
        )
        
        # Store summary
        conv['summary'] = asdict(summary)
        
        return summary

    def export_conversation(
        self,
        format: str = 'json',
        conversation_id: Optional[str] = None,
        include_metadata: bool = True
    ) -> Union[str, Dict, None]:
        """Export conversation in various formats."""
        conv_id = conversation_id or self.current_conversation_id
        if not conv_id or conv_id not in self.conversations:
            return None
        
        conv = self.conversations[conv_id].copy()
        
        # Decompress messages
        for msg in conv['messages']:
            msg['content'] = self._decompress_content(msg['content'])
        
        if format == 'json':
            return conv
        
        elif format == 'text':
            lines = []
            lines.append(f"=== Conversation: {conv_id} ===\n")
            lines.append(f"Started: {conv['start_time']}")
            lines.append(f"State: {conv['state']}")
            lines.append(f"Messages: {len(conv['messages'])}")
            lines.append(f"Problems: {len(conv['problems'])}")
            lines.append(f"Solutions: {len(conv['solutions'])}\n")
            
            for msg in conv['messages']:
                role = msg['role'].upper()
                time = msg['timestamp'][11:19] if isinstance(msg['timestamp'], str) else msg['timestamp'].strftime('%H:%M:%S')
                content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                lines.append(f"[{time}] {role}: {content}")
            
            return "\n".join(lines)
        
        elif format == 'markdown':
            lines = []
            lines.append(f"# Conversation: {conv_id}\n")
            lines.append(f"- **Started:** {conv['start_time']}")
            lines.append(f"- **State:** {conv['state']}")
            lines.append(f"- **Messages:** {len(conv['messages'])}")
            lines.append(f"- **Problems:** {len(conv['problems'])}")
            lines.append(f"- **Solutions:** {len(conv['solutions'])}\n")
            
            for msg in conv['messages']:
                role = msg['role'].capitalize()
                content = msg['content']
                lines.append(f"### {role}")
                lines.append(f"*{msg['timestamp']}*")
                lines.append(f"\n{content}\n")
                lines.append("---\n")
            
            return "\n".join(lines)
        
        return None

    def save_conversation(self, conversation_id: Optional[str] = None) -> bool:
        """Save conversation to both file and database."""
        conv_id = conversation_id or self.current_conversation_id
        if not conv_id or conv_id not in self.conversations:
            return False
        
        try:
            conv = self.conversations[conv_id]
                        # Save to JSON file
            filename = f"{conv_id}.json"
            filepath = self.storage_path / filename
            
            # Prepare for JSON serialization
            conv_copy = conv.copy()
            conv_copy['start_time'] = conv_copy['start_time'].isoformat() if isinstance(conv_copy['start_time'], datetime) else conv_copy['start_time']
            if conv_copy['end_time']:
                conv_copy['end_time'] = conv_copy['end_time'].isoformat() if isinstance(conv_copy['end_time'], datetime) else conv_copy['end_time']
            
            # Convert messages for JSON
            conv_copy['messages'] = []
            for msg in conv['messages']:
                msg_copy = msg.copy()
                msg_copy['content'] = self._decompress_content(msg['content'])
                conv_copy['messages'].append(msg_copy)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(conv_copy, f, indent=2, ensure_ascii=False, default=str)
            
            # Save to database
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Insert/update conversation
                cursor.execute('''
                    INSERT OR REPLACE INTO conversations
                    (id, start_time, end_time, state, summary, metadata, tags, rating, feedback)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    conv_id,
                    conv['start_time'].isoformat() if isinstance(conv['start_time'], datetime) else conv['start_time'],
                    conv['end_time'].isoformat() if conv['end_time'] and isinstance(conv['end_time'], datetime) else conv['end_time'],
                    conv['state'].value if isinstance(conv['state'], ConversationState) else conv['state'],
                    json.dumps(conv.get('summary', {})),
                    json.dumps(conv.get('metadata', {})),
                    json.dumps(conv.get('tags', [])),
                    None,
                    None
                ))
                
                # Insert messages
                for msg in conv['messages']:
                    cursor.execute('''
                        INSERT OR REPLACE INTO messages
                        (id, conversation_id, timestamp, role, content, metadata, tokens, sentiment, topics, references, attachments)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        msg['id'],
                        conv_id,
                        msg['timestamp'],
                        msg['role'],
                        msg['content'],  # Already decompressed
                        json.dumps(msg.get('metadata', {})),
                        msg.get('tokens'),
                        msg.get('sentiment'),
                        msg.get('topics', '[]'),
                        msg.get('references', '[]'),
                        msg.get('attachments', '[]')
                    ))
                
                conn.commit()
            
            logger.info(f"Saved conversation {conv_id} to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save conversation {conv_id}: {e}")
            return False

    def load_conversation(self, conversation_id: str) -> bool:
        """Load a conversation from disk."""
        try:
            # Try to load from JSON file
            filepath = self.storage_path / f"{conversation_id}.json"
            
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    conv_data = json.load(f)
                
                # Parse datetime strings
                conv_data['start_time'] = datetime.fromisoformat(conv_data['start_time'])
                if conv_data.get('end_time'):
                    conv_data['end_time'] = datetime.fromisoformat(conv_data['end_time'])
                
                # Parse state
                if isinstance(conv_data['state'], str):
                    try:
                        conv_data['state'] = ConversationState(conv_data['state'])
                    except ValueError:
                        conv_data['state'] = ConversationState.ACTIVE
                
                # Store in cache
                self.conversations[conversation_id] = conv_data
                self.current_conversation_id = conversation_id
                
                logger.info(f"Loaded conversation {conversation_id} from file")
                return True
            
            # Try to load from database
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Load conversation
                cursor.execute('SELECT * FROM conversations WHERE id = ?', (conversation_id,))
                conv_row = cursor.fetchone()
                
                if not conv_row:
                    logger.warning(f"Conversation {conversation_id} not found")
                    return False
                
                # Load messages
                cursor.execute('SELECT * FROM messages WHERE conversation_id = ? ORDER BY timestamp', (conversation_id,))
                messages = cursor.fetchall()
                
                # Build conversation object
                conv_data = {
                    'id': conv_row['id'],
                    'start_time': datetime.fromisoformat(conv_row['start_time']),
                    'end_time': datetime.fromisoformat(conv_row['end_time']) if conv_row['end_time'] else None,
                    'state': ConversationState(conv_row['state']),
                    'messages': [dict(msg) for msg in messages],
                    'problems': [],  # TODO: Load from separate table if needed
                    'solutions': [],  # TODO: Load from separate table if needed
                    'metadata': json.loads(conv_row['metadata']) if conv_row['metadata'] else {},
                    'tags': json.loads(conv_row['tags']) if conv_row['tags'] else [],
                    'participants': ['user', 'assistant'],  # Default
                    'summary': json.loads(conv_row['summary']) if conv_row['summary'] else None,
                    'total_tokens': sum(msg.get('tokens', 0) for msg in messages)
                }
                
                self.conversations[conversation_id] = conv_data
                self.current_conversation_id = conversation_id
                
                logger.info(f"Loaded conversation {conversation_id} from database")
                return True
            
        except Exception as e:
            logger.error(f"Failed to load conversation {conversation_id}: {e}")
            return False

    def list_conversations(
        self,
        limit: int = 50,
        offset: int = 0,
        state: Optional[ConversationState] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dict]:
        """List all saved conversations with filtering."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                query = "SELECT * FROM conversations"
                params = []
                
                conditions = []
                if state:
                    conditions.append("state = ?")
                    params.append(state.value)
                
                if tags:
                    # Simple tag matching (can be enhanced)
                    conditions.append("tags LIKE ?")
                    params.append(f"%{tags[0]}%")
                
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
                
                query += " ORDER BY start_time DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])
                
                cursor.execute(query, params)
                
                conversations = []
                for row in cursor.fetchall():
                    conv = dict(row)
                    conv['metadata'] = json.loads(conv['metadata']) if conv['metadata'] else {}
                    conv['tags'] = json.loads(conv['tags']) if conv['tags'] else []
                    conversations.append(conv)
                
                return conversations
                
        except Exception as e:
            logger.error(f"Failed to list conversations: {e}")
            return []

    def get_conversation_summary(self, conversation_id: Optional[str] = None) -> Optional[Dict]:
        """Get a summary of the conversation."""
        conv_id = conversation_id or self.current_conversation_id
        if not conv_id or conv_id not in self.conversations:
            return None
        
        conv = self.conversations[conv_id]
        
        # Generate or use existing summary
        if not conv.get('summary'):
            summary = self.generate_summary(conv_id)
            if summary:
                return asdict(summary)
        
        return conv.get('summary')

    def close_conversation(
        self,
        conversation_id: Optional[str] = None,
        rating: Optional[int] = None,
        feedback: Optional[str] = None
    ) -> bool:
        """Close a conversation with optional feedback."""
        conv_id = conversation_id or self.current_conversation_id
        if not conv_id or conv_id not in self.conversations:
            return False
        
        with self._lock:
            conv = self.conversations[conv_id]
            conv['end_time'] = datetime.now()
            conv['state'] = ConversationState.COMPLETED
            
            # Generate final summary
            self.generate_summary(conv_id)
            
            # Add closing message
            self.add_message(
                "Conversation ended",
                MessageType.SYSTEM,
                metadata={'event': 'conversation_end', 'rating': rating}
            )
            
            # Save with feedback
            if rating is not None:
                conv['rating'] = rating
            if feedback:
                conv['feedback'] = feedback
            
            self.save_conversation(conv_id)
            
            logger.info(f"Closed conversation {conv_id}")
            return True

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation."""
        try:
            # Remove from cache
            if conversation_id in self.conversations:
                del self.conversations[conversation_id]
            
            # Delete from database
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM messages WHERE conversation_id = ?', (conversation_id,))
                cursor.execute('DELETE FROM conversations WHERE id = ?', (conversation_id,))
                conn.commit()
            
            # Delete JSON file
            filepath = self.storage_path / f"{conversation_id}.json"
            if filepath.exists():
                filepath.unlink()
            
            logger.info(f"Deleted conversation {conversation_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete conversation {conversation_id}: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall conversation statistics."""
        stats = {
            'total_conversations': 0,
            'total_messages': 0,
            'total_problems': 0,
            'total_solutions': 0,
            'total_tokens': 0,
            'average_messages_per_conversation': 0,
            'average_problems_per_conversation': 0,
            'average_solutions_per_conversation': 0,
            'top_topics': [],
            'sentiment_distribution': defaultdict(int),
            'conversations_by_state': defaultdict(int),
            'average_duration': None,
            'total_duration': timedelta()
        }
        
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Basic counts
                cursor.execute('SELECT COUNT(*) FROM conversations')
                stats['total_conversations'] = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(*) FROM messages')
                stats['total_messages'] = cursor.fetchone()[0]
                
                # Calculate averages
                if stats['total_conversations'] > 0:
                    stats['average_messages_per_conversation'] = stats['total_messages'] / stats['total_conversations']
                
                # Get top topics
                if self.enable_topics:
                    cursor.execute('''
                        SELECT name, count FROM topics 
                        ORDER BY count DESC LIMIT 10
                    ''')
                    stats['top_topics'] = [dict(row) for row in cursor.fetchall()]
                
                # Get sentiment distribution
                cursor.execute('''
                    SELECT sentiment, COUNT(*) as count 
                    FROM messages 
                    WHERE sentiment IS NOT NULL 
                    GROUP BY sentiment
                ''')
                for row in cursor.fetchall():
                    stats['sentiment_distribution'][row['sentiment']] = row['count']
                
                # Get conversations by state
                cursor.execute('''
                    SELECT state, COUNT(*) as count 
                    FROM conversations 
                    GROUP BY state
                ''')
                for row in cursor.fetchall():
                    stats['conversations_by_state'][row['state']] = row['count']
                
                # Calculate durations
                cursor.execute('''
                    SELECT julianday(end_time) - julianday(start_time) as duration
                    FROM conversations
                    WHERE end_time IS NOT NULL
                ''')
                durations = [row[0] for row in cursor.fetchall() if row[0]]
                if durations:
                    avg_duration_days = sum(durations) / len(durations)
                    stats['average_duration'] = timedelta(days=avg_duration_days)
                    stats['total_duration'] = timedelta(days=sum(durations))
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return stats

    def export_all_conversations(self, format: str = 'json') -> bool:
        """Export all conversations to a single file."""
        try:
            all_convs = []
            conversations = self.list_conversations(limit=1000)
            
            for conv in conversations:
                self.load_conversation(conv['id'])
                conv_data = self.export_conversation(format='json', conversation_id=conv['id'])
                if conv_data:
                    all_convs.append(conv_data)
            
            if format == 'json':
                output_file = self.storage_path / "all_conversations.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(all_convs, f, indent=2, ensure_ascii=False, default=str)
            
            elif format == 'markdown':
                output_file = self.storage_path / "all_conversations.md"
                with open(output_file, 'w', encoding='utf-8') as f:
                    for conv in all_convs:
                        f.write(self.export_conversation(format='markdown', conversation_id=conv['id']))
                        f.write("\n\n---\n\n")
            
            logger.info(f"Exported {len(all_convs)} conversations to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export all conversations: {e}")
            return False

    def cleanup_old_conversations(self, days: int = 30) -> int:
        """Clean up conversations older than specified days."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id FROM conversations 
                    WHERE start_time < ?
                ''', (cutoff_date.isoformat(),))
                
                old_convs = [row['id'] for row in cursor.fetchall()]
                
                for conv_id in old_convs:
                    self.delete_conversation(conv_id)
                
                logger.info(f"Cleaned up {len(old_convs)} conversations older than {days} days")
                return len(old_convs)
                
        except Exception as e:
            logger.error(f"Failed to clean up conversations: {e}")
            return 0


