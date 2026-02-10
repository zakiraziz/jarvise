"""
AI Assistant Module
Enhanced with multiple AI providers, tools, and advanced capabilities
"""

import os
import json
import logging
import hashlib
import asyncio
import threading
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import pickle
from concurrent.futures import ThreadPoolExecutor

# Try to import various AI providers
try:
    from openai import OpenAI, AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False

try:
    from anthropic import Anthropic, AsyncAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

logger = logging.getLogger(__name__)


class AIProvider(Enum):
    """Supported AI providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OLLAMA = "ollama"
    LM_STUDIO = "lm_studio"


class AITool(Enum):
    """Available AI tools"""
    CALCULATOR = "calculator"
    WEB_SEARCH = "web_search"
    WEATHER = "weather"
    FILE_READER = "file_reader"
    CODE_EXECUTOR = "code_executor"
    KNOWLEDGE_BASE = "knowledge_base"


@dataclass
class AIMessage:
    """Structured AI message"""
    role: str  # "system", "user", "assistant", "tool"
    content: str
    timestamp: datetime = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AIResponse:
    """Structured AI response"""
    content: str
    provider: str
    model: str
    tokens_used: int = 0
    processing_time: float = 0.0
    tools_called: List[Dict] = None
    confidence: float = 0.0
    raw_response: Any = None
    
    def __post_init__(self):
        if self.tools_called is None:
            self.tools_called = []


@dataclass
class AIConfig:
    """AI Configuration"""
    # Provider settings
    default_provider: AIProvider = AIProvider.OPENAI
    fallback_providers: List[AIProvider] = None
    
    # Model settings
    openai_model: str = "gpt-3.5-turbo"
    anthropic_model: str = "claude-3-haiku-20240307"
    google_model: str = "gemini-pro"
    ollama_model: str = "llama2"
    lm_studio_model: str = "local-model"
    
    # API keys
    openai_api_key: str = None
    anthropic_api_key: str = None
    google_api_key: str = None
    
    # Performance settings
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    
    # Conversation settings
    max_history: int = 20
    enable_context_compression: bool = True
    context_compression_threshold: int = 4000
    
    # Tool settings
    enabled_tools: List[AITool] = None
    max_tool_calls: int = 3
    
    # System prompt
    system_prompt: str = """You are Jarvis, an advanced AI assistant with multiple capabilities.

PERSONALITY:
- Friendly, professional, and slightly witty
- Knowledgeable about technology, science, programming, and general knowledge
- Helpful but concise - get to the point
- Can use tools when needed

CAPABILITIES:
1. Answer questions accurately
2. Help with programming and technical problems
3. Use tools (calculator, web search, etc.) when appropriate
4. Remember conversation context
5. Provide step-by-step explanations when needed

RESPONSE GUIDELINES:
- Keep responses clear and structured when complex
- Use markdown formatting for code, lists, and emphasis
- Admit when you don't know something
- Ask clarifying questions when needed
- Use tools to get current information when relevant"""

    # Cache settings
    enable_cache: bool = True
    cache_ttl_hours: int = 24
    cache_path: str = "./ai_cache"
    
    # Rate limiting
    max_requests_per_minute: int = 60
    request_timeout: int = 30
    
    def __post_init__(self):
        if self.fallback_providers is None:
            self.fallback_providers = [AIProvider.ANTHROPIC, AIProvider.GOOGLE]
        if self.enabled_tools is None:
            self.enabled_tools = [AITool.CALCULATOR, AITool.WEB_SEARCH]


class AIAssistant:
    """Advanced AI Assistant with multiple providers and tools"""
    
    def __init__(self, config: AIConfig):
        self.config = config
        self.clients = {}
        self.conversation_history: List[AIMessage] = []
        self.cache = {}
        self.tools = {}
        self.rate_limiter = RateLimiter(config.max_requests_per_minute)
        
        # Initialize providers
        self._init_providers()
        
        # Initialize tools
        self._init_tools()
        
        # Load cache if enabled
        if config.enable_cache:
            self._load_cache()
        
        # Background thread for cache saving
        self.cache_save_thread = None
        self.cache_save_event = threading.Event()
        
        logger.info(f"AI Assistant initialized with provider: {config.default_provider.value}")
        logger.info(f"Enabled tools: {[t.value for t in config.enabled_tools]}")
    
    def _init_providers(self):
        """Initialize AI providers"""
        # OpenAI
        if OPENAI_AVAILABLE and self.config.openai_api_key:
            try:
                self.clients[AIProvider.OPENAI] = {
                    'sync': OpenAI(api_key=self.config.openai_api_key),
                    'async': AsyncOpenAI(api_key=self.config.openai_api_key)
                }
                logger.info("✅ OpenAI client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI: {e}")
        
        # Anthropic
        if ANTHROPIC_AVAILABLE and self.config.anthropic_api_key:
            try:
                self.clients[AIProvider.ANTHROPIC] = {
                    'sync': Anthropic(api_key=self.config.anthropic_api_key),
                    'async': AsyncAnthropic(api_key=self.config.anthropic_api_key)
                }
                logger.info("✅ Anthropic client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic: {e}")
        
        # Google AI
        if GOOGLE_AI_AVAILABLE and self.config.google_api_key:
            try:
                genai.configure(api_key=self.config.google_api_key)
                self.clients[AIProvider.GOOGLE] = genai
                logger.info("✅ Google AI client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Google AI: {e}")
        
