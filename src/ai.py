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
                # Check if at least one provider is available
        if not self.clients:
            logger.warning("⚠️ No AI providers available. Please configure API keys.")
    
    def _init_tools(self):
        """Initialize AI tools"""
        # Calculator tool
        self.tools[AITool.CALCULATOR] = {
            "function": self._tool_calculator,
            "description": "Perform mathematical calculations",
            "parameters": {
                "expression": "Mathematical expression to evaluate"
            }
        }
        
        # Web Search tool (placeholder - would need actual implementation)
        self.tools[AITool.WEB_SEARCH] = {
            "function": self._tool_web_search,
            "description": "Search the web for current information",
            "parameters": {
                "query": "Search query"
            }
        }
        
        # Weather tool
        self.tools[AITool.WEATHER] = {
            "function": self._tool_weather,
            "description": "Get current weather information",
            "parameters": {
                "location": "City name or coordinates"
            }
        }
        
        # File Reader tool
        self.tools[AITool.FILE_READER] = {
            "function": self._tool_file_reader,
            "description": "Read text files",
            "parameters": {
                "filepath": "Path to the file"
            }
        }
        
        logger.info(f"Initialized {len(self.tools)} tools")
    
    # Tool implementations
    def _tool_calculator(self, expression: str) -> str:
        """Evaluate mathematical expression"""
        try:
            # Safe evaluation of mathematical expressions
            import ast
            import operator
            
            # Define safe operations
            safe_ops = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.Pow: operator.pow,
                ast.USub: operator.neg,
                ast.Mod: operator.mod
            }
            
            def safe_eval(node):
                if isinstance(node, ast.Constant):
                    return node.value
                elif isinstance(node, ast.BinOp):
                    left = safe_eval(node.left)
                    right = safe_eval(node.right)
                    return safe_ops[type(node.op)](left, right)
                elif isinstance(node, ast.UnaryOp):
                    operand = safe_eval(node.operand)
                    return safe_ops[type(node.op)](operand)
                else:
                    raise ValueError(f"Unsupported operation: {type(node)}")
            
            tree = ast.parse(expression, mode='eval')
            result = safe_eval(tree.body)
            return f"Result: {result}"
            
        except Exception as e:
            return f"Calculation error: {e}"
    
    def _tool_web_search(self, query: str) -> str:
        """Search the web (placeholder - implement with actual search API)"""
        # This would integrate with a search API like Google Custom Search
        # For now, return a placeholder
        return f"Web search for '{query}' would be performed here. (Integration needed)"
    
    def _tool_weather(self, location: str) -> str:
        """Get weather information (placeholder)"""
        # Integrate with weather API like OpenWeatherMap
        return f"Weather for {location}: [Weather data would appear here with API integration]"
    
    def _tool_file_reader(self, filepath: str) -> str:
        """Read text file"""
        try:
            if not os.path.exists(filepath):
                return f"File not found: {filepath}"
            
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Limit content length
            if len(content) > 5000:
                content = content[:5000] + "\n...[truncated]"
            
            return f"File contents of {filepath}:\n\n{content}"
            
        except Exception as e:
            return f"Error reading file: {e}"
    
    def _generate_cache_key(self, messages: List[Dict]) -> str:
        """Generate cache key from messages"""
        content = json.dumps([{k: v for k, v in m.items() if k in ['role', 'content']} 
                             for m in messages], sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()
    
    def _load_cache(self):
        """Load cache from disk"""
        cache_file = os.path.join(self.config.cache_path, "ai_cache.pkl")
        try:
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                logger.info(f"Loaded cache with {len(self.cache)} entries")
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            self.cache = {}
    
    def _save_cache(self):
        """Save cache to disk"""
        if not self.config.enable_cache:
            return
        
        cache_file = os.path.join(self.config.cache_path, "ai_cache.pkl")
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        
        try:
            # Clean old cache entries
            now = datetime.now()
            self.cache = {k: v for k, v in self.cache.items() 
                         if (now - v['timestamp']).total_seconds() < self.config.cache_ttl_hours * 3600}
            
            with open(cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
            
            logger.debug(f"Cache saved with {len(self.cache)} entries")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def chat(self, user_input: str, use_tools: bool = True) -> AIResponse:
        """
        Generate AI response with optional tool usage
        
        Args:
            user_input: User's message
            use_tools: Whether to allow tool usage
            
        Returns:
            AIResponse object
        """
        start_time = datetime.now()
        
        # Check if any provider is available
        if not self.clients:
            return AIResponse(
                content="⚠️ AI is not available. Please check your API key configuration.",
                provider="none",
                model="none",
                processing_time=(datetime.now() - start_time).total_seconds()
            )
        
        # Rate limiting
        self.rate_limiter.wait()
        
        # Add user message to history
        self.conversation_history.append(AIMessage(
            role="user",
            content=user_input,
            metadata={"use_tools": use_tools}
        ))
        
        # Prepare messages for AI
        messages = self._prepare_messages(use_tools)
        
        # Check cache
        cache_key = None
        if self.config.enable_cache:
            cache_key = self._generate_cache_key(messages)
            if cache_key in self.cache:
                cached = self.cache[cache_key]
                logger.debug(f"Cache hit for query: {user_input[:50]}...")
                return AIResponse(**cached['response'])
        
        # Try providers in order
        providers_to_try = [self.config.default_provider] + self.config.fallback_providers
        last_error = None
        
        for provider in providers_to_try:
            if provider not in self.clients:
                continue
            
            try:
                response = self._call_provider(provider, messages, use_tools)
                
                # Add assistant response to history
                self.conversation_history.append(AIMessage(
                    role="assistant",
                    content=response.content,
                    metadata={
                        "provider": response.provider,
                        "model": response.model,
                        "tokens_used": response.tokens_used
                    }
                ))
                
                # Compress context if needed
                self._compress_context_if_needed()
                
                # Cache the response
                if cache_key and self.config.enable_cache:
                    self.cache[cache_key] = {
                        'response': asdict(response),
                        'timestamp': datetime.now()
                    }
                    # Start background save if not already running
                    self._schedule_cache_save()
                
                return response
                
            except Exception as e:
                last_error = e
                logger.warning(f"Provider {provider.value} failed: {e}")
                continue
        
