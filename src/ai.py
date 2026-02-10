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
                # All providers failed
        error_msg = f"All AI providers failed. Last error: {last_error}"
        logger.error(error_msg)
        
        return AIResponse(
            content=f"⚠️ Error: {error_msg}",
            provider="none",
            model="none",
            processing_time=(datetime.now() - start_time).total_seconds()
        )
    
    async def chat_async(self, user_input: str, use_tools: bool = True) -> AIResponse:
        """Async version of chat"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            lambda: self.chat(user_input, use_tools)
        )
    
    def _prepare_messages(self, use_tools: bool = True) -> List[Dict]:
        """Prepare messages for AI API call"""
        messages = [{"role": "system", "content": self.config.system_prompt}]
        
        # Add conversation history
        for msg in self.conversation_history[-self.config.max_history:]:
            messages.append({"role": msg.role, "content": msg.content})
        
        # Add tool descriptions if tools are enabled
        if use_tools and self.config.enabled_tools:
            tool_descriptions = "\n\nAVAILABLE TOOLS:\n"
            for tool in self.config.enabled_tools:
                if tool in self.tools:
                    desc = self.tools[tool]["description"]
                    tool_descriptions += f"- {tool.value}: {desc}\n"
            
            # Modify system prompt to include tools
            messages[0]["content"] += tool_descriptions
        
        return messages
    
    def _call_provider(self, provider: AIProvider, messages: List[Dict], use_tools: bool) -> AIResponse:
        """Call specific AI provider"""
        start_time = datetime.now()
        
        if provider == AIProvider.OPENAI:
            return self._call_openai(messages)
        elif provider == AIProvider.ANTHROPIC:
            return self._call_anthropic(messages)
        elif provider == AIProvider.GOOGLE:
            return self._call_google(messages)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _call_openai(self, messages: List[Dict]) -> AIResponse:
        """Call OpenAI API"""
        client = self.clients[AIProvider.OPENAI]['sync']
        
        response = client.chat.completions.create(
            model=self.config.openai_model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            frequency_penalty=self.config.frequency_penalty,
            presence_penalty=self.config.presence_penalty
        )
        
        return AIResponse(
            content=response.choices[0].message.content,
            provider=AIProvider.OPENAI.value,
            model=self.config.openai_model,
            tokens_used=response.usage.total_tokens if response.usage else 0,
            processing_time=(datetime.now() - datetime.fromtimestamp(start_time.timestamp())).total_seconds(),
            raw_response=response
        )
    
    def _call_anthropic(self, messages: List[Dict]) -> AIResponse:
        """Call Anthropic API"""
        client = self.clients[AIProvider.ANTHROPIC]['sync']
        
        # Convert messages to Anthropic format
        anthropic_messages = []
        for msg in messages:
            if msg["role"] == "system":
                continue  # System message handled separately
            anthropic_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        response = client.messages.create(
            model=self.config.anthropic_model,
            messages=anthropic_messages,
            system=self.config.system_prompt,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature
        )
        
        return AIResponse(
            content=response.content[0].text,
            provider=AIProvider.ANTHROPIC.value,
            model=self.config.anthropic_model,
            tokens_used=response.usage.input_tokens + response.usage.output_tokens,
            processing_time=(datetime.now() - datetime.fromtimestamp(start_time.timestamp())).total_seconds(),
            raw_response=response
        )
    
    def _call_google(self, messages: List[Dict]) -> AIResponse:
        """Call Google AI API"""
        model = self.clients[AIProvider.GOOGLE].GenerativeModel(self.config.google_model)
        
        # Convert messages to text
        conversation_text = ""
        for msg in messages:
            if msg["role"] == "system":
                conversation_text += f"System: {msg['content']}\n\n"
            else:
                conversation_text += f"{msg['role'].title()}: {msg['content']}\n\n"
        
        response = model.generate_content(
            conversation_text,
            generation_config={
                "temperature": self.config.temperature,
                "max_output_tokens": self.config.max_tokens,
                "top_p": self.config.top_p
            }
        )
        
        return AIResponse(
            content=response.text,
            provider=AIProvider.GOOGLE.value,
            model=self.config.google_model,
            processing_time=(datetime.now() - datetime.fromtimestamp(start_time.timestamp())).total_seconds(),
            raw_response=response
        )
    
    def _compress_context_if_needed(self):
        """Compress conversation history if it's too long"""
        if not self.config.enable_context_compression:
            return
        
        total_tokens = sum(len(msg.content.split()) for msg in self.conversation_history)
        
        if total_tokens > self.config.context_compression_threshold:
            logger.info("Compressing conversation context...")
            
            # Keep first and last few messages, summarize the middle
            if len(self.conversation_history) > 10:
                # Summarize middle messages
                middle_messages = self.conversation_history[4:-4]
                summary_prompt = "Summarize this conversation context briefly:\n\n"
                for msg in middle_messages:
                    summary_prompt += f"{msg.role}: {msg.content}\n"
                
                # Use AI to summarize (could be optimized)
                try:
                    summary_response = self.chat(summary_prompt, use_tools=False)
                    summary = f"[Compressed context: {summary_response.content[:200]}...]"
                    
                    # Replace middle messages with summary
                    self.conversation_history = (
                        self.conversation_history[:4] +
                        [AIMessage(role="system", content=summary)] +
                        self.conversation_history[-4:]
                    )
                except:
                    # If summarization fails, just truncate
                    self.conversation_history = (
                        self.conversation_history[:6] + 
                        self.conversation_history[-4:]
                    )
    
    def _schedule_cache_save(self):
        """Schedule cache save in background"""
        if self.cache_save_thread is None or not self.cache_save_thread.is_alive():
            self.cache_save_thread = threading.Thread(
                target=self._background_cache_save,
                daemon=True
            )
            self.cache_save_thread.start()
    
    def _background_cache_save(self):
        """Background thread for saving cache"""
        time.sleep(5)  # Wait a bit before saving
        self._save_cache()
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def get_history(self, limit: int = None) -> List[AIMessage]:
        """Get conversation history"""
        if limit:
            return self.conversation_history[-limit:]
        return self.conversation_history.copy()
    
    def export_history(self, filepath: str = None):
        """Export conversation history to JSON"""
        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"conversation_history_{timestamp}.json"
        
        try:
            history_data = []
            for msg in self.conversation_history:
                history_data.append({
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "metadata": msg.metadata
                })
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"History exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export history: {e}")
            return False
    
    def add_tool(self, tool_name: str, tool_func: Callable, description: str, parameters: Dict):
        """Add a custom tool"""
        tool_enum = AITool(tool_name) if tool_name in [t.value for t in AITool] else None
        
        if not tool_enum:
            # Create custom tool enum value
            tool_enum = AITool(tool_name)
        
        self.tools[tool_enum] = {
            "function": tool_func,
            "description": description,
            "parameters": parameters
        }
        
        logger.info(f"Added custom tool: {tool_name}")
    
    def get_available_providers(self) -> List[str]:
        """Get list of available AI providers"""
        return [p.value for p in self.clients.keys()]
    
    def get_stats(self) -> Dict:
        """Get assistant statistics"""
        total_tokens = sum(
            msg.metadata.get('tokens_used', 0) 
            for msg in self.conversation_history 
            if msg.role == "assistant"
        )
        
        return {
            "total_messages": len(self.conversation_history),
            "user_messages": sum(1 for msg in self.conversation_history if msg.role == "user"),
            "assistant_messages": sum(1 for msg in self.conversation_history if msg.role == "assistant"),
            "estimated_tokens": total_tokens,
            "cache_size": len(self.cache),
            "available_providers": self.get_available_providers(),
            "available_tools": [t.value for t in self.tools.keys()]
        }
    
    def is_available(self) -> bool:
        """Check if AI is available"""
        return len(self.clients) > 0
    
    def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up AI Assistant...")
        
        # Save cache
        if self.config.enable_cache:
            self._save_cache()
        
        # Stop background threads
        self.cache_save_event.set()
        
        logger.info("AI Assistant cleanup complete")


class RateLimiter:
    """Simple rate limiter for API calls"""
    
    def __init__(self, max_requests_per_minute: int):
        self.max_requests = max_requests_per_minute
        self.requests = []
        self.lock = threading.Lock()
    
    def wait(self):
        """Wait if rate limit would be exceeded"""
        with self.lock:
            now = time.time()
            
            # Remove old requests
            self.requests = [req_time for req_time in self.requests 
                           if now - req_time < 60]
            
            # Check if we can make a request
            if len(self.requests) >= self.max_requests:
                # Calculate wait time
                oldest = self.requests[0]
                wait_time = 60 - (now - oldest)
                if wait_time > 0:
                    time.sleep(wait_time)
                
                # Update list after waiting
                self.requests = [req_time for req_time in self.requests 
                               if now + wait_time - req_time < 60]
            
            # Add current request
            self.requests.append(time.time())


# Utility functions
def create_default_assistant() -> AIAssistant:
    """Create AI assistant with default configuration"""
    config = AIConfig(
        openai_api_key=os.environ.get('OPENAI_API_KEY'),
        anthropic_api_key=os.environ.get('ANTHROPIC_API_KEY'),
        google_api_key=os.environ.get('GOOGLE_AI_API_KEY'),
        default_provider=AIProvider.OPENAI,
        enabled_tools=[AITool.CALCULATOR, AITool.WEB_SEARCH, AITool.FILE_READER]
    )
    
    return AIAssistant(config)


def test_assistant():
    """Test function for the AI assistant"""
    assistant = create_default_assistant()
    
    if not assistant.is_available():
        print("⚠️ No AI providers available. Please set API keys.")
        print("Set environment variables:")
        print("  - OPENAI_API_KEY")
        print("  - ANTHROPIC_API_KEY (optional)")
        print("  - GOOGLE_AI_API_KEY (optional)")
        return
    
    print(f"✅ AI Assistant ready with providers: {assistant.get_available_providers()}")
    
    # Test conversation
    test_queries = [
        "What's 15 * 27?",
        "Who won the Nobel Prize in Physics in 2023?",
        "Write a Python function to calculate Fibonacci numbers"
    ]
    
    for query in test_queries:
        print(f"\n🧠 Query: {query}")
        response = assistant.chat(query)
        print(f"🤖 Response: {response.content[:200]}...")
        print(f"   Provider: {response.provider}, Tokens: {response.tokens_used}")
    
    # Show stats
    stats = assistant.get_stats()
    print(f"\n📊 Stats: {stats}")
    
    assistant.cleanup()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_assistant()

