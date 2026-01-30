"""
AI Assistant Module
"""

import os
import logging
from typing import List, Dict
from openai import OpenAI

logger = logging.getLogger(__name__)


class AIAssistant:
    """AI Assistant for natural language processing"""
    
    def __init__(self, config: dict):
        self.config = config
        self.model = config['openai']['model']
        self.temperature = config['openai']['temperature']
        self.max_tokens = config['openai']['max_tokens']
        
        # Get API key
        api_key = config['openai']['api_key']
        if not api_key:
            api_key = os.environ.get('OPENAI_API_KEY')
        
        if not api_key:
            logger.warning("No OpenAI API key provided. AI features will be limited.")
            self.client = None
        else:
            try:
                self.client = OpenAI(api_key=api_key)
                logger.info(f"AI Assistant initialized with model: {self.model}")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                self.client = None
        
        # Conversation history
        self.conversation_history: List[Dict[str, str]] = []
        self.max_history = config['assistant']['max_conversation_history']
        
        # System prompt
        self.system_prompt = """You are Jarvis, a helpful AI assistant. 
You are friendly, professional, and knowledgeable about technology, programming, 
and general knowledge. Keep responses concise but helpful."""
    
    def chat(self, user_input: str) -> str:
        """Generate AI response"""
        # Check if AI is available
        if not self.client:
            return "⚠️ AI is not available. Please check your OpenAI API key configuration."
        
        try:
            # Prepare messages
            messages = [{"role": "system", "content": self.system_prompt}]
            
            # Add conversation history
            if self.conversation_history:
                recent_history = self.conversation_history[-self.max_history:]
                messages.extend(recent_history)
            
            # Add current user input
            messages.append({"role": "user", "content": user_input})
            
            # Generate response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            ai_response = response.choices[0].message.content
            
            # Update history
            self.conversation_history.append({"role": "user", "content": user_input})
            self.conversation_history.append({"role": "assistant", "content": ai_response})
            
            # Trim history if too long
            if len(self.conversation_history) > self.max_history * 2:
                self.conversation_history = self.conversation_history[-(self.max_history * 2):]
            
            return ai_response
            
        except Exception as e:
            logger.error(f"AI error: {e}")
            return f"⚠️ Error: {str(e)}"
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def is_available(self) -> bool:
        """Check if AI is available"""
        return self.client is not None   
