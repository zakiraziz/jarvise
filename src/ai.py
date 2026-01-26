"""
AI Assistant Module
Handles AI interactions using OpenAI API
"""

from openai import OpenAI
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class AIAssistant:
    def __init__(self, config: dict):
        """
        Initialize AI assistant

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.client = OpenAI(api_key=config['openai']['api_key'])
        self.model = config['openai']['model']
        self.temperature = config['openai']['temperature']
        self.max_tokens = config['openai']['max_tokens']
        self.conversation_history: List[Dict[str, str]] = []
        self.max_history = config['assistant']['max_conversation_history']

    def generate_response(self, user_input: str) -> str:
        """
        Generate AI response to user input

        Args:
            user_input: User's message

        Returns:
            AI response string
        """
        try:
            # Add user message to history
            self.conversation_history.append({"role": "user", "content": user_input})

            # Trim history if too long
            if len(self.conversation_history) > self.max_history:
                self.conversation_history = self.conversation_history[-self.max_history:]

            # Generate response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            ai_response = response.choices[0].message.content

            # Add AI response to history
            self.conversation_history.append({"role": "assistant", "content": ai_response})

            return ai_response

        except Exception as e:
            logger.error(f"Error generating AI response: {e}")
            return "Sorry, I encountered an error processing your request."

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
    
    def chat(self, user_input: str) -> str:
        """
        Alias for generate_response for backward compatibility
        
        Args:
            user_input: User's message
            
        Returns:
            AI response string
        """
        return self.generate_response(user_input)
    
    def get_history_summary(self) -> str:
        """
        Get a summary of conversation history
        
        Returns:
            Formatted conversation history string
        """
        if not self.conversation_history:
            return "No conversation history yet."
        
        summary = ""
        for i, msg in enumerate(self.conversation_history[-10:], 1):  # Show last 10 messages
            role = "You" if msg['role'] == 'user' else "Jarvis"
            content = msg['content'][:100]  # Truncate long messages
            summary += f"{i}. {role}: {content}...\n" if len(msg['content']) > 100 else f"{i}. {role}: {content}\n"
        return summary
    
    def analyze_code(self, code: str) -> str:
        """
        Analyze code and provide feedback
        
        Args:
            code: Code to analyze
            
        Returns:
            AI analysis string
        """
        prompt = f"Please analyze this code and provide feedback:\n\n{code}"
        return self.generate_response(prompt)
    
    def explain_concept(self, concept: str) -> str:
        """
        Explain a programming concept
        
        Args:
            concept: Concept to explain
            
        Returns:
            AI explanation string
        """
        prompt = f"Please explain the programming concept: {concept}"
        return self.generate_response(prompt)
