import openai
import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class AIAssistant:
    """General AI assistant using OpenAI for natural language processing."""

    def __init__(self, api_key: str, model: str = "gpt-4-turbo-preview"):
        self.api_key = api_key
        self.model = model
        self.client = openai.OpenAI(api_key=api_key)

    def process_query(self, query: str, context: Optional[str] = None) -> str:
        """
        Process a natural language query and return response.

        Args:
            query: User's query
            context: Optional context information

        Returns:
            AI response as string
        """
        try:
            messages = [
                {"role": "system", "content": "You are Jarvis, a helpful AI assistant. Respond naturally and assist with user requests."}
            ]

            if context:
                messages.append({"role": "system", "content": f"Context: {context}"})

            messages.append({"role": "user", "content": query})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1000,
                temperature=0.7,
                timeout=30
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"AI processing error: {e}")
            return "I'm sorry, I encountered an error processing your request."

    def decide_action(self, user_input: str) -> Dict:
        """
        Decide what action to take based on user input.

        Args:
            user_input: User's spoken or typed input

        Returns:
            Dict with action type and parameters
        """
        try:
            prompt = """
Analyze this user input and decide what action Jarvis should take.
Return a JSON-like response with 'action' and 'params' keys.

Actions:
- 'command': Execute a modular command (params: command_name, command_params)
- 'respond': Just respond with text (params: response_text)
- 'search': Perform web search (params: query)
- 'schedule': Schedule a task (params: task, time)

User input: "{}"

Response format: {{"action": "action_name", "params": {}}}
""".format(user_input)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an action decision engine. Analyze user input and return structured action decisions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.1,
                timeout=15
            )

            # Parse the response (simplified parsing)
            content = response.choices[0].message.content.strip()
            # For simplicity, assume it's a dict-like string
            # In real implementation, use proper JSON parsing
            if 'command' in content.lower():
                return {"action": "command", "params": {"command_name": "unknown", "command_params": ""}}
            elif 'search' in content.lower():
                return {"action": "search", "params": {"query": user_input}}
            elif 'schedule' in content.lower():
                return {"action": "schedule", "params": {"task": user_input, "time": "now"}}
            else:
                return {"action": "respond", "params": {"response_text": content}}

        except Exception as e:
            logger.error(f"Action decision error: {e}")
            return {"action": "respond", "params": {"response_text": "I didn't understand that request."}}