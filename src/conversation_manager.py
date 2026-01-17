"""
Conversation Manager
Handles conversation history and context for interactive debugging sessions.
"""

import json
import logging
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class ConversationManager:
    """Manages conversation history and context."""

    def __init__(self, max_history: int = 50, storage_path: Optional[Path] = None):
        self.max_history = max_history
        self.storage_path = storage_path or Path("conversations")
        self.storage_path.mkdir(exist_ok=True)

        self.current_conversation = {
            'id': self._generate_conversation_id(),
            'start_time': datetime.now().isoformat(),
            'messages': [],
            'problems': [],
            'solutions': []
        }

    def _generate_conversation_id(self) -> str:
        """Generate a unique conversation ID."""
        return f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def add_problem(self, problem_text: str, analysis: Dict) -> None:
        """Add a problem to the current conversation."""
        problem_entry = {
            'timestamp': datetime.now().isoformat(),
            'text': problem_text,
            'analysis': analysis
        }

        self.current_conversation['problems'].append(problem_entry)
        self._add_message('user', problem_text)

        logger.info(f"Added problem to conversation {self.current_conversation['id']}")

    def add_solution(self, solution: Dict) -> None:
        """Add a solution to the current conversation."""
        solution_entry = {
            'timestamp': datetime.now().isoformat(),
            'solution': solution
        }

        self.current_conversation['solutions'].append(solution_entry)
        self._add_message('assistant', f"Generated solution: {solution.get('explanation', 'N/A')[:100]}...")

    def add_followup(self, question: str, answer: str) -> None:
        """Add a follow-up question and answer."""
        self._add_message('user', question)
        self._add_message('assistant', answer)

    def _add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history."""
        message = {
            'timestamp': datetime.now().isoformat(),
            'role': role,
            'content': content
        }

        self.current_conversation['messages'].append(message)

        # Maintain max history
        if len(self.current_conversation['messages']) > self.max_history:
            self.current_conversation['messages'] = self.current_conversation['messages'][-self.max_history:]

    def get_recent_context(self, max_messages: int = 10) -> List[Dict]:
        """Get recent conversation context."""
        return self.current_conversation['messages'][-max_messages:]

    def get_current_problem(self) -> Optional[Dict]:
        """Get the current active problem."""
        if self.current_conversation['problems']:
            return self.current_conversation['problems'][-1]
        return None

    def get_current_solution(self) -> Optional[Dict]:
        """Get the current active solution."""
        if self.current_conversation['solutions']:
            return self.current_conversation['solutions'][-1]
        return None

    def save_conversation(self) -> bool:
        """Save the current conversation to disk."""
        try:
            filename = f"{self.current_conversation['id']}.json"
            filepath = self.storage_path / filename

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.current_conversation, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved conversation to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to save conversation: {e}")
            return False

    def load_conversation(self, conversation_id: str) -> bool:
        """Load a conversation from disk."""
        try:
            filepath = self.storage_path / f"{conversation_id}.json"

            if not filepath.exists():
                logger.warning(f"Conversation {conversation_id} not found")
                return False

            with open(filepath, 'r', encoding='utf-8') as f:
                self.current_conversation = json.load(f)

            logger.info(f"Loaded conversation {conversation_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to load conversation: {e}")
            return False

    def list_conversations(self) -> List[Dict]:
        """List all saved conversations."""
        conversations = []

        try:
            for filepath in self.storage_path.glob("*.json"):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        conv = json.load(f)
                        conversations.append({
                            'id': conv['id'],
                            'start_time': conv['start_time'],
                            'problem_count': len(conv.get('problems', [])),
                            'message_count': len(conv.get('messages', []))
                        })
                except Exception as e:
                    logger.warning(f"Failed to read conversation file {filepath}: {e}")

        except Exception as e:
            logger.error(f"Failed to list conversations: {e}")

        return sorted(conversations, key=lambda x: x['start_time'], reverse=True)

    def new_conversation(self) -> str:
        """Start a new conversation."""
        # Save current conversation if it has content
        if self.current_conversation['messages'] or self.current_conversation['problems']:
            self.save_conversation()

        # Create new conversation
        old_id = self.current_conversation['id']
        self.current_conversation = {
            'id': self._generate_conversation_id(),
            'start_time': datetime.now().isoformat(),
            'messages': [],
            'problems': [],
            'solutions': []
        }

        logger.info(f"Started new conversation {self.current_conversation['id']} (previous: {old_id})")
        return self.current_conversation['id']

    def get_conversation_summary(self) -> Dict:
        """Get a summary of the current conversation."""
        return {
            'id': self.current_conversation['id'],
            'start_time': self.current_conversation['start_time'],
            'total_messages': len(self.current_conversation['messages']),
            'total_problems': len(self.current_conversation['problems']),
            'total_solutions': len(self.current_conversation['solutions']),
            'last_activity': self.current_conversation['messages'][-1]['timestamp'] if self.current_conversation['messages'] else None
        }