"""
Jarvis AI Assistant Main Entry Point
"""

import os
import sys
import yaml
import logging
from typing import Optional

# Add src directory to path for imports
sys.path.append(os.path.dirname(__file__))

from ai import AIAssistant
from speech import SpeechHandler
from commands import CommandHandler

def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

def setup_logging(config: dict):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, config['logging']['level']),
        format=config['logging']['format'],
        handlers=[
            logging.FileHandler(config['logging']['file']),
            logging.StreamHandler()
        ]
    )

def handle_command(command: str, ai_assistant: AIAssistant, command_handler: CommandHandler, speech_handler: SpeechHandler):
    """
    Handle user command - either built-in or AI response
    
    Args:
        command: User command/input
        ai_assistant: AI assistant instance
        command_handler: Command handler instance
        speech_handler: Speech handler instance
    """
    # First try to process as built-in command
    command_handler.process_command(command)
    
    # If not a built-in command, get AI response
    if command.strip() and not any(cmd in command.lower() for cmd in command_handler.commands.keys()):
        response = ai_assistant.generate_response(command)
        print(f"[AI] {response}")
        speech_handler.speak(response)

def text_mode_loop(ai_assistant: AIAssistant, command_handler: CommandHandler, speech_handler: SpeechHandler):
    """Main loop for text input mode"""
    print("[TEXT] Text mode activated. Type 'voice' to switch to voice mode, 'quit' to exit.")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'voice':
                voice_mode_loop(ai_assistant, command_handler, speech_handler)
                print("[TEXT] Returned to text mode. Type 'voice' to switch to voice mode, 'quit' to exit.")
                continue
            
            handle_command(user_input, ai_assistant, command_handler, speech_handler)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

def voice_mode_loop(ai_assistant: AIAssistant, command_handler: CommandHandler, speech_handler: SpeechHandler):
    """Main loop for voice input mode"""
    print("[VOICE] Voice mode activated. Say your command after the wake word.")
    
    # Check if voice mode is actually enabled
    if not speech_handler.enabled:
        print("[ERROR] Falling back to text mode...")
        return
    
    def on_command_recognized(command: str):
        print(f"You said: {command}")
        handle_command(command, ai_assistant, command_handler, speech_handler)
    
    try:
        if not speech_handler.start_voice_mode(on_command_recognized):
            print("Failed to start voice mode. Returning to text mode.")
            return
    except KeyboardInterrupt:
        speech_handler.stop_voice_mode()
        return

def main():
    """Main entry point"""
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    config = load_config(config_path)
    
    # Setup logging
    setup_logging(config)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Jarvis AI Assistant")
    
    # Initialize components
    ai_assistant = AIAssistant(config)
    speech_handler = SpeechHandler(config)
    command_handler = CommandHandler(config, ai_assistant)
    
    # Welcome message
    welcome_msg = f"Hello! I'm {config['assistant']['name']} v{config['assistant']['version']}. How can I help you?"
    print(f"[ASSISTANT] {welcome_msg}")
    speech_handler.speak(welcome_msg)
    
    # Choose input mode
    print("Choose input mode:")
    print("1. Text mode")
    print("2. Voice mode")
    
    while True:
        choice = input("Enter choice (1 or 2): ").strip()
        if choice == '1':
            text_mode_loop(ai_assistant, command_handler, speech_handler)
            break
        elif choice == '2':
            voice_mode_loop(ai_assistant, command_handler, speech_handler)
            # If voice mode falls back or fails, continue to text mode
            print("Text mode activated. Type 'voice' to switch to voice mode, 'quit' to exit.")
            text_mode_loop(ai_assistant, command_handler, speech_handler)
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")
    
    logger.info("Jarvis AI Assistant shutting down")

if __name__ == "__main__":
    main()
