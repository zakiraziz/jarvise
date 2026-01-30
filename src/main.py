#!/usr/bin/env python3
"""
Jarvis AI Assistant - Main Entry Point
RUN THIS FILE TO START JARVIS!
"""

import os
import sys
import logging
import traceback
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from utils import ConfigManager, setup_logging
from ai import AIAssistant
from commands import CommandHandler
from speech import SpeechHandler


class JarvisAssistant:
    """Main Jarvis AI Assistant class"""
    
    def __init__(self):
        """Initialize the assistant"""
        self.logger = None
        self.config = None
        self.ai = None
        self.commands = None
        self.speech = None
        self.running = False
        
    def initialize(self):
        """Initialize all components"""
        try:
            # Create necessary directories
            for dir_name in ['logs', 'data', 'temp']:
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name, exist_ok=True)
            
            # Load configuration
            self.config = ConfigManager().config
            print(f"✅ Configuration loaded: {self.config['assistant']['name']} v{self.config['assistant']['version']}")
            
            # Setup logging
            self.logger = setup_logging(self.config)
            self.logger.info(f"Starting {self.config['assistant']['name']} v{self.config['assistant']['version']}")
            
            # Initialize AI assistant
            self.ai = AIAssistant(self.config)
            print("✅ AI Assistant initialized")
            self.logger.info("AI Assistant initialized")
            
            # Initialize command handler
            self.commands = CommandHandler(self.config, self.ai)
            print(f"✅ Command handler ready with {len(self.commands.get_commands())} commands")
            self.logger.info(f"Command handler ready with {len(self.commands.get_commands())} commands")
            
            # Initialize speech handler
            self.speech = SpeechHandler(self.config)
            print("✅ Speech handler initialized")
            self.logger.info("Speech handler initialized")
            
            return True
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            if self.logger:
                self.logger.error(f"Failed to initialize: {e}")
            return False
    
    def show_welcome(self):
        """Show welcome message"""
        print("\n" + "="*70)
        print(f"🤖 {self.config['assistant']['name']} AI Assistant v{self.config['assistant']['version']}")
        print("="*70)
        print("\nCommands: 'help' | 'menu' | 'quit'")
        print("Type anything to chat with AI!")
        print("-"*70)
        
        # Speak welcome message
        welcome_text = f"Hello! I am {self.config['assistant']['name']}. How can I assist you today?"
        print(f"🔊 {welcome_text}")
        self.speech.speak(welcome_text)
    
    def show_menu(self):
        """Show interactive menu"""
        print("\n" + "="*70)
        print("📋 JARVIS MAIN MENU")
        print("="*70)
        print("\n1. 💬 Chat Mode (Talk to AI)")
        print("2. ⚡ Command Mode (Quick commands)")
        print("3. 📝 Notes & Todo")
        print("4. 🛠️ System Tools")
        print("5. 🌐 Web & Search")
        print("6. ⚙️ Settings")
        print("7. ℹ️ Help")
        print("8. 🚪 Exit")
        print("-"*70)
        
        choice = input("\nSelect option (1-8): ").strip()
        
        if choice == "1":
            self.chat_mode()
        elif choice == "2":
            self.command_mode()
        elif choice == "3":
            self.notes_mode()
        elif choice == "4":
            self.system_mode()
        elif choice == "5":
            self.web_mode()
        elif choice == "6":
            self.settings_mode()
        elif choice == "7":
            self.show_help()
        elif choice == "8":
            return False
        else:
            print("❌ Invalid choice")
            
        return True
    
    def chat_mode(self):
        """Chat with AI"""
        print("\n" + "="*70)
        print("💬 CHAT MODE - Type 'back' to return to menu")
        print("="*70)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ['back', 'menu', 'exit']:
                    break
                
                # Process with AI
                print("🤔 Thinking...", end='\r')
                response = self.ai.chat(user_input)
                
                print("\n" + "-"*70)
                print(f"💡 {self.config['assistant']['name']}:")
                print("-"*70)
                print(response)
                print("-"*70)
                
                # Speak response
                self.speech.speak(response[:200])  # Limit speech length
                
            except KeyboardInterrupt:
                print("\n⏹️ Returning to menu...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def command_mode(self):
        """Quick command mode"""
        print("\n" + "="*70)
        print("⚡ COMMAND MODE - Type 'back' to return to menu")
        print("="*70)
        print("\nQuick commands:")
        print("  time / date / weather [city] / joke / quote")
        print("  open [app] / search [query] / calculate [math]")
        print("  system / ip / ping [host] / clear")
        print("-"*70)
        
        while True:
            try:
                user_input = input("\nCommand: ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ['back', 'menu', 'exit']:
                    break
                
                # Process command
                self.commands.process(user_input)
                
            except KeyboardInterrupt:
                print("\n⏹️ Returning to menu...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def notes_mode(self):
        """Notes and todo management"""
        print("\n" + "="*70)
        print("📝 NOTES & TODO - Type 'back' to return to menu")
        print("="*70)
        print("\nCommands:")
        print("  note [text]        - Take a note")
        print("  notes              - View all notes")
        print("  todo add [task]    - Add todo")
        print("  todo list          - View todos")
        print("  todo complete [#]  - Complete todo")
        print("  todo remove [#]    - Remove todo")
        print("  todo clear         - Clear all todos")
        print("-"*70)
        
        while True:
            try:
                user_input = input("\nNotes> ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ['back', 'menu', 'exit']:
                    break
                
                # Process notes/todo commands
                self.commands.process(user_input)
                
            except KeyboardInterrupt:
                print("\n⏹️ Returning to menu...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def system_mode(self):
        """System tools"""
        print("\n" + "="*70)
        print("🛠️ SYSTEM TOOLS - Type 'back' to return to menu")
        print("="*70)
        print("\nSystem commands:")
        print("  system     - System information")
        print("  processes  - Running processes")
        print("  disk       - Disk usage")
        print("  network    - Network information")
        print("  battery    - Battery status")
        print("  shutdown   - System shutdown (admin)")
        print("  restart    - System restart (admin)")
        print("-"*70)
        
        while True:
            try:
                user_input = input("\nSystem> ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ['back', 'menu', 'exit']:
                    break
                
                # Process system commands
                self.commands.process(user_input)
                
            except KeyboardInterrupt:
                print("\n⏹️ Returning to menu...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def web_mode(self):
        """Web and search tools"""
        print("\n" + "="*70)
        print("🌐 WEB & SEARCH - Type 'back' to return to menu")
        print("="*70)
        print("\nWeb commands:")
        print("  search [query]     - Web search")
        print("  youtube [query]    - Search YouTube")
        print("  wikipedia [topic]  - Wikipedia search")
        print("  news [topic]       - Get news")
        print("  weather [city]     - Weather forecast")
        print("  translate [text]   - Translate text")
        print("-"*70)
        
        while True:
            try:
                user_input = input("\nWeb> ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ['back', 'menu', 'exit']:
                    break
                
                # Process web commands
                self.commands.process(user_input)
                
            except KeyboardInterrupt:
                print("\n⏹️ Returning to menu...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def settings_mode(self):
        """Settings and configuration"""
        print("\n" + "="*70)
        print("⚙️ SETTINGS - Type 'back' to return to menu")
        print("="*70)
        print("\nSettings:")
        print(f"  1. AI Model: {self.config['openai']['model']}")
        print(f"  2. Voice: {'Enabled' if self.config['speech']['enabled'] else 'Disabled'}")
        print(f"  3. Auto-save notes: {self.config['assistant']['auto_save_notes']}")
        print(f"  4. Show current config")
        print(f"  5. Reload config")
        print(f"  6. Clear conversation history")
        print("-"*70)
        
        choice = input("\nSelect option (1-6 or 'back'): ").strip()
        
        if choice == "1":
            model = input(f"Enter new model [current: {self.config['openai']['model']}]: ").strip()
            if model:
                self.config['openai']['model'] = model
                print(f"✅ Model updated to: {model}")
        elif choice == "2":
            enabled = not self.config['speech']['enabled']
            self.config['speech']['enabled'] = enabled
            print(f"✅ Voice {'enabled' if enabled else 'disabled'}")
        elif choice == "3":
            auto_save = not self.config['assistant']['auto_save_notes']
            self.config['assistant']['auto_save_notes'] = auto_save
            print(f"✅ Auto-save notes {'enabled' if auto_save else 'disabled'}")
        elif choice == "4":
            self.commands.process("config")
        elif choice == "5":
            self.config = ConfigManager().reload_config()
            print("✅ Configuration reloaded")
        elif choice == "6":
            self.ai.clear_history()
            print("✅ Conversation history cleared")
