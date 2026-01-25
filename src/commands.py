"""
Command Handler Module
Processes and executes user commands
"""

import logging
import subprocess
import os
import webbrowser
import platform
import sys
from typing import Dict, List
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class CommandHandler:
    def __init__(self, config: dict, ai_assistant):
        """
        Initialize command handler
        
        Args:
            config: Configuration dictionary
            ai_assistant: AIAssistant instance
        """
        self.config = config
        self.ai = ai_assistant
        self.safety_check = config['assistant'].get('safety_check_enabled', True)
        
        # Data storage
        self.notes_file = "notes.txt"
        self.todo_file = "todo.json"
        self._init_data_files()
        
        # Built-in commands dictionary
        self.commands = {
            'help': self.show_help,
            'clear': self._clear_history,
            'time': self._show_time,
            'date': self._show_date,
            'weather': self._weather_info,
            'news': self._news_info,
            'calculate': self._calculate,
            'calc': self._calculate,
            'open': self._open_application,
            'search': self._web_search,
            'google': self._web_search,
            'note': self._take_note,
            'notes': self._list_notes,
            'todo': self._manage_todo,
            'system': self._system_info,
            'config': self._show_config,
            'history': self._show_history,
            'joke': self._tell_joke,
            'quote': self._get_quote,
            'wiki': self._wikipedia_search,
            'translate': self._translate_text,
            'remind': self._set_reminder,
            'alarm': self._set_alarm,
            'timer': self._set_timer,
            'email': self._send_email,
            'file': self._file_operations,
            'run': self._run_command,
            'python': self._run_python,
            'code': self._analyze_code,
            'learn': self._learn_concept,
            'ping': self._ping_host,
            'ip': self._show_ip,
            'shutdown': self._system_shutdown,
            'restart': self._system_restart,
            'screenshot': self._take_screenshot,
            'record': self._record_screen,
            'music': self._play_music,
            'video': self._play_video,
            'game': self._play_game,
            'book': self._read_book,
            'recipe': self._get_recipe,
            'exercise': self._get_exercise,
            'meditate': self._start_meditation,
            'budget': self._manage_budget,
            'shopping': self._shopping_list,
            'travel': self._travel_info,
            'crypto': self._crypto_info,
            'stocks': self._stock_info,
            'news': self._get_news,
            'movies': self._movie_info,
            'tv': self._tv_show_info,
            'sports': self._sports_info,
            'weather': self._weather_forecast,
            'map': self._show_map,
            'directions': self._get_directions,
            'flight': self._flight_info,
            'hotel': self._hotel_info,
            'restaurant': self._restaurant_info,
            'event': self._event_info,
            'meeting': self._schedule_meeting,
            'call': self._make_call,
            'message': self._send_message,
            'social': self._social_media,
            'backup': self._backup_files,
            'update': self._update_system,
            'install': self._install_package,
            'uninstall': self._uninstall_package,
            'virus': self._scan_virus,
            'clean': self._clean_system,
            'optimize': self._optimize_system,
            'network': self._network_info,
            'wifi': self._wifi_info,
            'bluetooth': self._bluetooth_info,
            'printer': self._printer_info,
            'camera': self._camera_info,
            'microphone': self._microphone_info,
            'speaker': self._speaker_info,
            'display': self._display_info,
            'battery': self._battery_info,
            'disk': self._disk_info,
            'memory': self._memory_info,
            'cpu': self._cpu_info,
            'gpu': self._gpu_info,
            'os': self._os_info,
            'version': self._version_info,
            'license': self._license_info,
            'about': self._about_info,
            'contact': self._contact_info,
            'donate': self._donate_info,
            'feedback': self._send_feedback,
            'bug': self._report_bug,
            'feature': self._request_feature,
            'docs': self._show_docs,
            'tutorial': self._show_tutorial,
            'examples': self._show_examples,
            'api': self._show_api,
            'github': self._open_github,
            'website': self._open_website,
            'forum': self._open_forum,
            'blog': self._open_blog,
            'youtube': self._open_youtube,
            'twitter': self._open_twitter,
            'facebook': self._open_facebook,
            'instagram': self._open_instagram,
            'linkedin': self._open_linkedin,
            'reddit': self._open_reddit,
            'discord': self._open_discord,
            'slack': self._open_slack,
            'whatsapp': self._open_whatsapp,
            'telegram': self._open_telegram,
            'signal': self._open_signal,
        }
        
        logger.info(f"[OK] Command handler initialized with {len(self.commands)} commands")
    
    def _init_data_files(self):
        """Initialize data files if they don't exist"""
        # Notes file
        if not os.path.exists(self.notes_file):
            with open(self.notes_file, 'w') as f:
                f.write("# Jarvis Notes\n")
                f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*50 + "\n\n")
        
        # Todo file
        if not os.path.exists(self.todo_file):
            with open(self.todo_file, 'w') as f:
                json.dump({"todos": [], "completed": []}, f, indent=2)
    
    def process_command(self, command: str):
        """
        Process user command
        
        Args:
            command: User command string
        """
        command = command.strip()
        if not command:
            return
        
        logger.info(f"Processing command: {command[:50]}...")
        
        # Safety check
        if self.safety_check and self._is_blocked(command):
            print("[?] This command contains blocked keywords for safety.")
            logger.warning(f"Blocked command attempted: {command}")
            return
        
        # Extract command keyword (first word)
        parts = command.split()
        cmd_keyword = parts[0].lower()
        
        # Check if it's a built-in command
        if cmd_keyword in self.commands:
            try:
                print(f"[*] Executing: {cmd_keyword}...")
                self.commands[cmd_keyword](command)
                return
            except Exception as e:
                logger.error(f"Error executing command {cmd_keyword}: {e}")
                print(f"[X] Error executing command: {e}")
                return
        
        # Otherwise, use AI
        print("[?] Thinking...", end='\r')
        response = self.ai.chat(command)
        
        # Format and display response
        print("\n" + "="*60)
        print("[*] JARVIS:")
        print("="*60)
        print(response)
        print("="*60)
    
    def show_help(self, command: str = None):
        """Show help information"""
        help_text = """
JARVIS AI ASSISTANT - COMMAND REFERENCE

BASIC COMMANDS:
  * help - Show this help message
  * clear - Clear conversation history
  * time - Show current time
  * date - Show current date
  * history - Show conversation history

SYSTEM COMMANDS:
  * system - Show system information
  * ip - Show IP address
  * ping [host] - Ping a network host

TOOLS & UTILITIES:
  * calculate [expression] - Calculate math expression
  * note [text] - Take a quick note
  * notes - List all notes
  * todo [add/list/remove] - Manage todo list

Type 'quit' or 'exit' to exit.
"""
        print(help_text)
    
    def _is_blocked(self, command: str) -> bool:
        """Check if command contains blocked keywords"""
        blocked_keywords = self.config['safety'].get('blocked_keywords', [])
        command_lower = command.lower()
        
        for keyword in blocked_keywords:
            if keyword.lower() in command_lower:
                logger.warning(f"Blocked keyword detected: {keyword}")
                return True
        return False
    
    # ========== BASIC COMMANDS ==========
    
    def _clear_history(self, command: str):
        """Clear conversation history"""
        self.ai.clear_history()
        print(" Conversation history cleared")
        logger.info("Conversation history cleared by user")
    
    def _show_time(self, command: str):
        """Show current time"""
        current_time = datetime.now().strftime("%I:%M:%S %p")
        print(f"[TIME] Current time: {current_time}")
    
    def _show_date(self, command: str):
        """Show current date"""
        current_date = datetime.now().strftime("%A, %B %d, %Y")
        print(f"[DATE] Current date: {current_date}")
        print(f"[*] Current date: {current_date}")
    
    def _show_history(self, command: str):
        """Show conversation history"""
        history = self.ai.get_history_summary()
        print("[?] Conversation History:")
        print("-" * 40)
        print(history)
        print("-" * 40)
    
    def _show_config(self, command: str):
        """Show current configuration"""
        print("[?] Current Configuration:")
        print("-" * 40)
        
        # Show important config sections
        sections = ['openai', 'speech', 'assistant', 'logging']
        for section in sections:
            if section in self.config:
                print(f"\n{section.upper()}:")
                for key, value in self.config[section].items():
                    if 'key' in key.lower() and value:
                        print(f"  {key}: {'*' * 10}{value[-4:]}")
                    else:
                        print(f"  {key}: {value}")
        
        print("-" * 40)
    
    # ========== CALCULATOR ==========
    
    def _calculate(self, command: str):
        """Calculate mathematical expression"""
        try:
            # Extract expression (remove command keyword)
            expr = command.split(' ', 1)[1] if ' ' in command else ''
            
            if not expr:
                print("Usage: calculate [expression]")
                print("Examples:")
                print("  calculate 2 + 2")
                print("  calculate sin(45)")
                print("  calculate 2 * (3 + 4) / 5")
                return
            
            # Safety check - only allow safe operations
            allowed_chars = set('0123456789+-*/().^% ')
            allowed_functions = ['sin', 'cos', 'tan', 'sqrt', 'log', 'exp']
            
            # Check for function calls
            expr_lower = expr.lower()
            for func in allowed_functions:
                if func in expr_lower:
                    # Replace function names with math module calls
                    expr = expr.replace(func, f'math.{func}')
            
            # Check for unsafe characters
            if not all(c in allowed_chars or expr[i:i+5] in ['math.', 'math.'] for i, c in enumerate(expr)):
                print("[X] Only basic math operations and functions allowed")
                print("Allowed: numbers, + - * / ( ) . ^ % sin cos tan sqrt log exp")
                return
            
            # Import math if needed
            if 'math.' in expr:
                import math
            
            # Calculate
            result = eval(expr, {"__builtins__": {}}, {"math": math} if 'math' in locals() else {})
            print(f"[?] {expr} = {result}")
            
        except NameError:
            print("[X] Unknown function or variable")
        except ZeroDivisionError:
            print("[X] Division by zero")
        except SyntaxError:
            print("[X] Invalid expression syntax")
        except Exception as e:
            print(f"[X] Calculation error: {e}")
    
    # ========== NOTES ==========
    
    def _take_note(self, command: str):
        """Take a quick note"""
        try:
            note_text = command[5:].strip()  # Remove 'note '
            
            if not note_text:
                print("Usage: note [your note text]")
                print("Example: note Buy groceries tomorrow")
                return
            
