"""
Command Handler Module
"""

import os
import sys
import logging
import subprocess
import webbrowser
import platform
import json
import shutil
from datetime import datetime
from typing import Dict

logger = logging.getLogger(__name__)


class CommandHandler:
    """Handles all user commands"""
    
    def __init__(self, config: dict, ai_assistant):
        self.config = config
        self.ai = ai_assistant
        
        # Data files
        self.notes_file = "notes.txt"
        self.todo_file = "todos.json"
        self.init_data_files()
        
        # Command registry
        self.commands = self._register_commands()
        logger.info(f"Command handler initialized with {len(self.commands)} commands")
    
    def _register_commands(self) -> Dict[str, callable]:
        """Register all available commands"""
        return {
            # Basic commands
            'help': self.cmd_help,
            'menu': self.cmd_menu,
            'clear': self.cmd_clear,
            'exit': self.cmd_exit,
            'quit': self.cmd_exit,
            
            # Time & Date
            'time': self.cmd_time,
            'date': self.cmd_date,
            
            # System
            'system': self.cmd_system,
            'ip': self.cmd_ip,
            'ping': self.cmd_ping,
            
            # Files & Notes
            'note': self.cmd_note,
            'notes': self.cmd_notes,
            'todo': self.cmd_todo,
            
            # Web
            'search': self.cmd_search,
            'google': self.cmd_search,
            'youtube': self.cmd_youtube,
            'weather': self.cmd_weather,
            
            # Tools
            'calculate': self.cmd_calculate,
            'calc': self.cmd_calculate,
            
            # Fun
            'joke': self.cmd_joke,
            'quote': self.cmd_quote,
            
            # AI
            'ask': self.cmd_ask,
            
            # Config
            'config': self.cmd_config,
            'about': self.cmd_about,
        }
    
    def init_data_files(self):
        """Initialize data files"""
        # Notes file
        if not os.path.exists(self.notes_file):
            with open(self.notes_file, 'w', encoding='utf-8') as f:
                f.write("# Jarvis Notes\n")
                f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*50 + "\n\n")
        
        # Todo file
        if not os.path.exists(self.todo_file):
            with open(self.todo_file, 'w', encoding='utf-8') as f:
                json.dump({"todos": []}, f, indent=2)
    
    def process(self, command_str: str):
        """Process a command string"""
        if not command_str.strip():
            return
        
        # Parse command
        parts = command_str.split()
        cmd = parts[0].lower()
        
        logger.info(f"Processing command: {cmd}")
        
        # Check if it's a registered command
        if cmd in self.commands:
            try:
                self.commands[cmd](command_str)
            except Exception as e:
                logger.error(f"Error executing command {cmd}: {e}")
                print(f"❌ Error: {e}")
        else:
            # If not a built-in command, use AI
            self.cmd_ai_chat(command_str)
    
    def get_commands(self) -> Dict[str, callable]:
        """Get all registered commands"""
        return self.commands
    
    def save_data(self):
        """Save all data files"""
        logger.info("Saving data files...")
    
    # ===== COMMAND IMPLEMENTATIONS =====
    
    def cmd_help(self, command: str = ""):
        """Show help information"""
        help_text = """
🤖 JARVIS AI ASSISTANT - COMMAND REFERENCE

📋 BASIC COMMANDS:
  help / menu          - Show this help / main menu
  clear                - Clear screen
  exit / quit          - Exit Jarvis

⏰ TIME & DATE:
  time                 - Current time
  date                 - Current date

💻 SYSTEM COMMANDS:
  system               - System information
  ip                   - Show IP addresses
  ping [host]          - Ping a host

📝 NOTES & FILES:
  note [text]          - Take a note
  notes                - List all notes
  todo [cmd] [args]    - Manage todo list

🌐 WEB COMMANDS:
  search [query]       - Web search
  youtube [query]      - Search YouTube
  weather [city]       - Weather forecast

🔧 TOOLS:
  calculate [expr]     - Calculator

🎮 FUN COMMANDS:
  joke                 - Tell a joke
  quote                - Inspirational quote

🤖 AI COMMANDS:
  ask [question]       - Ask AI anything

⚙️ CONFIGURATION:
  config               - Show configuration
  about                - About Jarvis

💡 TIP: Type anything else to chat with AI!
"""
        print(help_text)
    
    def cmd_menu(self, command: str = ""):
        """Show menu"""
        print("Use the number menu or type commands directly.")
    
    def cmd_clear(self, command: str = ""):
        """Clear screen"""
        os.system('cls' if platform.system() == 'Windows' else 'clear')
    
    def cmd_exit(self, command: str = ""):
        """Exit command"""
        print("👋 Goodbye!")
        sys.exit(0)
    
    def cmd_time(self, command: str = ""):
        """Show current time"""
        current_time = datetime.now().strftime("%I:%M:%S %p")
        print(f"🕐 Current time: {current_time}")
    
    def cmd_date(self, command: str = ""):
        """Show current date"""
        current_date = datetime.now().strftime("%A, %B %d, %Y")
        print(f"📅 Current date: {current_date}")
    
    def cmd_system(self, command: str = ""):
        """Show system information"""
        print("💻 SYSTEM INFORMATION")
        print("="*60)
        print(f"OS: {platform.system()} {platform.release()}")
        print(f"Version: {platform.version()}")
        print(f"Architecture: {platform.machine()}")
        print(f"Python: {sys.version.split()[0]}")
        print("="*60)
    
    def cmd_ip(self, command: str = ""):
        """Show IP addresses"""
        try:
            import socket
            
            print("🌐 NETWORK INFORMATION")
            print("="*40)
            
            # Local IP
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            print(f"Hostname: {hostname}")
            print(f"Local IP: {local_ip}")
            print("="*40)
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def cmd_ping(self, command: str):
        """Ping a host"""
        try:
            parts = command.split()
            host = parts[1] if len(parts) > 1 else "google.com"
            
            print(f"📡 Pinging {host}...")
            
            # Platform-specific ping command
            param = '-n' if platform.system().lower() == 'windows' else '-c'
            count = '4'
            
            result = subprocess.run(
                ['ping', param, count, host],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print("✅ Host is reachable")
            else:
                print("❌ Host is not reachable")
            
            print(f"\n{result.stdout}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def cmd_note(self, command: str):
        """Take a note"""
        try:
            note_text = command[5:].strip()
            if not note_text:
                print("Usage: note [your note text]")
                return
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with open(self.notes_file, 'a', encoding='utf-8') as f:
                f.write(f"[{timestamp}] {note_text}\n")
            
            print(f"📝 Note saved: {note_text}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def cmd_notes(self, command: str = ""):
        """List all notes"""
        try:
            if not os.path.exists(self.notes_file):
                print("📝 No notes found")
                return
            
            with open(self.notes_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            print("📝 YOUR NOTES")
            print("="*60)
            print(content)
            print("="*60)
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def cmd_todo(self, command: str):
        """Manage todo list"""
        try:
            parts = command.split(' ', 2)
            action = parts[1].lower() if len(parts) > 1 else "list"
            
            # Load todos
            with open(self.todo_file, 'r') as f:
                data = json.load(f)
            
            todos = data.get("todos", [])
            
            if action == "list":
                if not todos:
                    print("✅ Todo list is empty!")
                else:
                    print("📋 TODO LIST:")
                    for i, task in enumerate(todos, 1):
                        print(f"  {i}. {task}")
            
            elif action == "add":
                if len(parts) > 2:
                    task = parts[2]
                    todos.append(task)
                    print(f"✅ Added: {task}")
                else:
                    print("Usage: todo add [task]")
            
            elif action == "remove":
                if len(parts) > 2:
                    try:
                        idx = int(parts[2]) - 1
                        if 0 <= idx < len(todos):
                            removed = todos.pop(idx)
                            print(f"🗑️ Removed: {removed}")
                        else:
                            print("❌ Invalid task number")
                    except:
                        print("Usage: todo remove [number]")
                else:
                    print("Usage: todo remove [number]")
            
            elif action == "clear":
                confirm = input("Clear all todos? (y/n): ").lower()
                if confirm == 'y':
                    data["todos"] = []
                    print("✅ Todo list cleared")
                else:
                    print("❌ Cancelled")
            
            else:
                print("Usage: todo [list|add|remove|clear]")
                return
            
            # Save updated todos
            data["todos"] = todos
            
            with open(self.todo_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def cmd_search(self, command: str):
        """Web search"""
        try:
            parts = command.split(' ', 1)
            if len(parts) < 2:
                print("Usage: search [query]")
                return
            
            query = parts[1].strip()
            url = f'https://www.google.com/search?q={query.replace(" ", "+")}'
            webbrowser.open(url)
            
            print(f"🔍 Searching for: {query}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def cmd_calculate(self, command: str):
        """Calculate mathematical expression"""
        try:
            expr = command.split(' ', 1)[1] if ' ' in command else ''
            
            if not expr:
                print("Usage: calculate [expression]")
                return
            
            # Safety check
            allowed_chars = set('0123456789+-*/().^% ')
            
            # Check for unsafe characters
            expr_clean = expr.replace(' ', '')
            if not all(c in allowed_chars for c in expr_clean):
                print("❌ Only basic math operations allowed")
                return
            
            # Calculate
            result = eval(expr, {"__builtins__": {}}, {})
            print(f"🧮 {expr} = {result}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def cmd_joke(self, command: str = ""):
        """Tell a joke"""
        import random
        jokes = [
            "Why do programmers prefer dark mode? Because light attracts bugs!",
            "Why do Python developers need glasses? Because they can't C!",
            "What's a programmer's favorite hangout place? Foo Bar!",
            "Why did the programmer quit his job? He didn't get arrays!",
        ]
        print(f"😂 {random.choice(jokes)}")
    
    def cmd_quote(self, command: str = ""):
        """Get inspirational quote"""
        import random
        quotes = [
            "The only way to do great work is to love what you do. - Steve Jobs",
            "The future belongs to those who believe in the beauty of their dreams. - Eleanor Roosevelt",
            "Don't watch the clock; do what it does. Keep going. - Sam Levenson",
            "Believe you can and you're halfway there. - Theodore Roosevelt",
        ]
        print(f"💭 {random.choice(quotes)}")
    
    def cmd_weather(self, command: str):
        """Get weather information"""
        try:
            city = command[8:].strip()
            if not city:
                print("Usage: weather [city]")
                return
            
            if self.ai and self.ai.is_available():
                print(f"🌤️ Getting weather for {city}...")
                response = self.ai.chat(f"What's the current weather in {city}?")
                print(f"\n{response}")
            else:
                print("⚠️ Opening weather website...")
                webbrowser.open(f"https://www.google.com/search?q=weather+{city}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def cmd_youtube(self, command: str):
        """Search YouTube"""
        try:
            query = command[8:].strip()
            if not query:
                webbrowser.open("https://youtube.com")
                print("📺 Opened YouTube")
                return
            
            webbrowser.open(f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}")
            print(f"📺 Searching YouTube for: {query}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def cmd_ask(self, command: str):
        """Ask AI anything"""
        try:
            question = command[4:].strip()
            if not question:
                print("Usage: ask [your question]")
                return
            
            print("🤔 Thinking...", end='\r')
            response = self.ai.chat(question)
            print("\n" + "="*60)
            print("💡 Answer:")
            print("="*60)
            print(response)
            print("="*60)
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def cmd_config(self, command: str = ""):
        """Show configuration"""
        print("⚙️ CURRENT CONFIGURATION")
        print("="*60)
        print(f"Assistant: {self.config['assistant']['name']} v{self.config['assistant']['version']}")
        print(f"AI Model: {self.config['openai']['model']}")
        print(f"Voice: {'Enabled' if self.config['speech']['enabled'] else 'Disabled'}")
        print("="*60)
    
    def cmd_about(self, command: str = ""):
        """About Jarvis"""
        about = f"""
🤖 {self.config['assistant']['name']} AI Assistant v{self.config['assistant']['version']}

A complete AI assistant built with Python, featuring:
• OpenAI GPT integration
• Command system
• Text-to-speech
• File and note management

Type 'help' to see all available commands.
"""
        print(about)
    
    def cmd_ai_chat(self, command: str):
        """Handle non-command input as AI chat"""
        if self.ai and self.ai.is_available():
            print("🤔 Thinking...", end='\r')
            response = self.ai.chat(command)
            print("\n" + "="*70)
            print(f"💡 {self.config['assistant']['name']}:")
            print("="*70)
            print(response)
            print("="*70)
        else:
            print("⚠️ AI chat not available. Please check your OpenAI API key.")
            print("\nAvailable commands:")
            print("  • time / date / system / ip")
            print("  • note [text] / notes / todo [cmd]")
            print("  • search [query] / calculate [math]")
            print("  • Type 'help' for all commands")
