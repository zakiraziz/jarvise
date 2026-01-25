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
                        # Save note to file
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with open(self.notes_file, 'a', encoding='utf-8') as f:
                f.write(f"[{timestamp}] {note_text}\n")
            
            print(f"[?] Note saved: {note_text}")
            logger.info(f"Note saved: {note_text[:50]}...")
            
        except Exception as e:
            print(f"[X] Error saving note: {e}")
    
    def _list_notes(self, command: str):
        """List all notes"""
        try:
            if not os.path.exists(self.notes_file):
                print("[?] No notes found")
                return
            
            with open(self.notes_file, 'r', encoding='utf-8') as f:
                notes = f.read()
            
            if not notes.strip():
                print("[?] No notes found")
                return
            
            print("[?] YOUR NOTES:")
            print("=" * 50)
            print(notes)
            print("=" * 50)
            
        except Exception as e:
            print(f"[X] Error reading notes: {e}")
    
    # ========== TODO LIST ==========
    
    def _manage_todo(self, command: str):
        """Manage todo list"""
        try:
            parts = command.split(' ', 1)
            action = parts[1].lower() if len(parts) > 1 else "list"
            
            # Load todos
            with open(self.todo_file, 'r') as f:
                data = json.load(f)
            
            todos = data.get("todos", [])
            completed = data.get("completed", [])
            
            if action == "list":
                print("[?] TODO LIST:")
                print("-" * 40)
                
                if not todos and not completed:
                    print("No todos yet! Add one with: todo add [task]")
                    return
                
                if todos:
                    print("Pending:")
                    for i, todo in enumerate(todos, 1):
                        print(f"  {i}. {todo}")
                
                if completed:
                    print("\nCompleted:")
                    for i, todo in enumerate(completed, 1):
                        print(f"  {i}. {todo} ()")
                
                print("-" * 40)
                
            elif action.startswith("add"):
                task = command[9:].strip()  # Remove 'todo add '
                if task:
                    todos.append(task)
                    print(f" Added: {task}")
                else:
                    print("Usage: todo add [task description]")
                    
            elif action.startswith("remove"):
                try:
                    task_num = int(command.split()[2]) - 1
                    if 0 <= task_num < len(todos):
                        removed = todos.pop(task_num)
                        print(f" Removed: {removed}")
                    else:
                        print("[X] Invalid task number")
                except (IndexError, ValueError):
                    print("Usage: todo remove [task_number]")
            
            elif action.startswith("complete"):
                try:
                    task_num = int(command.split()[2]) - 1
                    if 0 <= task_num < len(todos):
                        completed_task = todos.pop(task_num)
                        completed.append(completed_task)
                        print(f" Completed: {completed_task}")
                    else:
                        print("[X] Invalid task number")
                except (IndexError, ValueError):
                    print("Usage: todo complete [task_number]")
            
            elif action == "clear":
                data["todos"] = []
                data["completed"] = []
                print(" Todo list cleared")
            
            else:
                print("Usage: todo [list|add|remove|complete|clear]")
                return
            
            # Save updated todos
            data["todos"] = todos
            data["completed"] = completed
            
            with open(self.todo_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"[X] Todo error: {e}")
    
    # ========== SYSTEM INFO ==========
    
    def _system_info(self, command: str):
        """Show system information"""
        try:
            import platform
            import psutil
            
            print("[*] SYSTEM INFORMATION")
            print("=" * 50)
            
            # OS Info
            print(f"OS: {platform.system()} {platform.release()}")
            print(f"Version: {platform.version()}")
            print(f"Architecture: {platform.machine()}")
            print(f"Processor: {platform.processor()}")
            
            # CPU Info
            cpu_count = psutil.cpu_count(logical=True)
            cpu_percent = psutil.cpu_percent(interval=1)
            print(f"CPU Cores: {cpu_count} (Logical)")
            print(f"CPU Usage: {cpu_percent}%")
            
            # Memory Info
            memory = psutil.virtual_memory()
            print(f"Memory Usage: {memory.percent}%")
            print(f"Total RAM: {memory.total / (1024**3):.2f} GB")
            print(f"Available RAM: {memory.available / (1024**3):.2f} GB")
            print(f"Used RAM: {memory.used / (1024**3):.2f} GB")
            
            # Disk Info
            disk = psutil.disk_usage('/')
            print(f"Disk Usage: {disk.percent}%")
            print(f"Total Disk: {disk.total / (1024**3):.2f} GB")
            print(f"Free Disk: {disk.free / (1024**3):.2f} GB")
            print(f"Used Disk: {disk.used / (1024**3):.2f} GB")
            
            # Network Info
            net_io = psutil.net_io_counters()
            print(f"Bytes Sent: {net_io.bytes_sent / (1024**2):.2f} MB")
            print(f"Bytes Received: {net_io.bytes_recv / (1024**2):.2f} MB")
            
            # Boot Time
            boot_time = datetime.fromtimestamp(psutil.boot_time())
            print(f"System Boot: {boot_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Python Info
            print(f"Python: {sys.version.split()[0]}")
            print(f"Python Path: {sys.executable}")
            
            print("=" * 50)
            
        except ImportError:
            print("[X] System info requires psutil library")
            print("Install with: pip install psutil")
        except Exception as e:
            print(f"[X] Error getting system info: {e}")
    
    # ========== WEB & APPLICATIONS ==========
    
    def _open_application(self, command: str):
        """Open application or website"""
        try:
            target = command[5:].strip().lower()  # Remove 'open '
            
            if not target:
                print("Usage: open [application or website]")
                print("\nExamples:")
                print("  open chrome")
                print("  open notepad")
                print("  open youtube")
                print("  open google.com")
                return
            
            # Common applications mapping
            apps = {
                'chrome': 'chrome',
                'firefox': 'firefox',
                'edge': 'msedge',
                'notepad': 'notepad',
                'calculator': 'calc',
                'paint': 'mspaint',
                'cmd': 'cmd',
                'powershell': 'powershell',
                'explorer': 'explorer',
                'word': 'winword',
                'excel': 'excel',
                'powerpoint': 'powerpnt',
                'vscode': 'code',
                'pycharm': 'pycharm',
                'spotify': 'spotify',
                'discord': 'discord',
            }
            
            # Common websites
            websites = {
                'youtube': 'https://youtube.com',
                'google': 'https://google.com',
                'github': 'https://github.com',
                'stackoverflow': 'https://stackoverflow.com',
                'wikipedia': 'https://wikipedia.org',
                'reddit': 'https://reddit.com',
                'twitter': 'https://twitter.com',
                'facebook': 'https://facebook.com',
                'instagram': 'https://instagram.com',
                'linkedin': 'https://linkedin.com',
                'gmail': 'https://gmail.com',
                'outlook': 'https://outlook.com',
                'netflix': 'https://netflix.com',
                'amazon': 'https://amazon.com',
                'ebay': 'https://ebay.com',
            }
            
            if target in apps:
                # Open application
                try:
                    if platform.system() == "Windows":
                        os.system(f'start {apps[target]}')
                    elif platform.system() == "Darwin":  # macOS
                        os.system(f'open -a {apps[target]}')
                    else:  # Linux
                        os.system(f'{apps[target]} &')
                    
                    print(f" Opened {target}")
                except Exception as e:
                    print(f"[X] Error opening {target}: {e}")
            
            elif target in websites:
                # Open website
                webbrowser.open(websites[target])
                print(f" Opened {target}")
            
            elif '.' in target:
                # Try to open as website
                if not target.startswith('http'):
                    target = f'https://{target}'
                webbrowser.open(target)
                print(f" Opened {target}")
            
            else:
                print(f"[X] Don't know how to open: {target}")
                print("\nAvailable applications:")
                print(", ".join(sorted(apps.keys())))
                print("\nAvailable websites:")
                print(", ".join(sorted(websites.keys())))
        
        except Exception as e:
            print(f"[X] Error opening application: {e}")
    
    def _web_search(self, command: str):
        """Search the web"""
        try:
            # Extract query
            parts = command.split(' ', 1)
            if len(parts) < 2:
                print("Usage: search [query]")
                print("Example: search python tutorials")
                return
            
            query = parts[1].strip()
            
            # Encode query for URL
            import urllib.parse
            encoded_query = urllib.parse.quote(query)
            
            # Choose search engine
            search_engine = self.config.get('web', {}).get('search_engine', 'google')
            
            search_urls = {
                'google': f'https://www.google.com/search?q={encoded_query}',
                'bing': f'https://www.bing.com/search?q={encoded_query}',
                'duckduckgo': f'https://duckduckgo.com/?q={encoded_query}',
                'yahoo': f'https://search.yahoo.com/search?p={encoded_query}',
            }
            
            url = search_urls.get(search_engine.lower(), search_urls['google'])
            webbrowser.open(url)
            
            print(f"[?] Searching {search_engine} for: {query}")
            
        except Exception as e:
            print(f"[X] Search error: {e}")
    
    # ========== WEATHER & NEWS ==========
    
    def _weather_info(self, command: str):
        """Get weather information"""
        try:
            # Extract location
            parts = command.split(' ', 1)
            location = parts[1].strip() if len(parts) > 1 else ""
            
            if not location:
                print("Usage: weather [city]")
                print("Example: weather New York")
                print("Example: weather London, UK")
                return
            
            # Use OpenWeatherMap API (would need API key)
            print(f"[INFO] Getting weather for {location}...")
            print("(Weather API not configured. Add API key to config.yaml)")
            
            # For now, use AI to simulate weather
            weather_prompt = f"What's the weather like in {location}? Give a realistic forecast."
            response = self.ai.chat(weather_prompt)
            print(f"\n{response}")
            
        except Exception as e:
            print(f"[X] Weather error: {e}")
    
    def _news_info(self, command: str):
        """Get news information"""
        try:
            print("[?] Getting latest news...")
            print("(News API not configured. Add API key to config.yaml)")
            
            # Use AI to generate news summary
            news_prompt = "What are the top 3 news headlines today? Be brief."
            response = self.ai.chat(news_prompt)
            print(f"\n{response}")
            
        except Exception as e:
            print(f"[X] News error: {e}")
    


