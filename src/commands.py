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
        # ========== ENTERTAINMENT ==========
    
    def _tell_joke(self, command: str):
        """Tell a joke"""
        try:
            print("[?] Here's a joke:")
            jokes = [
                "Why do programmers prefer dark mode? Because light attracts bugs!",
                "Why do Python developers need glasses? Because they can't C!",
                "What's a programmer's favorite hangout place? Foo Bar!",
                "Why did the programmer quit his job? He didn't get arrays!",
                "How many programmers does it take to change a light bulb? None, that's a hardware problem!",
                "Why do Java developers wear glasses? Because they don't C#!",
                "What's the object-oriented way to become wealthy? Inheritance!",
                "Why did the developer go broke? He used up all his cache!",
                "What do you call a programmer from Finland? Nerdic!",
                "Why was the JavaScript developer sad? He didn't Node how to Express himself!",
            ]
            
            import random
            print(f"\n{random.choice(jokes)}\n")
            
        except Exception as e:
            print(f"[X] Joke error: {e}")
    
    def _get_quote(self, command: str):
        """Get inspirational quote"""
        try:
            print("[*] Inspirational quote:")
            quotes = [
                "The only way to do great work is to love what you do. - Steve Jobs",
                "Innovation distinguishes between a leader and a follower. - Steve Jobs",
                "The future belongs to those who believe in the beauty of their dreams. - Eleanor Roosevelt",
                "The way to get started is to quit talking and begin doing. - Walt Disney",
                "Don't watch the clock; do what it does. Keep going. - Sam Levenson",
                "The only limit to our realization of tomorrow will be our doubts of today. - Franklin D. Roosevelt",
                "It does not matter how slowly you go as long as you do not stop. - Confucius",
                "Everything you've ever wanted is on the other side of fear. - George Addair",
                "Success is not final, failure is not fatal: it is the courage to continue that counts. - Winston Churchill",
                "Believe you can and you're halfway there. - Theodore Roosevelt",
            ]
            
            import random
            print(f"\n{random.choice(quotes)}\n")
            
        except Exception as e:
            print(f"[X] Quote error: {e}")
    
    # ========== CODE ANALYSIS ==========
    
    def _analyze_code(self, command: str):
        """Analyze code"""
        try:
            code = command[5:].strip()  # Remove 'code '
            
            if not code:
                print("Usage: code [your code here]")
                print("Example: code print('Hello, World!')")
                return
            
            print("[?] Analyzing code...")
            response = self.ai.analyze_code(code)
            print(f"\n{response}")
            
        except Exception as e:
            print(f"[X] Code analysis error: {e}")
    
    def _learn_concept(self, command: str):
        """Learn a programming concept"""
        try:
            concept = command[6:].strip()  # Remove 'learn '
            
            if not concept:
                print("Usage: learn [concept]")
                print("Example: learn recursion")
                print("Example: learn object oriented programming")
                return
            
            print(f"[?] Learning about {concept}...")
            response = self.ai.explain_concept(concept)
            print(f"\n{response}")
            
        except Exception as e:
            print(f"[X] Learning error: {e}")
    
    def _manage_budget(self, command: str):
        print("[*] Budget management coming soon!")
    
    def _shopping_list(self, command: str):
        print("[?] Shopping list coming soon!")
    
    def _travel_info(self, command: str):
        print("[?] Travel info coming soon!")
    
    def _crypto_info(self, command: str):
        print("[?] Cryptocurrency info coming soon!")
    
    def _stock_info(self, command: str):
        print("[*] Stock info coming soon!")
    
    def _get_news(self, command: str):
        print("[?] News coming soon!")
    
    def _movie_info(self, command: str):
        print("[?] Movie info coming soon!")
    
    def _tv_show_info(self, command: str):
        print("[?] TV show info coming soon!")
    
    def _sports_info(self, command: str):
        print(" Sports info coming soon!")
    
    def _weather_forecast(self, command: str):
        print("[INFO] Weather forecast coming soon!")
    
    def _show_map(self, command: str):
        print("[*] Map viewer coming soon!")
    
    def _get_directions(self, command: str):
        print("[?] Directions coming soon!")
    
    def _flight_info(self, command: str):
        print("[?] Flight info coming soon!")
    
    def _hotel_info(self, command: str):
        print("[?] Hotel info coming soon!")
    
    def _restaurant_info(self, command: str):
        print("[*] Restaurant info coming soon!")
    
    def _event_info(self, command: str):
        print("[?] Event info coming soon!")
    
    def _schedule_meeting(self, command: str):
        print("[*] Meeting scheduler coming soon!")
    
    def _make_call(self, command: str):
        print("[?] Call feature coming soon!")
    
    def _send_message(self, command: str):
        print("[*] Message feature coming soon!")
    
    def _wikipedia_search(self, command: str):
        print("[?] Wikipedia search coming soon!")
    
    def _translate_text(self, command: str):
        print("[W] Translation service coming soon!")
    
    def _set_reminder(self, command: str):
        print(" Reminder feature coming soon!")
    
    def _set_alarm(self, command: str):
        print(" Alarm feature coming soon!")
    
    def _set_timer(self, command: str):
        print("[?] Timer feature coming soon!")
    
    def _send_email(self, command: str):
        print("[?] Email feature coming soon!")
    
    def _file_operations(self, command: str):
        print("[?] File operations coming soon!")
    
    def _run_command(self, command: str):
        print("[?] Command execution coming soon!")
    
    def _run_python(self, command: str):
        print("[?] Python execution coming soon!")
    
    def _system_shutdown(self, command: str):
        print("[?] System shutdown coming soon!")
    
    def _system_restart(self, command: str):
        print("[?] System restart coming soon!")
    
    def _take_screenshot(self, command: str):
        print("[?] Screenshot coming soon!")
    
    def _record_screen(self, command: str):
        print("[?] Screen recording coming soon!")
    
    def _play_music(self, command: str):
        print("[?] Music player coming soon!")
    
    def _play_video(self, command: str):
        print("[?] Video player coming soon!")
    
    def _play_game(self, command: str):
        print("[?] Game launcher coming soon!")
    
    def _read_book(self, command: str):
        print("[?] E-reader coming soon!")
    
    def _get_recipe(self, command: str):
        print("[*] Recipe finder coming soon!")
    
    def _get_exercise(self, command: str):
        print("[?] Fitness guide coming soon!")
    
    def _start_meditation(self, command: str):
        print("[?] Meditation guide coming soon!")
    
    def _social_media(self, command: str):
        print("[M] Social media tools coming soon!")
    
    def _backup_files(self, command: str):
        print("[?] Backup utility coming soon!")
        def _update_system(self, command: str):
        print("[?] System update coming soon!")
    
    def _install_package(self, command: str):
        print("[?] Package installer coming soon!")
    
    def _uninstall_package(self, command: str):
        print("[*] Package uninstaller coming soon!")
    
    def _scan_virus(self, command: str):
        print("[*] Virus scanner coming soon!")
    
    def _clean_system(self, command: str):
        print("[?] System cleaner coming soon!")
    
    def _optimize_system(self, command: str):
        print("[*] System optimizer coming soon!")
    
    def _network_info(self, command: str):
        print("[W] Network info coming soon!")
    
    def _wifi_info(self, command: str):
        print("[?] WiFi info coming soon!")
    
    def _bluetooth_info(self, command: str):
        print("[?] Bluetooth info coming soon!")
    
    def _printer_info(self, command: str):
        print("[*] Printer info coming soon!")
    
    def _camera_info(self, command: str):
        print("[?] Camera info coming soon!")
    
    def _microphone_info(self, command: str):
        print("[*] Microphone info coming soon!")
    
    def _speaker_info(self, command: str):
        print("[*] Speaker info coming soon!")
    
    def _display_info(self, command: str):
        print("[*] Display info coming soon!")
    
    def _battery_info(self, command: str):
        print("[?] Battery info coming soon!")
    
    def _disk_info(self, command: str):
        print("[?] Disk info coming soon!")
    
    def _memory_info(self, command: str):
        print("[?] Memory info coming soon!")
    
    def _cpu_info(self, command: str):
        print("[?] CPU info coming soon!")
    
    def _gpu_info(self, command: str):
        print("[?] GPU info coming soon!")
    
    def _os_info(self, command: str):
        print("[*] OS info coming soon!")
    
    def _version_info(self, command: str):
        print("[?] Version info coming soon!")
    
    def _license_info(self, command: str):
        print("[?] License info coming soon!")
    
    def _contact_info(self, command: str):
        print("[*] Contact info coming soon!")
    
    def _donate_info(self, command: str):
        print("[?] Donation info coming soon!")
    
    def _send_feedback(self, command: str):
        print("[*] Feedback system coming soon!")
    
    def _report_bug(self, command: str):
        print("[?] Bug reporting coming soon!")
    
    def _request_feature(self, command: str):
        print(" Feature request coming soon!")
    
    def _show_docs(self, command: str):
        print("[?] Documentation coming soon!")
    
    def _show_tutorial(self, command: str):
        print("[*] Tutorial coming soon!")
    
    def _show_examples(self, command: str):
        print("[?] Examples coming soon!")
    
    def _show_api(self, command: str):
        print("[?] API documentation coming soon!")
    
    def _open_github(self, command: str):
        print("[?] Opening GitHub...")
    
    def _open_website(self, command: str):
        print("[W] Opening website...")
    
    def _open_forum(self, command: str):
        print("[*] Opening forum...")
    
    def _open_blog(self, command: str):
        print("[?] Opening blog...")
    
    def _open_youtube(self, command: str):
        print("[?] Opening YouTube...")
    
    def _open_twitter(self, command: str):
        print("[?] Opening Twitter...")
    
    def _open_facebook(self, command: str):
        print("[?] Opening Facebook...")
    
    def _open_instagram(self, command: str):
        print("[?] Opening Instagram...")
    
    def _open_linkedin(self, command: str):
        print("[?] Opening LinkedIn...")
    
    def _open_reddit(self, command: str):
        print("[?] Opening Reddit...")
    
    def _open_discord(self, command: str):
        print("[?] Opening Discord...")
    
    def _open_slack(self, command: str):
        print("[?] Opening Slack...")
    
    def _open_whatsapp(self, command: str):
        print("[*] Opening WhatsApp...")
    
    def _open_telegram(self, command: str):
        print("[?] Opening Telegram...")
    
    def _open_signal(self, command: str):
        print("[*] Opening Signal...")
    
    # ========== NETWORK COMMANDS ==========
    
    def _ping_host(self, command: str):
        """Ping a network host"""
        try:
            parts = command.split(' ', 1)
            host = parts[1] if len(parts) > 1 else "google.com"
            
            print(f"[?] Pinging {host}...")
            
            # Run ping command based on OS
            param = '-n' if platform.system().lower() == 'windows' else '-c'
            count = '4'
            
            result = subprocess.run(
                ['ping', param, count, host],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print(" Host is reachable")
            else:
                print("[X] Host is not reachable")
            
            print(f"\n{result.stdout}")
            
        except Exception as e:
            print(f"[X] Ping error: {e}")
    
    def _show_ip(self, command: str):
        """Show IP address"""
        try:
            import socket
            
            # Get local IP
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            
            print(f"[W] Network Information:")
            print(f"  Hostname: {hostname}")
            print(f"  Local IP: {local_ip}")
            
            # Try to get public IP
            try:
                import requests
                public_ip = requests.get('https://api.ipify.org').text
                print(f"  Public IP: {public_ip}")
            except:
                print("  Public IP: Could not determine")
            
        except Exception as e:
            print(f"[X] IP error: {e}")
    
    def _about_info(self, command: str):
        """Show about information"""
        about_text = """
[?] JARVIS AI ASSISTANT
Version 1.0.0

A complete AI assistant built with Python, featuring:
 OpenAI GPT integration
 Voice recognition
 Command system
 File management
 System monitoring
 Web utilities
 And much more!

Created with [?] for developers and AI enthusiasts.

[?] GitHub: https://github.com/yourusername/jarvis-ai
[?] Contact: your.email@example.com
[?] Report issues: GitHub Issues

License: MIT
"""
        print(about_text)




