#!/usr/bin/env python3
"""
Jarvis AI Assistant - Main Entry Point
RUN THIS FILE TO START JARVIS!
"""

import os
import sys
import io
import json
import time
import datetime
import platform
import subprocess
import threading
import queue
import random
import hashlib
import pickle
import sqlite3
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from contextlib import contextmanager

# Set UTF-8 encoding for Windows
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

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


class Mood(Enum):
    """Jarvis mood states"""
    HAPPY = "😊"
    THINKING = "🤔"
    WORKING = "⚡"
    CONFUSED = "😕"
    SLEEPY = "😴"
    EXCITED = "🎉"
    SASSY = "😏"
    WORRIED = "😰"


@dataclass
class Conversation:
    """Conversation history entry"""
    timestamp: float
    user_input: str
    response: str
    mood: str
    duration: float


class PluginManager:
    """Manage Jarvis plugins"""
    
    def __init__(self, jarvis):
        self.jarvis = jarvis
        self.plugins = {}
        self.plugin_dir = Path("plugins")
        self.plugin_dir.mkdir(exist_ok=True)
        
    def load_plugins(self):
        """Load all plugins from plugin directory"""
        try:
            sys.path.insert(0, str(self.plugin_dir))
            
            for plugin_file in self.plugin_dir.glob("*.py"):
                if plugin_file.name.startswith("_"):
                    continue
                    
                plugin_name = plugin_file.stem
                try:
                    module = __import__(plugin_name)
                    if hasattr(module, 'setup'):
                        plugin = module.setup(self.jarvis)
                        self.plugins[plugin_name] = plugin
                        self.jarvis.logger.info(f"Loaded plugin: {plugin_name}")
                except Exception as e:
                    self.jarvis.logger.error(f"Failed to load plugin {plugin_name}: {e}")
                    
        except Exception as e:
            self.jarvis.logger.error(f"Plugin loading error: {e}")
            
    def get_plugin_info(self):
        """Get information about loaded plugins"""
        info = []
        for name, plugin in self.plugins.items():
            info.append({
                'name': name,
                'description': getattr(plugin, 'description', 'No description'),
                'version': getattr(plugin, 'version', '1.0.0'),
                'author': getattr(plugin, 'author', 'Unknown')
            })
        return info


class DatabaseManager:
    """Manage SQLite database for Jarvis"""
    
    def __init__(self, db_path="data/jarvis.db"):
        self.db_path = db_path
        Path("data").mkdir(exist_ok=True)
        self.init_database()
        
    def init_database(self):
        """Initialize database tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Conversations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    user_input TEXT,
                    response TEXT,
                    mood TEXT,
                    duration REAL,
                    tags TEXT
                )
            ''')
            
            # Notes table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    title TEXT,
                    content TEXT,
                    category TEXT,
                    pinned INTEGER DEFAULT 0
                )
            ''')
            
            # Tasks table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created REAL,
                    due REAL,
                    task TEXT,
                    priority INTEGER DEFAULT 3,
                    completed INTEGER DEFAULT 0,
                    category TEXT
                )
            ''')
            
            # Memory table (for learning)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT UNIQUE,
                    value TEXT,
                    timestamp REAL,
                    access_count INTEGER DEFAULT 1
                )
            ''')
            
            conn.commit()
            
    @contextmanager
    def get_connection(self):
        """Get database connection with context manager"""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()
            
    def save_conversation(self, user_input, response, mood, duration):
        """Save conversation to database"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO conversations (timestamp, user_input, response, mood, duration)
                VALUES (?, ?, ?, ?, ?)
            ''', (time.time(), user_input, response, mood, duration))
            conn.commit()
            
    def get_recent_conversations(self, limit=10):
        """Get recent conversations"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT timestamp, user_input, response, mood 
                FROM conversations 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            return cursor.fetchall()
            
    def add_note(self, title, content, category="general"):
        """Add a note"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO notes (timestamp, title, content, category)
                VALUES (?, ?, ?, ?)
            ''', (time.time(), title, content, category))
            conn.commit()
            return cursor.lastrowid
            
    def add_task(self, task, due=None, priority=3, category="general"):
        """Add a task"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO tasks (created, due, task, priority, category)
                VALUES (?, ?, ?, ?, ?)
            ''', (time.time(), due, task, priority, category))
            conn.commit()
            return cursor.lastrowid
            
    def remember(self, key, value):
        """Store something in memory"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO memory (key, value, timestamp, access_count)
                VALUES (?, ?, ?, COALESCE(
                    (SELECT access_count + 1 FROM memory WHERE key = ?), 1
                ))
            ''', (key, value, time.time(), key))
            conn.commit()
            
    def recall(self, key):
        """Recall from memory"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE memory SET access_count = access_count + 1 
                WHERE key = ?
            ''', (key,))
            cursor.execute('''
                SELECT value FROM memory WHERE key = ?
            ''', (key,))
            result = cursor.fetchone()
            conn.commit()
            return result[0] if result else None


class PerformanceMonitor:
    """Monitor system performance"""
    
    def __init__(self):
        self.start_time = time.time()
        self.command_times = []
        self.response_times = []
        
    def measure_command(self, func):
        """Decorator to measure command execution time"""
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start
            self.command_times.append(duration)
            return result, duration
        return wrapper
    
    def get_stats(self):
        """Get performance statistics"""
        stats = {
            'uptime': time.time() - self.start_time,
            'total_commands': len(self.command_times),
            'avg_command_time': sum(self.command_times) / len(self.command_times) if self.command_times else 0,
            'avg_response_time': sum(self.response_times) / len(self.response_times) if self.response_times else 0,
        }
        return stats


class JarvisAssistant:
    """Main Jarvis AI Assistant class - Enhanced version"""
    
    def __init__(self):
        """Initialize the assistant"""
        self.logger = None
        self.config = None
        self.ai = None
        self.commands = None
        self.speech = None
        self.running = False
        self.mood = Mood.HAPPY
        self.mood_queue = queue.Queue()
        self.notification_queue = queue.Queue()
        self.db = None
        self.plugins = None
        self.performance = PerformanceMonitor()
        self.user_preferences = {}
        self.background_tasks = []
        
    def initialize(self):
        """Initialize all components"""
        try:
            # Create necessary directories
            for dir_name in ['logs', 'data', 'temp', 'plugins', 'backups', 'profiles']:
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name, exist_ok=True)
            
            # Load configuration
            self.config = ConfigManager().config
            print(f"✅ Configuration loaded: {self.config['assistant']['name']} v{self.config['assistant']['version']}")
            
            # Setup logging
            self.logger = setup_logging(self.config)
            self.logger.info(f"Starting {self.config['assistant']['name']} v{self.config['assistant']['version']}")
            
            # Initialize database
            self.db = DatabaseManager()
            print("✅ Database initialized")
            
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
            
            # Initialize plugin manager
            self.plugins = PluginManager(self)
            self.plugins.load_plugins()
            print(f"✅ Loaded {len(self.plugins.plugins)} plugins")
            
            # Load user preferences
            self.load_preferences()
            
            # Start background threads
            self.start_background_tasks()
            
            return True
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            if self.logger:
                self.logger.error(f"Failed to initialize: {e}")
            return False
    
    def load_preferences(self):
        """Load user preferences"""
        try:
            prefs_file = Path("data/preferences.json")
            if prefs_file.exists():
                with open(prefs_file, 'r') as f:
                    self.user_preferences = json.load(f)
            else:
                self.user_preferences = {
                    'theme': 'default',
                    'voice_speed': 1.0,
                    'auto_save': True,
                    'notification_sound': True,
                    'language': 'en',
                    'shortcuts': {}
                }
        except Exception as e:
            self.logger.error(f"Failed to load preferences: {e}")
            
    def save_preferences(self):
        """Save user preferences"""
        try:
            with open(Path("data/preferences.json"), 'w') as f:
                json.dump(self.user_preferences, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save preferences: {e}")
    
    def start_background_tasks(self):
        """Start background threads"""
        # Mood updater
        mood_thread = threading.Thread(target=self._update_mood, daemon=True)
        mood_thread.start()
        
        # Notification handler
        notify_thread = threading.Thread(target=self._handle_notifications, daemon=True)
        notify_thread.start()
        
        # Auto-save thread
        if self.user_preferences.get('auto_save', True):
            save_thread = threading.Thread(target=self._auto_save, daemon=True)
            save_thread.start()
            
    def _update_mood(self):
        """Update Jarvis mood based on context"""
        moods = list(Mood)
        last_interaction = time.time()
        
        while self.running:
            try:
                time.sleep(30)  # Check every 30 seconds
                
                # Change mood based on inactivity
                if time.time() - last_interaction > 300:  # 5 minutes
                    self.mood = Mood.SLEEPY
                elif random.random() < 0.1:  # Random mood changes
                    self.mood = random.choice(moods)
                    
            except Exception:
                pass
                
    def _handle_notifications(self):
        """Handle queued notifications"""
        while self.running:
            try:
                notification = self.notification_queue.get(timeout=1)
                self._show_notification(notification)
            except queue.Empty:
                continue
            except Exception:
                pass
                
    def _auto_save(self):
        """Auto-save data periodically"""
        while self.running:
            try:
                time.sleep(300)  # Every 5 minutes
                self.commands.save_data()
                self.save_preferences()
                self.logger.info("Auto-save completed")
            except Exception:
                pass
                
    def _show_notification(self, notification):
        """Show a notification"""
        title = notification.get('title', 'Jarvis')
        message = notification.get('message', '')
        
        # Try to show system notification
        try:
            if platform.system() == 'Windows':
                from plyer import notification
                notification.notify(
                    title=title,
                    message=message,
                    timeout=5
                )
            elif platform.system() == 'Darwin':  # macOS
                os.system(f'''
                    osascript -e 'display notification "{message}" with title "{title}"'
                ''')
            else:  # Linux
                os.system(f'notify-send "{title}" "{message}"')
        except Exception:
            # Fallback to console notification
            print(f"\n🔔 {title}: {message}")
    
    def show_welcome(self):
        """Show welcome message"""
        print("\n" + "="*70)
        print(f"🤖 {self.config['assistant']['name']} AI Assistant v{self.config['assistant']['version']}")
        print("="*70)
        
        # Show system info
        print(f"🖥️ System: {platform.system()} {platform.release()}")
        print(f"📅 Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🔌 Plugins: {len(self.plugins.plugins)} active")
        print("-"*70)
        print("\nCommands: 'help' | 'menu' | 'quit' | 'mood' | 'stats'")
        print("Type anything to chat with AI!")
        print("-"*70)
        
        # Random welcome message
        welcomes = [
            f"Hello! I am {self.config['assistant']['name']}. How can I assist you today?",
            f"At your service! What can I do for you?",
            f"Ready and waiting! What's on your mind?",
            f"Systems online and ready to help!"
        ]
        welcome_text = random.choice(welcomes)
        
        print(f"🔊 {welcome_text}")
        self.speech.speak(welcome_text)
    
    def show_menu(self):
        """Show interactive menu"""
        print("\n" + "="*70)
        print(f"📋 JARVIS MAIN MENU {self.mood.value}")
        print("="*70)
        print("\n1. 💬 Chat Mode (Talk to AI)")
        print("2. ⚡ Command Mode (Quick commands)")
        print("3. 📝 Notes & Todo")
        print("4. 🛠️ System Tools")
        print("5. 🌐 Web & Search")
        print("6. 📊 Statistics & Memory")
        print("7. 🎮 Entertainment")
        print("8. 🔌 Plugins")
        print("9. ⚙️ Settings")
        print("10. ℹ️ Help")
        print("11. 🚪 Exit")
        print("-"*70)
        
        choice = input("\nSelect option (1-11): ").strip()
        
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
            self.stats_mode()
        elif choice == "7":
            self.entertainment_mode()
        elif choice == "8":
            self.plugin_mode()
        elif choice == "9":
            self.settings_mode()
        elif choice == "10":
            self.show_help()
        elif choice == "11":
            return False
        else:
            print("❌ Invalid choice")
            
        return True
    
    def chat_mode(self):
        """Enhanced chat with AI"""
        print("\n" + "="*70)
        print(f"💬 CHAT MODE - Type 'back' to return to menu {self.mood.value}")
        print("="*70)
        
        conversation_id = int(time.time())
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ['back', 'menu', 'exit']:
                    # Save conversation summary
                    self.db.remember(f"last_conversation_{conversation_id}", 
                                    f"Ended at {datetime.datetime.now()}")
                    break
                
                # Update mood
                self.mood = Mood.THINKING
                print(f"{self.mood.value} Thinking...", end='\r')
                
                # Measure response time
                start_time = time.time()
                response = self.ai.chat(user_input)
                duration = time.time() - start_time
                
                self.performance.response_times.append(duration)
                
                # Random mood after response
                self.mood = random.choice([Mood.HAPPY, Mood.EXCITED, Mood.SASSY])
                
                print("\n" + "-"*70)
                print(f"💡 {self.config['assistant']['name']} {self.mood.value}:")
                print("-"*70)
                print(response)
                print("-"*70)
                print(f"⏱️ Response time: {duration:.2f}s")
                
                # Speak response
                self.speech.speak(response[:200])
                
                # Save to database
                self.db.save_conversation(
                    user_input, response, self.mood.value, duration
                )
                
                # Check for learning opportunities
                if "remember" in user_input.lower() or "remember that" in user_input.lower():
                    self._learn_from_conversation(user_input, response)
                
            except KeyboardInterrupt:
                print("\n⏹️ Returning to menu...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                self.logger.error(f"Chat error: {e}")
    
    def _learn_from_conversation(self, user_input, response):
        """Learn from conversation"""
        try:
            # Extract key information
            words = user_input.lower().split()
            if "remember" in words:
                idx = words.index("remember")
                if idx < len(words) - 1:
                    key = f"memory_{hashlib.md5(user_input.encode()).hexdigest()[:8]}"
                    self.db.remember(key, response)
                    print("🧠 I'll remember that!")
        except Exception as e:
            self.logger.error(f"Learning error: {e}")
    
    def stats_mode(self):
        """Show statistics and memory"""
        print("\n" + "="*70)
        print("📊 STATISTICS & MEMORY")
        print("="*70)
        
        # Performance stats
        stats = self.performance.get_stats()
        print(f"\n⏱️ Performance:")
        print(f"  Uptime: {stats['uptime']/3600:.1f} hours")
        print(f"  Total commands: {stats['total_commands']}")
        print(f"  Avg response time: {stats['avg_response_time']*1000:.1f}ms")
        
        # Recent conversations
        print(f"\n💬 Recent conversations:")
        recent = self.db.get_recent_conversations(5)
        for ts, user, resp, mood in recent:
            time_str = datetime.datetime.fromtimestamp(ts).strftime("%H:%M")
            print(f"  {time_str} {mood}: {user[:30]}...")
        
        # Memory stats
        print(f"\n🧠 Memory usage:")
        memory_file = Path("data/jarvis.db")
        if memory_file.exists():
            size = memory_file.stat().st_size / 1024
            print(f"  Database size: {size:.1f} KB")
            
        input("\nPress Enter to continue...")
    
    def entertainment_mode(self):
        """Entertainment features"""
        print("\n" + "="*70)
        print("🎮 ENTERTAINMENT MODE")
        print("="*70)
        print("\n1. 🎲 Roll dice")
        print("2. 🃏 Card game")
        print("3. 🔢 Number guessing")
        print("4. 🎵 Play music suggestion")
        print("5. 📖 Tell a story")
        print("6. 🎭 Random fact")
        print("7. 🧩 Puzzle")
        print("8. 🎨 ASCII art")
        print("9. Back to menu")
        
        choice = input("\nSelect option: ").strip()
        
        if choice == "1":
            dice = random.randint(1, 6)
            print(f"🎲 You rolled a {dice}!")
            self.speech.speak(f"You rolled a {dice}")
            
        elif choice == "2":
            cards = ['Ace', 'King', 'Queen', 'Jack', '10', '9', '8', '7', '6']
            suits = ['♥️', '♦️', '♣️', '♠️']
            card = f"{random.choice(cards)} of {random.choice(suits)}"
            print(f"🃏 Your card: {card}")
            
        elif choice == "3":
            self._number_guessing_game()
            
        elif choice == "4":
            self._music_suggestion()
            
        elif choice == "5":
            self._tell_story()
            
        elif choice == "6":
            self._random_fact()
            
        elif choice == "7":
            self._puzzle_game()
            
        elif choice == "8":
            self._show_ascii_art()
    
    def _number_guessing_game(self):
        """Number guessing game"""
        print("\n🔢 NUMBER GUESSING GAME")
        print("I'm thinking of a number between 1 and 100...")
        
        number = random.randint(1, 100)
        attempts = 0
        max_attempts = 7
        
        while attempts < max_attempts:
            try:
                guess = input(f"Attempt {attempts+1}/{max_attempts}: ").strip()
                if not guess:
                    continue
                    
                if guess.lower() == 'quit':
                    print(f"The number was {number}")
                    break
                    
                guess = int(guess)
                attempts += 1
                
                if guess < number:
                    print("Higher! ⬆️")
                elif guess > number:
                    print("Lower! ⬇️")
                else:
                    print(f"🎉 Correct! You got it in {attempts} attempts!")
                    self.speech.speak(f"Congratulations! You got it in {attempts} attempts!")
                    break
            except ValueError:
                print("Please enter a number!")
                
        if attempts >= max_attempts and guess != number:
            print(f"😔 Out of attempts! The number was {number}")
    
    def _music_suggestion(self):
        """Suggest music based on time of day"""
        hour = datetime.datetime.now().hour
        
        if 5 <= hour < 12:
            mood = "morning"
            songs = ["Here Comes the Sun", "Morning Has Broken", "Beautiful Day"]
        elif 12 <= hour < 18:
            mood = "afternoon"
            songs = ["Happy", "Walking on Sunshine", "Good Vibrations"]
        elif 18 <= hour < 22:
            mood = "evening"
            songs = ["Perfect", "All of Me", "Thinking Out Loud"]
        else:
            mood = "night"
            songs = ["Clair de Lune", "Moonlight Sonata", "Nocturne"]
            
        print(f"\n🎵 For this {mood}, I suggest: {random.choice(songs)}")
        print("Would you like me to play it? (y/n)")
        
        if input().lower().startswith('y'):
            print("🎶 Opening YouTube... (feature coming soon!)")
    
    def _tell_story(self):
        """Tell a random story"""
        stories = [
            "Once upon a time, in a digital kingdom far, far away...",
            "In the year 2042, AI assistants became the best friends of humanity...",
            "There was a clever AI named Jarvis who loved to help people...",
            "In a world where technology and magic intertwined..."
        ]
        
        story = random.choice(stories)
        print(f"\n📖 {story}")
        
        # Continue the story interactively
        for _ in range(3):
            choice = input("\nWhat happens next? (type 'continue' or 'end'): ").lower()
            if choice == 'continue':
                continuations = [
                    "And so the adventure continued...",
                    "Suddenly, something unexpected happened...",
                    "The hero faced a new challenge..."
                ]
                print(random.choice(continuations))
            else:
                print("🎉 The End!")
                break
    
    def _random_fact(self):
        """Show random fact"""
        facts = [
            "Honey never spoils. Archaeologists found 3000-year-old honey in Egyptian tombs!",
            "A day on Venus is longer than a year on Venus.",
            "Bananas are berries, but strawberries aren't.",
            "Octopuses have three hearts and blue blood.",
            "The Eiffel Tower can be 15 cm taller during summer.",
            "Your brain generates enough electricity to power a lightbulb."
        ]
        
        fact = random.choice(facts)
        print(f"\n🎭 Did you know?\n{fact}")
        self.speech.speak(fact)
    
    def _puzzle_game(self):
        """Simple puzzle game"""
        puzzles = [
            {
                "question": "I speak without a mouth and hear without ears. I have no body, but I come alive with wind. What am I?",
                "answer": "echo"
            },
            {
                "question": "The more you take, the more you leave behind. What are they?",
                "answer": "footsteps"
            },
            {
                "question": "What has keys but can't open locks?",
                "answer": "piano"
            }
        ]
        
        puzzle = random.choice(puzzles)
        print(f"\n🧩 PUZZLE:\n{puzzle['question']}")
        
        attempts = 3
        while attempts > 0:
            answer = input("Your answer: ").lower().strip()
            if answer == puzzle['answer']:
                print("🎉 Correct! You're brilliant!")
                break
            else:
                attempts -= 1
                if attempts > 0:
                    print(f"Not quite. Try again! ({attempts} attempts left)")
                else:
                    print(f"The answer was: {puzzle['answer']}")
        def _show_ascii_art(self):
        """Show ASCII art"""
        arts = {
            "robot": """
    ╱|、
   (˚ˎ 。7  
    |、˜〵          
    じしˍ,)ノ
            """,
            "cat": """
     /\_/\\
    ( o.o )
     > ^ <
            """,
            "jarvis": """
     ╔╦╗╦═╗╦ ╦╔╦╗╔═╗
     ║║║╠╦╝║ ║ ║ ║ ║
     ╩ ╩╩╚═╚═╝ ╩ ╚═╝
            """
        }
        
        print("\n🎨 ASCII ART")
        for name, art in arts.items():
            print(f"\n{name.title()}:{art}")
    
    def plugin_mode(self):
        """Plugin management"""
        print("\n" + "="*70)
        print("🔌 PLUGIN MANAGER")
        print("="*70)
        
        plugins = self.plugins.get_plugin_info()
        
        if not plugins:
            print("\nNo plugins installed.")
            print("\nTo install plugins:")
            print("1. Create a .py file in the 'plugins' directory")
            print("2. Add a setup(jarvis) function")
            print("3. Define description, version, and author attributes")
        else:
            print(f"\nActive plugins ({len(plugins)}):")
            for i, plugin in enumerate(plugins, 1):
                print(f"\n  {i}. {plugin['name']} v{plugin['version']}")
                print(f"     {plugin['description']}")
                print(f"     by {plugin['author']}")
                
        input("\nPress Enter to continue...")
    
    def settings_mode(self):
        """Enhanced settings"""
        print("\n" + "="*70)
        print("⚙️ SETTINGS")
        print("="*70)
        
        while True:
            print(f"\nCurrent settings:")
            print(f"  1. AI Model: {self.config['openai']['model']}")
            print(f"  2. Voice: {'Enabled' if self.config['speech']['enabled'] else 'Disabled'}")
            print(f"  3. Voice Speed: {self.user_preferences.get('voice_speed', 1.0)}x")
            print(f"  4. Theme: {self.user_preferences.get('theme', 'default')}")
            print(f"  5. Auto-save: {self.user_preferences.get('auto_save', True)}")
            print(f"  6. Notification Sound: {self.user_preferences.get('notification_sound', True)}")
            print(f"  7. Show current config")
            print(f"  8. Reload config")
            print(f"  9. Clear conversation history")
            print(f"  10. Backup data")
            print(f"  11. Restore backup")
            print(f"  12. Back to main menu")
            
            choice = input("\nSelect option (1-12): ").strip()
            
            if choice == "12":
                break
            elif choice == "1":
                model = input(f"Enter new model [current: {self.config['openai']['model']}]: ").strip()
                if model:
                    self.config['openai']['model'] = model
                    print(f"✅ Model updated to: {model}")
            elif choice == "2":
                enabled = not self.config['speech']['enabled']
                self.config['speech']['enabled'] = enabled
                print(f"✅ Voice {'enabled' if enabled else 'disabled'}")
            elif choice == "3":
                speed = input("Enter voice speed (0.5-2.0): ").strip()
                try:
                    speed = float(speed)
                    if 0.5 <= speed <= 2.0:
                        self.user_preferences['voice_speed'] = speed
                        print(f"✅ Voice speed set to {speed}x")
                except ValueError:
                    print("❌ Invalid speed")
            elif choice == "4":
                themes = ['default', 'dark', 'light', 'colorful']
                print(f"Themes: {', '.join(themes)}")
                theme = input("Enter theme: ").strip().lower()
                if theme in themes:
                    self.user_preferences['theme'] = theme
                    print(f"✅ Theme set to {theme}")
            elif choice == "5":
                auto_save = not self.user_preferences.get('auto_save', True)
                self.user_preferences['auto_save'] = auto_save
                print(f"✅ Auto-save {'enabled' if auto_save else 'disabled'}")
            elif choice == "6":
                sound = not self.user_preferences.get('notification_sound', True)
                self.user_preferences['notification_sound'] = sound
                print(f"✅ Notification sound {'enabled' if sound else 'disabled'}")
            elif choice == "7":
                self.commands.process("config")
            elif choice == "8":
                self.config = ConfigManager().reload_config()
                print("✅ Configuration reloaded")
            elif choice == "9":
                self.ai.clear_history()
                print("✅ Conversation history cleared")
            elif choice == "10":
                self._backup_data()
            elif choice == "11":
                self._restore_backup()
                
        self.save_preferences()
    
    def _backup_data(self):
        """Create a backup of all data"""
        try:
            backup_dir = Path("backups")
            backup_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = backup_dir / f"backup_{timestamp}.zip"
            
            import zipfile
            with zipfile.ZipFile(backup_file, 'w') as zipf:
                for folder in ['data', 'logs', 'config']:
                    folder_path = Path(folder)
                    if folder_path.exists():
                        for file in folder_path.rglob('*'):
                            if file.is_file():
                                zipf.write(file, file.relative_to(Path('.')))
                                
            print(f"✅ Backup created: {backup_file}")
            self.notification_queue.put({
                'title': 'Backup Complete',
                'message': f'Data backed up to {backup_file.name}'
            })
            
        except Exception as e:
            print(f"❌ Backup failed: {e}")
    
    def _restore_backup(self):
        """Restore from backup"""
        backup_dir = Path("backups")
        if not backup_dir.exists():
            print("❌ No backups found")
            return
            
        backups = list(backup_dir.glob("*.zip"))
        if not backups:
            print("❌ No backups found")
            return
            
        print("\nAvailable backups:")
        for i, backup in enumerate(backups, 1):
            size = backup.stat().st_size / 1024 / 1024
            print(f"  {i}. {backup.name} ({size:.1f} MB)")
            
        try:
            choice = int(input("\nSelect backup to restore (0 to cancel): "))
            if 1 <= choice <= len(backups):
                backup_file = backups[choice-1]
                
                import zipfile
                with zipfile.ZipFile(backup_file, 'r') as zipf:
                    zipf.extractall('.')
                    
                print(f"✅ Restored from {backup_file.name}")
                print("⚠️ Please restart Jarvis for changes to take effect")
            elif choice != 0:
                print("❌ Invalid choice")
        except ValueError:
            print("❌ Invalid input")
    
    def command_mode(self):
        """Quick command mode"""
        print("\n" + "="*70)
        print("⚡ COMMAND MODE - Type 'back' to return to menu")
        print("="*70)
        print("\nQuick commands:")
        print("  time / date / weather [city] / joke / quote")
        print("  open [app] / search [query] / calculate [math]")
        print("  system / ip / ping [host] / clear")
        print("  remember [key] [value] / recall [key]")
        print("  mood / stats / fact / game")
        print("-"*70)
        
        while True:
            try:
                user_input = input("\nCommand: ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ['back', 'menu', 'exit']:
                    break
                
                # Measure command execution
                result, duration = self.performance.measure_command(
                    self.commands.process
                )(user_input)
                
                print(f"⏱️ Command completed in {duration*1000:.1f}ms")
                
            except KeyboardInterrupt:
                print("\n⏹️ Returning to menu...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def notes_mode(self):
        """Enhanced notes and todo management"""
        print("\n" + "="*70)
        print("📝 NOTES & TODO - Type 'back' to return to menu")
        print("="*70)
        print("\nCommands:")
        print("  note [title] [content]   - Take a note")
        print("  notes [category]          - View all notes")
        print("  note search [text]        - Search notes")
        print("  note edit [#] [content]   - Edit note")
        print("  note delete [#]            - Delete note")
        print("  note pin [#]               - Pin note")
        print("  todo add [task]            - Add todo")
        print("  todo list [category]       - View todos")
        print("  todo complete [#]          - Complete todo")
        print("  todo remove [#]            - Remove todo")
        print("  todo clear                 - Clear completed")
        print("  todo due [#] [date]        - Set due date")
        print("-"*70)
        
:
    sys.exit(main())
        while True:
            try:
                user_input = input("\nNotes> ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ['back', 'menu', 'exit']:
                    break
                
                # Parse and process notes commands
                parts = user_input.split()
                cmd = parts[0].lower()
                
                if cmd == "note":
                    if len(parts) >= 3 and parts[1] == "search":
                        search_term = " ".join(parts[2:])
                        self._search_notes(search_term)
                    elif len(parts) >= 3:
                        title = parts[1]
                        content = " ".join(parts[2:])
                        note_id = self.db.add_note(title, content)
                        print(f"✅ Note #{note_id} saved: {title}")
                    else:
                        self._list_notes()
                        
                elif cmd == "todo":
                    self._handle_todo(parts)
                else:
                    self.commands.process(user_input)
                
            except KeyboardInterrupt:
                print("\n⏹️ Returning to menu...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def _search_notes(self, term):
        """Search notes"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, title, content, timestamp 
                FROM notes 
                WHERE title LIKE ? OR content LIKE ?
                ORDER BY pinned DESC, timestamp DESC
            ''', (f'%{term}%', f'%{term}%'))
            
            results = cursor.fetchall()
            
            if results:
                print(f"\n📝 Found {len(results)} notes:")
                for id, title, content, ts in results:
                    time_str = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
                    preview = content[:50] + "..." if len(content) > 50 else content
                    print(f"\n  #{id}: {title} ({time_str})")
                    print(f"      {preview}")
            else:
                print("No notes found")
    
    def _list_notes(self, category=None):
        """List notes"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            if category:
                cursor.execute('''
                    SELECT id, title, timestamp, pinned 
                    FROM notes 
                    WHERE category = ?
                    ORDER BY pinned DESC, timestamp DESC
                ''', (category,))
            else:
                cursor.execute('''
                    SELECT id, title, timestamp, pinned 
                    FROM notes 
                    ORDER BY pinned DESC, timestamp DESC
                ''')
            
            notes = cursor.fetchall()
            
            if notes:
                print(f"\n📝 Notes ({len(notes)}):")
                for id, title, ts, pinned in notes:
                    time_str = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
                    pin_mark = "📌" if pinned else "  "
                    print(f"  {pin_mark} #{id}: {title} ({time_str})")
            else:
                print("No notes found")
    
    def _handle_todo(self, parts):
        """Handle todo commands"""
        if len(parts) < 2:
            self._list_todos()
            return
            
        subcmd = parts[1].lower()
        
        if subcmd == "add" and len(parts) >= 3:
            task = " ".join(parts[2:])
            task_id = self.db.add_task(task)
            print(f"✅ Task #{task_id} added: {task}")
            
        elif subcmd == "list":
            category = parts[2] if len(parts) >= 3 else None
            self._list_todos(category)
            
        elif subcmd == "complete" and len(parts) >= 3:
            try:
                task_id = int(parts[2])
                with self.db.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        UPDATE tasks SET completed = 1 
                        WHERE id = ?
                    ''', (task_id,))
                    conn.commit()
                    if cursor.rowcount > 0:
                        print(f"✅ Task #{task_id} completed!")
                    else:
                        print(f"❌ Task #{task_id} not found")
            except ValueError:
                print("❌ Invalid task ID")
                
        elif subcmd == "remove" and len(parts) >= 3:
            try:
                task_id = int(parts[2])
                with self.db.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('DELETE FROM tasks WHERE id = ?', (task_id,))
                    conn.commit()
                    if cursor.rowcount > 0:
                        print(f"✅ Task #{task_id} removed")
                    else:
                        print(f"❌ Task #{task_id} not found")
            except ValueError:
                print("❌ Invalid task ID")
                
        elif subcmd == "clear":
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM tasks WHERE completed = 1')
                conn.commit()
                print(f"✅ Cleared {cursor.rowcount} completed tasks")
                
        elif subcmd == "due" and len(parts) >= 4:
            try:
                task_id = int(parts[2])
                due_date = parts[3]
                # Simple date parsing - could be enhanced
                with self.db.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        UPDATE tasks SET due = ? 
                        WHERE id = ?
                    ''', (due_date, task_id))
                    conn.commit()
                    print(f"✅ Due date set for task #{task_id}")
            except ValueError:
                print("❌ Invalid input")
    
    def _list_todos(self, category=None):
        """List todos"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            if category:
                cursor.execute('''
                    SELECT id, task, priority, completed, due 
                    FROM tasks 
                    WHERE category = ? AND completed = 0
                    ORDER BY priority, due
                ''', (category,))
            else:
                cursor.execute('''
                    SELECT id, task, priority, completed, due 
                    FROM tasks 
                    WHERE completed = 0
                    ORDER BY priority, due
                ''')
            
            tasks = cursor.fetchall()
            
            if tasks:
                print(f"\n📋 Todo list ({len(tasks)}):")
                for id, task, priority, completed, due in tasks:
                    priority_mark = "🔴" if priority <= 2 else "🟡" if priority <= 4 else "🟢"
                    due_str = f" (due: {due})" if due else ""
                    print(f"  {priority_mark} #{id}: {task}{due_str}")
            else:
                print("No pending tasks!")
    
    def system_mode(self):
        """Enhanced system tools"""
        print("\n" + "="*70)
        print("🛠️ SYSTEM TOOLS - Type 'back' to return to menu")
        print("="*70)
        print("\nSystem commands:")
        print("  system     - System information")
        print("  processes  - Running processes")
        print("  disk       - Disk usage")
        print("  network    - Network information")
        print("  battery    - Battery status")
        print("  cpu        - CPU usage")
        print("  memory     - Memory usage")
        print("  users      - Logged in users")
        print("  services   - Running services")
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
                
                # Enhanced system commands
                if user_input.lower() == "cpu":
                    self._show_cpu_usage()
                elif user_input.lower() == "memory":
                    self._show_memory_usage()
                elif user_input.lower() == "disk":
                    self._show_disk_usage()
                elif user_input.lower() == "battery":
                    self._show_battery_status()
                elif user_input.lower() == "users":
                    self._show_logged_users()
                elif user_input.lower() == "services":
                    self._show_services()
                else:
                    self.commands.process(user_input)
                
            except KeyboardInterrupt:
                print("\n⏹️ Returning to menu...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def _show_cpu_usage(self):
        """Show CPU usage"""
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            print(f"\n🖥️ CPU Information:")
            print(f"  Usage: {cpu_percent}%")
            print(f"  Cores: {cpu_count}")
            if cpu_freq:
                print(f"  Frequency: {cpu_freq.current:.0f} MHz")
                
            # Create ASCII bar
            bar_length = 50
            filled = int(bar_length * cpu_percent / 100)
            bar = "█" * filled + "░" * (bar_length - filled)
            print(f"\n  [{bar}] {cpu_percent}%")
            
        except ImportError:
            print("psutil not installed. Install with: pip install psutil")
        except Exception as e:
            print(f"Error getting CPU info: {e}")
    
    def _show_memory_usage(self):
        """Show memory usage"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            
            print(f"\n🧠 Memory Information:")
            print(f"  Total: {memory.total / (1024**3):.1f} GB")
            print(f"  Available: {memory.available / (1024**3):.1f} GB")
            print(f"  Used: {memory.used / (1024**3):.1f} GB ({memory.percent}%)")
                        # Create ASCII bar
            bar_length = 50
            filled = int(bar_length * memory.percent / 100)
            bar = "█" * filled + "░" * (bar_length - filled)
            print(f"\n  [{bar}] {memory.percent}%")
            
        except ImportError:
            print("psutil not installed")
        except Exception as e:
            print(f"Error getting memory info: {e}")
    
    def _show_disk_usage(self):
        """Show disk usage"""
        try:
            import psutil
            disk = psutil.disk_usage('/')
            
            print(f"\n💾 Disk Usage:")
            print(f"  Total: {disk.total / (1024**3):.1f} GB")
            print(f"  Used: {disk.used / (1024**3):.1f} GB ({disk.percent}%)")
            print(f"  Free: {disk.free / (1024**3):.1f} GB")
            
            # Create ASCII bar
            bar_length = 50
            filled = int(bar_length * disk.percent / 100)
            bar = "█" * filled + "░" * (bar_length - filled)
            print(f"\n  [{bar}] {disk.percent}%")
            
        except ImportError:
            print("psutil not installed")
        except Exception as e:
            print(f"Error getting disk info: {e}")
    
    def _show_battery_status(self):
        """Show battery status"""
        try:
            import psutil
            battery = psutil.sensors_battery()
            
            if battery:
                print(f"\n🔋 Battery Status:")
                print(f"  Percentage: {battery.percent}%")
                print(f"  Charging: {'Yes' if battery.power_plugged else 'No'}")
                if battery.secsleft != -1:
                    hours = battery.secsleft // 3600
                    minutes = (battery.secsleft % 3600) // 60
                    print(f"  Time left: {hours}h {minutes}m")
            else:
                print("No battery detected")
                
        except ImportError:
            print("psutil not installed")
        except Exception as e:
            print(f"Error getting battery info: {e}")
    
    def _show_logged_users(self):
        """Show logged in users"""
        try:
            import psutil
            users = psutil.users()
            
            if users:
                print(f"\n👥 Logged in users:")
                for user in users:
                    print(f"  {user.name} from {user.host} (since {user.started})")
            else:
                print("No other users logged in")
                
        except ImportError:
            print("psutil not installed")
        except Exception as e:
            print(f"Error getting user info: {e}")
    
    def _show_services(self):
        """Show running services (Windows only)"""
        if platform.system() == 'Windows':
            try:
                result = subprocess.run(
                    ['sc', 'query', 'state=', 'all'],
                    capture_output=True,
                    text=True
                )
                
                lines = result.stdout.split('\n')
                services = []
                current_service = {}
                
                for line in lines:
                    if 'SERVICE_NAME:' in line:
                        if current_service:
                            services.append(current_service)
                        current_service = {'name': line.split(':')[1].strip()}
                    elif 'DISPLAY_NAME:' in line:
                        current_service['display'] = line.split(':')[1].strip()
                    elif 'STATE' in line:
                        if 'RUNNING' in line:
                            current_service['state'] = 'Running'
                        else:
                            current_service['state'] = 'Stopped'
                            
                if current_service:
                    services.append(current_service)
                
                # Show running services
                running = [s for s in services if s.get('state') == 'Running']
                print(f"\n🔄 Running services ({len(running)}):")
                for service in running[:10]:  # Show first 10
                    print(f"  • {service.get('display', service['name'])}")
                    
                if len(running) > 10:
                    print(f"  ... and {len(running) - 10} more")
                    
            except Exception as e:
                print(f"Error getting services: {e}")
        else:
            # Linux/Unix
            try:
                result = subprocess.run(
                    ['systemctl', 'list-units', '--type=service', '--state=running'],
                    capture_output=True,
                    text=True
                )
                print(result.stdout[:500])  # Show first 500 chars
            except Exception:
                print("Could not list services")
    
    def web_mode(self):
        """Enhanced web and search tools"""
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
        print("  define [word]      - Dictionary definition")
        print("  synonym [word]     - Find synonyms")
        print("  antonym [word]     - Find antonyms")
        print("  stock [symbol]     - Stock price")
        print("  crypto [coin]      - Cryptocurrency price")
        print("  map [location]     - Show on map")
        print("-"*70)
        
        while True:
            try:
                user_input = input("\nWeb> ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ['back', 'menu', 'exit']:
                    break
                
                # Enhanced web commands
                parts = user_input.split()
                cmd = parts[0].lower()
                
                if cmd == "define" and len(parts) >= 2:
                    word = " ".join(parts[1:])
                    self._dictionary_definition(word)
                elif cmd == "synonym" and len(parts) >= 2:
                    word = " ".join(parts[1:])
                    self._thesaurus_lookup(word, "synonym")
                elif cmd == "antonym" and len(parts) >= 2:
                    word = " ".join(parts[1:])
                    self._thesaurus_lookup(word, "antonym")
                elif cmd == "stock" and len(parts) >= 2:
                    symbol = parts[1].upper()
                    self._stock_price(symbol)
                elif cmd == "crypto" and len(parts) >= 2:
                    coin = parts[1].lower()
                    self._crypto_price(coin)
                elif cmd == "map" and len(parts) >= 2:
                    location = " ".join(parts[1:])
                    self._open_map(location)
                else:
                    self.commands.process(user_input)
                
            except KeyboardInterrupt:
                print("\n⏹️ Returning to menu...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def _dictionary_definition(self, word):
        """Get word definition"""
        try:
            import requests
            url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()[0]
                print(f"\n📖 {word.title()}:")
                
                for meaning in data.get('meanings', [])[:2]:
                    part_of_speech = meaning.get('partOfSpeech', '')
                    for definition in meaning.get('definitions', [])[:2]:
                        print(f"  [{part_of_speech}] {definition.get('definition', '')}")
                        if definition.get('example'):
                            print(f"    Example: \"{definition.get('example')}\"")
            else:
                print(f"Could not find definition for '{word}'")
                
        except ImportError:
            print("requests not installed. Install with: pip install requests")
        except Exception as e:
            print(f"Error getting definition: {e}")
    
    def _thesaurus_lookup(self, word, lookup_type):
        """Look up synonyms/antonyms"""
        try:
            import requests
            # Using Datamuse API
            url = f"https://api.datamuse.com/words?rel_{lookup_type}={word}"
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                if data:
                    print(f"\n{lookup_type.title()}s for '{word}':")
                    for item in data[:10]:
                        print(f"  • {item.get('word')}")
                else:
                    print(f"No {lookup_type}s found for '{word}'")
            else:
                print(f"Could not find {lookup_type}s for '{word}'")
                
        except ImportError:
            print("requests not installed")
        except Exception as e:
            print(f"Error looking up {lookup_type}s: {e}")
    
    def _stock_price(self, symbol):
        """Get stock price"""
        print(f"\n📈 Stock price for {symbol}:")
        print("(API key required for real-time data)")
        print("Demo: AAPL $175.34 (+1.2%)")
        
        # This would use a real API with proper authentication
    
    def _crypto_price(self, coin):
        """Get cryptocurrency price"""
        print(f"\n💰 {coin.upper()} price:")
        print("(API key required for real-time data)")
        print(f"Demo: {coin.upper()} $45,678.90")
    
    def _open_map(self, location):
        """Open location in maps"""
        import webbrowser
        url = f"https://www.google.com/maps/search/{location.replace(' ', '+')}"
        webbrowser.open(url)
        print(f"🗺️ Opening map for: {location}")
    
    def show_help(self):
        """Show comprehensive help"""
        print("\n" + "="*70)
        print("ℹ️ JARVIS HELP")
        print("="*70)
        
        help_categories = {
            "General": ["help", "menu", "quit", "clear", "mood", "stats"],
            "Chat": ["chat", "ask [question]", "remember [info]", "recall"],
            "Commands": ["time", "date", "weather [city]", "joke", "quote", "calculate"],
            "System": ["system", "cpu", "memory", "disk", "battery", "processes"],
            "Web": ["search [query]", "youtube [query]", "wikipedia [topic]", "news"],
            "Notes": ["note [title] [text]", "notes", "todo [task]"],
            "Entertainment": ["game", "fact", "story", "ascii", "roll"],
            "Settings": ["config", "settings", "backup", "restore"]
        }
        
        for category, commands in help_categories.items():
            print(f"\n{category}:")
            print("  " + ", ".join(commands))
            
        print("\n" + "-"*70)
        print("For detailed help on a specific command, type: help [command]")
        print("Example: help weather")
        
        # Check for plugin help
        if self.plugins.plugins:
            print("\n🔌 Installed plugins:")
            for name, plugin in self.plugins.plugins.items():
                print(f"  {name}: {getattr(plugin, 'description', 'No description')}")
        
        input("\nPress Enter to continue...")
    
    def run(self):
        """Main run loop"""
        try:
            # Initialize
            if not self.initialize():
                return False
            
            self.running = True
            self.show_welcome()
            
            # Main loop
            while self.running:
                try:
                    if not self.show_menu():
                        break
                        
                except KeyboardInterrupt:
                    print("\n\n⏹️ Interrupted by user")
                    break
                except Exception as e:
                    print(f"❌ Error: {e}")
                    self.logger.error(f"Main loop error: {e}")
                    continue
            
            # Shutdown
            self.shutdown()
            return True
            
        except Exception as e:
            print(f"💥 Fatal error: {e}")
            traceback.print_exc()
            return False
    
    def shutdown(self):
        """Clean shutdown"""
        print("\n" + "="*70)
        print("🛑 Shutting down Jarvis...")
        print("="*70)
        
        # Save all data
        print("📝 Saving data...")
        if self.commands:
            self.commands.save_data()
        
        if self.db:
            self.db.remember("last_shutdown", str(datetime.datetime.now()))
        
        self.save_preferences()
        
        # Stop background threads
        self.running = False
        
        # Clear resources
        if self.speech:
            print("🔊 Stopping speech...")
            self.speech.cleanup()
        
        if self.logger:
            self.logger.info("Jarvis shutdown complete")
        
        # Show summary
        stats = self.performance.get_stats()
        print(f"\n📊 Session Summary:")
        print(f"  Uptime: {stats['uptime']/3600:.1f} hours")
        print(f"  Commands: {stats['total_commands']}")
        print(f"  Avg response: {stats['avg_response_time']*1000:.1f}ms")
        
        print("\n✅ Jarvis has been shut down. Goodbye! 👋")
        
        # Speak goodbye
        goodbyes = [
            "Goodbye! Have a wonderful day!",
            "Until next time! Take care!",
            "Signing off! It was great helping you!",
            "Farewell, my friend!"
        ]
        self.speech.speak(random.choice(goodbyes))


def main():
    """Main entry point - RUN THIS!"""
    print("🚀 Starting Jarvis AI Assistant...")
    print("Loading enhanced features...")
    
    try:
        # Create and run assistant
        assistant = JarvisAssistant()
        result = assistant.run()
        
        if result:
            print("\n✨ Jarvis session completed successfully")
        else:
            print("\n⚠️ Jarvis session ended with issues")
            
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye! Thanks for using Jarvis!")
    except Exception as e:
        print(f"💥 Application failed: {e}")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__"
