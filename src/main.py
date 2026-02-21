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
    
