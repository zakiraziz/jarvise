import requests
import pyautogui
import schedule
import time
import threading
from abc import ABC, abstractmethod

class Command(ABC):
    def __init__(self, name):
        self.name = name

    def parse_params(self, params_str):
        # Simple parsing: assume params_str is "key1=value1,key2=value2"
        params = {}
        if params_str:
            for pair in params_str.split(','):
                if '=' in pair:
                    key, value = pair.split('=', 1)
                    params[key.strip()] = value.strip()
        return params

    @abstractmethod
    def execute(self, params):
        pass

    def handle_results(self, results):
        # Default handling: print results
        print(f"Results from {self.name}: {results}")
        return results

class WebSearchCommand(Command):
    def __init__(self):
        super().__init__("web_search")

    def execute(self, params):
        query = params.get('query', '')
        if not query:
            return "No query provided"
        # Using a simple search API, e.g., DuckDuckGo instant answer API
        url = f"https://api.duckduckgo.com/?q={query}&format=json"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data.get('AbstractText', 'No abstract found')
        else:
            return "Search failed"

class AutomationCommand(Command):
    def __init__(self):
        super().__init__("automation")

    def execute(self, params):
        action = params.get('action', '')
        if action == 'click':
            x = int(params.get('x', 0))
            y = int(params.get('y', 0))
            pyautogui.click(x, y)
            return "Clicked at ({}, {})".format(x, y)
        elif action == 'type':
            text = params.get('text', '')
            pyautogui.typewrite(text)
            return f"Typed: {text}"
        else:
            return "Unknown action"

class SmartHomeCommand(Command):
    def __init__(self, api_url, api_key):
        super().__init__("smart_home")
        self.api_url = api_url
        self.api_key = api_key

    def execute(self, params):
        device = params.get('device', '')
        action = params.get('action', '')
        headers = {'Authorization': f'Bearer {self.api_key}'}
        data = {'device': device, 'action': action}
        response = requests.post(self.api_url, json=data, headers=headers)
        if response.status_code == 200:
            return f"Smart home action '{action}' on '{device}' executed"
        else:
            return "Smart home command failed"

class ScheduleCommand(Command):
    def __init__(self, command_registry):
        super().__init__("schedule")
        self.command_registry = command_registry
        self.scheduler_thread = None

    def execute(self, params):
        command_name = params.get('command', '')
        time_str = params.get('time', '')  # e.g., "10:00"
        params_str = params.get('params', '')
        if not command_name or not time_str:
            return "Command name and time required"
        # Schedule the command
        schedule.every().day.at(time_str).do(self._run_scheduled_command, command_name, params_str)
        if not self.scheduler_thread or not self.scheduler_thread.is_alive():
            self.scheduler_thread = threading.Thread(target=self._run_scheduler)
            self.scheduler_thread.start()
        return f"Scheduled {command_name} at {time_str}"

    def _run_scheduled_command(self, command_name, params_str):
        self.command_registry.execute(command_name, params_str)

    def _run_scheduler(self):
        while True:
            schedule.run_pending()
            time.sleep(1)

class CommandRegistry:
    def __init__(self):
        self.commands = {}

    def register(self, command):
        self.commands[command.name] = command

    def execute(self, command_name, params_str):
        if command_name in self.commands:
            command = self.commands[command_name]
            params = command.parse_params(params_str)
            results = command.execute(params)
            return command.handle_results(results)
        else:
            return f"Command '{command_name}' not found"