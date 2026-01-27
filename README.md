# 🤖 AI Coding Assistant

An intelligent coding assistant that automatically generates complete, working solutions from natural language problem descriptions. Perfect for debugging, learning, and rapid prototyping.

## ✨ Features

- **Natural Language Processing**: Understands complex coding problems described in plain English
- **Multi-Language Support**: Python, JavaScript, Java, C++, Go, Rust, and more
- **Intelligent Analysis**: Identifies error types, libraries, and constraints automatically
- **Complete Solutions**: Generates production-ready code with explanations
- **Safety First**: Built-in security checks prevent malicious code generation
- **Interactive Sessions**: Follow-up questions and conversation history
- **Best Practices**: Includes coding standards and prevention tips
- **Rich CLI Interface**: Beautiful, user-friendly terminal experience

## 🚀 Quick Start

### Installation

1. **Clone and setup:**
   ```bash
   git clone <repository>
   cd code_assistant
   pip install -r requirements.txt
   ```

2. **Configure OpenAI API:**
   ```bash
   # Edit config/config.yaml
   openai:
     api_key: "your_openai_api_key_here"
   ```

3. **Run the assistant:**
   ```bash
   python src/main.py
   ```

## 💡 Usage Examples

### Example 1: SSL Certificate Error
```
Describe your coding problem: I'm trying to create a Python script that scrapes data from a website but getting SSL certificate errors
```

**Output:**
- Detects: Python, requests library, SSL error
- Generates: Complete solution with certificate handling
- Includes: Error handling, best practices, prevention tips

### Example 2: React State Update Issue
```
Describe your coding problem: My React component isn't updating when state changes, even though I'm calling setState correctly
```

**Output:**
- Detects: JavaScript, React, state management issue
- Generates: Corrected component with proper state handling
- Includes: useEffect explanations, immutability tips

### Example 3: Database Connection Problem
```
Describe your coding problem: My Node.js app can't connect to MongoDB, getting connection timeout errors
```

**Output:**
- Detects: JavaScript, Node.js, MongoDB, connection issue
- Generates: Proper connection setup with error handling
- Includes: Connection pooling, retry logic, security considerations

## 🔧 Commands

- **Describe problems naturally** - Just type your coding issue
- `new` - Start a fresh conversation
- `history` - View past conversations
- `load <id>` - Resume a previous session
- `quit` - Exit the assistant

## 🏗️ Architecture

```
code_assistant/
├── src/
│   ├── main.py              # CLI interface with rich formatting
│   ├── problem_parser.py    # NLP analysis engine
│   ├── code_generator.py    # OpenAI-powered solution generation
│   ├── safety_checker.py    # Security and safety validation
│   ├── conversation_manager.py # Session and history management
│   └── __init__.py
├── config/
│   └── config.yaml          # Configuration settings
├── conversations/           # Saved conversation history
├── requirements.txt         # Python dependencies
└── README.md               # This documentation
```

## 🔒 Safety Features

- **Blocked Keywords**: Prevents generation of malicious code
- **Pattern Detection**: Identifies dangerous operations
- **Language-Specific Checks**: Validates code safety per language
- **Security Warnings**: Clear alerts for potentially unsafe code
- **Sanitization**: Attempts to clean up risky code patterns

## 🎯 Problem Types Handled

- **Syntax Errors**: Missing brackets, typos, incorrect syntax
- **Logic Errors**: Wrong algorithms, incorrect conditions
- **Dependency Issues**: Missing imports, version conflicts
- **Runtime Errors**: Exceptions, crashes, memory issues
- **Configuration Problems**: Environment setup, API keys
- **Architecture Issues**: Design patterns, code structure
- **Security Vulnerabilities**: Injection, authentication problems

## 💬 Interactive Features

- **Clarification Requests**: Asks for missing information
- **Follow-up Support**: Answers questions about generated solutions
- **Context Preservation**: Maintains conversation history
- **Progressive Refinement**: Improves solutions based on feedback

## 📊 Configuration

Customize behavior in `config/config.yaml`:

```yaml
openai:
  api_key: "your_key"
  model: "gpt-4-turbo-preview"
  temperature: 0.1

assistant:
  max_conversation_history: 50

safety:
  blocked_keywords:
    - "malware"
    - "virus"
```

## 🧪 Testing

Run the test suite:
```bash
python -m pytest tests/
```

## 📝 API Usage

The assistant can also be used programmatically:

```python
from code_assistant.src.problem_parser import ProblemParser
from code_assistant.src.code_generator import CodeGenerator

parser = ProblemParser()
generator = CodeGenerator(api_key="your_key")

analysis = parser.parse_problem("My Python script has SSL errors")
solution = generator.generate_solution("My Python script has SSL errors", analysis)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## ⚠️ Disclaimer

- **Security**: Always review generated code before execution
- **Testing**: Test solutions in isolated environments first
- **Production**: Generated code may need adaptation for production use
- **Legal**: Ensure compliance with OpenAI's terms of service

## 🆘 Troubleshooting

**Common Issues:**

- **API Key Errors**: Verify your OpenAI API key is correct
- **Network Issues**: Check internet connection for API calls
- **Permission Errors**: Ensure write access to conversations directory
- **Import Errors**: Install all requirements with `pip install -r requirements.txt`

**Getting Help:**

- Check conversation logs in `conversations/` directory
- Review the analysis output for detection accuracy
- Try rephrasing complex problems more clearly

---

**Made with ❤️ for developers who want to code faster and learn better.**
