"""
Problem Parser Module
Analyzes natural language problem descriptions to extract key information.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

logger = logging.getLogger(__name__)


class ComplexityLevel(Enum):
    """Complexity levels for coding problems."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class ProblemDomain(Enum):
    """Problem domains for classification."""
    WEB_DEVELOPMENT = "web_development"
    DATA_SCIENCE = "data_science"
    MACHINE_LEARNING = "machine_learning"
    DEVOPS = "devops"
    MOBILE_DEVELOPMENT = "mobile_development"
    DESKTOP_APPLICATION = "desktop_application"
    GAME_DEVELOPMENT = "game_development"
    EMBEDDED_SYSTEMS = "embedded_systems"
    DATABASE = "database"
    SECURITY = "security"
    ALGORITHMS = "algorithms"
    TESTING = "testing"
    OTHER = "other"


@dataclass
class ProblemAnalysis:
    """Structured analysis of a coding problem."""
    language: Optional[str] = None
    libraries: List[str] = None
    error_messages: List[str] = None
    functionality: Optional[str] = None
    constraints: List[str] = None
    error_type: Optional[str] = None  # syntax, logic, dependency, architecture
    confidence: float = 0.0
    needs_clarification: bool = False
    clarification_questions: List[str] = None
    
    # New fields for enhanced analysis
    code_snippets: List[str] = None
    problem_domain: Optional[str] = None
    complexity: Optional[str] = None
    keywords: List[str] = None
    suggested_tags: List[str] = None
    estimated_time: Optional[int] = None  # in minutes
    similar_problems: List[Dict[str, Any]] = None
    prerequisites: List[str] = None
    expected_output: Optional[str] = None
    input_format: Optional[str] = None
    output_format: Optional[str] = None
    test_cases: List[Dict[str, str]] = None
    performance_requirements: List[str] = None
    security_requirements: List[str] = None
    version_info: Dict[str, str] = None
    operating_system: Optional[str] = None
    ide_info: Optional[str] = None
    timestamp: str = None

    def __post_init__(self):
        if self.libraries is None:
            self.libraries = []
        if self.error_messages is None:
            self.error_messages = []
        if self.constraints is None:
            self.constraints = []
        if self.clarification_questions is None:
            self.clarification_questions = []
        if self.code_snippets is None:
            self.code_snippets = []
        if self.keywords is None:
            self.keywords = []
        if self.suggested_tags is None:
            self.suggested_tags = []
        if self.similar_problems is None:
            self.similar_problems = []
        if self.prerequisites is None:
            self.prerequisites = []
        if self.test_cases is None:
            self.test_cases = []
        if self.performance_requirements is None:
            self.performance_requirements = []
        if self.security_requirements is None:
            self.security_requirements = []
        if self.version_info is None:
            self.version_info = {}
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis to dictionary."""
        return {
            'language': self.language,
            'libraries': self.libraries,
            'error_messages': self.error_messages,
            'functionality': self.functionality,
            'constraints': self.constraints,
            'error_type': self.error_type,
            'confidence': self.confidence,
            'needs_clarification': self.needs_clarification,
            'clarification_questions': self.clarification_questions,
            'code_snippets': self.code_snippets,
            'problem_domain': self.problem_domain,
            'complexity': self.complexity,
            'keywords': self.keywords,
            'suggested_tags': self.suggested_tags,
            'estimated_time': self.estimated_time,
            'prerequisites': self.prerequisites,
            'expected_output': self.expected_output,
            'input_format': self.input_format,
            'output_format': self.output_format,
            'test_cases': self.test_cases,
            'performance_requirements': self.performance_requirements,
            'security_requirements': self.security_requirements,
            'version_info': self.version_info,
            'operating_system': self.operating_system,
            'ide_info': self.ide_info,
            'timestamp': self.timestamp
        }

    def to_json(self) -> str:
        """Convert analysis to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class ProblemParser:
    """Parses natural language coding problem descriptions."""

    def __init__(self):
        # Language detection patterns (enhanced)
        self.language_patterns = {
            'python': re.compile(r'\b(python|py|django|flask|fastapi|pandas|numpy|tensorflow|keras|scikit|pytorch|jupyter|anaconda|pip|virtualenv)\b', re.IGNORECASE),
            'javascript': re.compile(r'\b(javascript|js|node|react|vue|angular|express|jquery|typescript|ts|npm|yarn|webpack|babel|es6)\b', re.IGNORECASE),
            'java': re.compile(r'\b(java|spring|maven|gradle|hibernate|android|jvm|jdk|jre|eclipse|intellij)\b', re.IGNORECASE),
            'cpp': re.compile(r'\b(c\+\+|cpp|cpp11|cpp14|cpp17|qt|opencv|boost|stl|cmake|makefile)\b', re.IGNORECASE),
            'c': re.compile(r'\b(c\b|glibc|posix|stdio|stdlib|gcc|clang)\b', re.IGNORECASE),
            'go': re.compile(r'\b(go|golang|gin|echo|goroutine|channel|gopath)\b', re.IGNORECASE),
            'rust': re.compile(r'\b(rust|cargo|tokio|actix|rocket|rustc|clippy)\b', re.IGNORECASE),
            'html': re.compile(r'\b(html|css|dom|jquery|bootstrap|tailwind|sass|less)\b', re.IGNORECASE),
            'sql': re.compile(r'\b(sql|mysql|postgresql|sqlite|oracle|mongodb|nosql|query|database|db)\b', re.IGNORECASE),
            'csharp': re.compile(r'\b(c#|csharp|dotnet|.net|asp|visual studio|mono|unity)\b', re.IGNORECASE),
            'php': re.compile(r'\b(php|lumen|laravel|symfony|composer|wordpress|drupal)\b', re.IGNORECASE),
            'ruby': re.compile(r'\b(ruby|rails|gem|bundler|sinatra|rack)\b', re.IGNORECASE),
            'swift': re.compile(r'\b(swift|ios|macos|cocoa|xcode|uikit|swiftui)\b', re.IGNORECASE),
            'kotlin': re.compile(r'\b(kotlin|android|kts|gradle|ktor|anko)\b', re.IGNORECASE),
            'scala': re.compile(r'\b(scala|sbt|play|akka|spark|intellij)\b', re.IGNORECASE),
            'r': re.compile(r'\b(r\b|rstudio|tidyverse|ggplot2|dplyr|shiny|cran)\b', re.IGNORECASE),
        }

        # Error type patterns (enhanced)
        self.error_patterns = {
            'syntax': re.compile(r'\b(syntax error|parse error|invalid syntax|unexpected token|compilation error|missing parenthesis|missing bracket|indentation error|unexpected character)\b', re.IGNORECASE),
            'logic': re.compile(r'\b(logic error|wrong output|incorrect result|bug|not working|infinite loop|off-by-one|algorithm bug|calculation error)\b', re.IGNORECASE),
            'dependency': re.compile(r'\b(import error|module not found|dependency|package|pip|npm|install|missing dependency|version conflict|class not found|no module named)\b', re.IGNORECASE),
            'runtime': re.compile(r'\b(runtime error|exception|crash|segmentation fault|null pointer|index out of range|key error|attribute error|type error|value error|zerodivision|memory error|stack overflow)\b', re.IGNORECASE),
            'ssl': re.compile(r'\b(ssl|certificate|tls|https|connection refused|timeout|verification failed|ssl error)\b', re.IGNORECASE),
            'performance': re.compile(r'\b(slow|memory leak|high cpu|performance|optimization|lag|bottleneck|efficient|timeout|latency|throughput)\b', re.IGNORECASE),
            'security': re.compile(r'\b(security|vulnerability|xss|sql injection|csrf|authentication|authorization|encryption|hashing|password|jwt|oauth|sanitization)\b', re.IGNORECASE),
            'concurrency': re.compile(r'\b(concurrency|race condition|deadlock|thread-safe|parallel|async|await|synchronization|mutex|semaphore|goroutine|coroutine)\b', re.IGNORECASE),
            'network': re.compile(r'\b(network|connection|socket|http|tcp/ip|dns|proxy|firewall|port|bind|listen|client|server|request|response)\b', re.IGNORECASE),
        }

        # Problem domain patterns
        self.domain_patterns = {
            'web_development': re.compile(r'\b(web|website|html|css|http|rest|api|endpoint|frontend|backend|fullstack|server|client|browser|url|route|middleware)\b', re.IGNORECASE),
            'data_science': re.compile(r'\b(data|analysis|statistics|visualization|plot|chart|dashboard|eda|pandas|dataframe|dataset|cleaning|preprocessing)\b', re.IGNORECASE),
            'machine_learning': re.compile(r'\b(machine learning|ml|ai|artificial intelligence|neural network|deep learning|tensorflow|pytorch|training|model|prediction|classification|regression|clustering|nlp)\b', re.IGNORECASE),
            'devops': re.compile(r'\b(devops|deployment|ci/cd|pipeline|docker|kubernetes|aws|azure|gcp|cloud|terraform|ansible|jenkins|monitoring|logging)\b', re.IGNORECASE),
            'mobile_development': re.compile(r'\b(mobile|android|ios|app|smartphone|tablet|react native|flutter|swift|kotlin|xcode)\b', re.IGNORECASE),
            'database': re.compile(r'\b(database|db|sql|query|table|index|join|migration|schema|orm|mongodb|postgresql|mysql|redis|cassandra)\b', re.IGNORECASE),
            'algorithms': re.compile(r'\b(algorithm|sort|search|tree|graph|dynamic programming|recursion|complexity|big o|time complexity|space complexity)\b', re.IGNORECASE),
            'game_development': re.compile(r'\b(game|unity|unreal|godot|sprite|physics|collision|3d|2d|animation|rendering|shader)\b', re.IGNORECASE),
            'security': re.compile(r'\b(security|encryption|decryption|hash|authentication|authorization|cybersecurity|penetration|vulnerability|exploit)\b', re.IGNORECASE),
            'embedded': re.compile(r'\b(embedded|arduino|raspberry pi|firmware|microcontroller|sensor|iot|real-time|hardware|gpio)\b', re.IGNORECASE),
        }

        # Complexity indicators
        self.complexity_patterns = {
            'beginner': re.compile(r'\b(beginner|simple|basic|easy|new to|just started|tutorial|introductory|first time)\b', re.IGNORECASE),
            'intermediate': re.compile(r'\b(intermediate|some experience|familiar with|basic understanding)\b', re.IGNORECASE),
            'advanced': re.compile(r'\b(advanced|complex|complicated|challenging|production|enterprise|scalable|optimized|efficient)\b', re.IGNORECASE),
            'expert': re.compile(r'\b(expert|expert-level|research|novel|cutting-edge|state-of-the-art|high-performance|distributed)\b', re.IGNORECASE),
        }

        # Common libraries/frameworks (enhanced)
        self.library_keywords = {
            'python': ['requests', 'beautifulsoup', 'selenium', 'pandas', 'numpy', 'matplotlib', 'django', 'flask', 'fastapi', 'scipy', 'scikit-learn', 'tensorflow', 'pytorch', 'jupyter', 'asyncio', 'sqlalchemy', 'pytest', 'unittest', 'celery', 'redis', 'psycopg2'],
            'javascript': ['react', 'vue', 'angular', 'express', 'jquery', 'axios', 'lodash', 'moment', 'next.js', 'nuxt.js', 'gatsby', 'webpack', 'babel', 'jest', 'mocha', 'chai', 'redux', 'node-fetch', 'socket.io'],
            'java': ['spring', 'hibernate', 'maven', 'gradle', 'junit', 'log4j', 'slf4j', 'tomcat', 'jetty', 'guava', 'apache commons', 'mockito', 'lombok', 'jackson', 'jdbc'],
            'cpp': ['boost', 'qt', 'opencv', 'poco', 'eigen', 'openmp', 'mpi', 'cuda', 'sfml', 'sdl', 'glfw', 'vulkan', 'webview'],
            'csharp': ['entity framework', 'asp.net', 'xamarin', 'unity', 'linq', 'nunit', 'moq', 'serilog', 'automapper', 'signalr'],
            'ruby': ['rails', 'rspec', 'capybara', 'sidekiq', 'devise', 'pundit', 'active record', 'slim', 'haml', 'grape'],
            'go': ['gorilla', 'chi', 'viper', 'cobra', 'zap', 'gorm', 'fiber', 'echo', 'gin', 'ent'],
            'rust': ['tokio', 'actix', 'rocket', 'serde', 'diesel', 'rayon', 'clap', 'reqwest', 'async-std', 'wasm-bindgen'],
        }

        # Version patterns
        self.version_patterns = [
            re.compile(r'(python|node|npm|java|django|flask|react|angular|vue|spring|dotnet)[\s:=]*(>=|<=|==|~=)?\s*(\d+\.?\d*\.?\d*)', re.IGNORECASE),
            re.compile(r'version[\s:=]+(\d+\.?\d*\.?\d*)', re.IGNORECASE),
            re.compile(r'v(\d+\.?\d*\.?\d*)', re.IGNORECASE),
        ]

        # OS patterns
        self.os_patterns = {
            'windows': re.compile(r'\b(windows|win10|win11|win32|win64|microsoft windows)\b', re.IGNORECASE),
            'linux': re.compile(r'\b(linux|ubuntu|debian|centos|fedora|redhat|rhel|arch|unix|posix|wsl|wsl2)\b', re.IGNORECASE),
            'macos': re.compile(r'\b(mac|macos|osx|darwin|apple|iphone|ipad)\b', re.IGNORECASE),
        }

        # IDE patterns
        self.ide_patterns = re.compile(r'\b(vscode|visual studio code|pycharm|intellij|eclipse|netbeans|xcode|android studio|sublime|vim|emacs|atom|notepad\+\+|ide|text editor)\b', re.IGNORECASE)

        # Input/Output patterns
        self.io_patterns = {
            'input': re.compile(r'\b(input|stdin|console input|user input|parameter|argument|read from|accepts?)\b', re.IGNORECASE),
            'output': re.compile(r'\b(output|stdout|print|display|show|return|result|generate|produce)\b', re.IGNORECASE),
        }

        # Performance requirement patterns
        self.performance_patterns = {
            'time': re.compile(r'\b(\d+)\s*(ms|milliseconds?|seconds?|minutes?)\s*(time|runtime|execution|speed|fast|quick|faster|optimize|optimisation)\b', re.IGNORECASE),
            'memory': re.compile(r'\b(\d+)\s*(kb|mb|gb|kilobytes?|megabytes?|gigabytes?)\s*(memory|ram|space|storage|heap)\b', re.IGNORECASE),
            'complexity': re.compile(r'\b(o\([^)]+\)|big o|time complexity|space complexity|n log n|exponential|quadratic|linear|logarithmic)\b', re.IGNORECASE),
        }

    def parse_problem(self, description: str) -> ProblemAnalysis:
        """
        Parse a natural language problem description.

        Args:
            description: The problem description text

        Returns:
            ProblemAnalysis: Structured analysis of the problem
        """
        analysis = ProblemAnalysis()

        # Core parsing
        analysis.language = self._detect_language(description)
        analysis.libraries = self._extract_libraries(description, analysis.language)
        analysis.error_messages = self._extract_errors(description)
        analysis.error_type = self._classify_error(description)
        analysis.functionality = self._extract_functionality(description)
        analysis.constraints = self._extract_constraints(description)

        # Enhanced parsing
        analysis.code_snippets = self._extract_code_snippets(description)
        analysis.problem_domain = self._detect_problem_domain(description)
        analysis.complexity = self._estimate_complexity(description)
        analysis.keywords = self._extract_keywords(description)
        analysis.suggested_tags = self._generate_suggested_tags(analysis)
        analysis.estimated_time = self._estimate_solution_time(analysis)
        analysis.prerequisites = self._extract_prerequisites(description)
        analysis.expected_output, analysis.input_format, analysis.output_format = self._extract_io_formats(description)
        analysis.test_cases = self._extract_test_cases(description)
        analysis.performance_requirements = self._extract_performance_requirements(description)
        analysis.security_requirements = self._extract_security_requirements(description)
        analysis.version_info = self._extract_versions(description)
        analysis.operating_system = self._detect_os(description)
        analysis.ide_info = self._detect_ide(description)

        # Clarification needs and confidence
        analysis.needs_clarification, analysis.clarification_questions = self._check_clarification_needed(description, analysis)
        analysis.confidence = self._calculate_confidence(analysis)

        logger.info(f"Parsed problem: language={analysis.language}, domain={analysis.problem_domain}, "
                   f"error_type={analysis.error_type}, complexity={analysis.complexity}, "
                   f"confidence={analysis.confidence:.2f}")

        return analysis

    def _detect_language(self, text: str) -> Optional[str]:
        """Detect the programming language mentioned in the text."""
        scores = {}

        for lang, pattern in self.language_patterns.items():
            matches = len(pattern.findall(text))
            if matches > 0:
                scores[lang] = matches

        if scores:
            return max(scores, key=scores.get)

        # Fallback: check for common file extensions or keywords
        ext_patterns = {
            'python': r'\.py\b',
            'javascript': r'\.js\b|\.jsx\b|\.ts\b|\.tsx\b',
            'java': r'\.java\b',
            'cpp': r'\.cpp|\.cc|\.cxx|\.hpp|\.hh',
            'c': r'\.c\b|\.h\b',
            'go': r'\.go\b',
            'rust': r'\.rs\b',
            'html': r'\.html?|\.css|\.scss|\.sass|\.less',
            'csharp': r'\.cs\b',
            'php': r'\.php\b',
            'ruby': r'\.rb\b',
            'swift': r'\.swift\b',
            'kotlin': r'\.kt\b|\.kts\b',
            'scala': r'\.scala\b',
            'r': r'\.r\b|\.rds\b',
        }

        for lang, pattern in ext_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                return lang

        return None

    def _extract_libraries(self, text: str, language: Optional[str]) -> List[str]:
        """Extract mentioned libraries and frameworks."""
        libraries = []

        # Language-specific libraries
        if language and language in self.library_keywords:
            for lib in self.library_keywords[language]:
                if lib.lower() in text.lower():
                    libraries.append(lib)

        # General library extraction with context
        lib_patterns = [
            (r'\b(using|with|import|from|require|include|use|add)\s+([a-zA-Z_][a-zA-Z0-9_\.\-]*)\b', re.IGNORECASE),
            (r'pip\s+install\s+([a-zA-Z_][a-zA-Z0-9_\-\.]*)', re.IGNORECASE),
            (r'npm\s+install\s+([a-zA-Z_][a-zA-Z0-9_\-\.]*)', re.IGNORECASE),
            (r'gem\s+install\s+([a-zA-Z_][a-zA-Z0-9_\-\.]*)', re.IGNORECASE),
            (r'cargo\s+install\s+([a-zA-Z_][a-zA-Z0-9_\-\.]*)', re.IGNORECASE),
            (r'go\s+get\s+([a-zA-Z_][a-zA-Z0-9_\-\./]*)', re.IGNORECASE),
        ]

        for pattern, flags in lib_patterns:
            matches = re.findall(pattern, text, flags)
            for match in matches:
                if isinstance(match, tuple):
                    lib = match[-1]  # Get the last element which should be the library name
                else:
                    lib = match
                lib = lib.strip().lower()
                if lib not in ['i', 'a', 'the', 'this', 'that', 'and', 'or', 'with'] and len(lib) > 2:
                    if '.' not in lib or lib.split('.')[-1] in ['py', 'js', 'jar', 'rb', 'so', 'dll']:
                        libraries.append(lib)

        return list(set(libraries))

    def _extract_code_snippets(self, text: str) -> List[str]:
        """Extract code snippets from the description."""
        snippets = []

        # Markdown code blocks
        code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', text, re.DOTALL)
        snippets.extend([block.strip() for block in code_blocks if block.strip()])

        # Inline code with backticks
        inline_code = re.findall(r'`([^`]+)`', text)
        snippets.extend([code.strip() for code in inline_code if len(code.strip()) > 20])

        # Indented code blocks (lines starting with 4+ spaces)
        lines = text.split('\n')
        current_block = []
        for line in lines:
            if line.startswith('    ') or line.startswith('\t'):
                current_block.append(line.lstrip())
            elif current_block:
                snippets.append('\n'.join(current_block))
                current_block = []
        if current_block:
            snippets.append('\n'.join(current_block))

        return list(set(snippets))

    def _extract_errors(self, text: str) -> List[str]:
        """Extract error messages from the description."""
        errors = []

        # Look for quoted error messages
        quoted_errors = re.findall(r'["\']([^"\']*error[^"\']*)["\']', text, re.IGNORECASE)
        errors.extend(quoted_errors)

        # Look for error patterns
        error_sentences = re.findall(r'[^.!?]*(?:error|exception|crash|fail|bug)[^.!?]*[.!?]', text, re.IGNORECASE)
        errors.extend([s.strip() for s in error_sentences if len(s.strip()) > 10])
        # Extract stack traces
        stack_trace_patterns = [
            r'(?:Traceback.*?\n)(.*?)(?=\n\n|\Z)',
            r'(?:at\s+[\w\.$]+\(.*?\).*?\n)+',
            r'(?:[\w\.]+:\d+:in\s+.*?\n)+',
        ]

        for pattern in stack_trace_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            errors.extend([match.strip() for match in matches if match.strip()])

        return list(set(errors))

    def _classify_error(self, text: str) -> Optional[str]:
        """Classify the type of error."""
        scores = {}

        for error_type, pattern in self.error_patterns.items():
            matches = pattern.findall(text)
            if matches:
                scores[error_type] = scores.get(error_type, 0) + len(matches)

        if scores:
            return max(scores, key=scores.get)

        return None

    def _extract_functionality(self, text: str) -> Optional[str]:
        """Extract what the code is supposed to do."""
        # Look for sentences describing functionality
        sentences = re.split(r'[.!?]+', text)

        functionality_indicators = [
            'trying to', 'want to', 'need to', 'should', 'must', 'will',
            'create', 'build', 'make', 'implement', 'develop', 'design',
            'calculate', 'compute', 'process', 'handle', 'manage',
            'display', 'show', 'render', 'generate', 'produce',
            'read', 'write', 'parse', 'convert', 'transform',
            'connect', 'fetch', 'download', 'upload', 'send', 'receive',
            'authenticate', 'validate', 'check', 'verify', 'test',
        ]

        for sentence in sentences:
            sentence = sentence.strip()
            if any(indicator in sentence.lower() for indicator in functionality_indicators):
                # Remove common problem indicators to get cleaner description
                cleaned = re.sub(r'(i am|im|i\'m|i have|we have|the code|my code|it should|it will)', '', sentence, flags=re.IGNORECASE)
                return cleaned.strip()

        return None

    def _extract_constraints(self, text: str) -> List[str]:
        """Extract constraints and requirements."""
        constraints = []

        # Version constraints
        version_matches = self._extract_versions(text)
        for lib, version in version_matches.items():
            if lib:
                constraints.append(f"{lib} version: {version}")
            else:
                constraints.append(f"Version: {version}")

        # Platform constraints
        os_detected = self._detect_os(text)
        if os_detected:
            constraints.append(f"Platform: {os_detected}")

        # Time constraints
        time_constraints = re.findall(r'\b(within|in|under|less than)\s+(\d+)\s*(ms|milliseconds?|seconds?|minutes?|hours?)\b', text, re.IGNORECASE)
        for constraint in time_constraints:
            constraints.append(f"Time constraint: {constraint[1]} {constraint[2]}")

        # Memory constraints
        memory_constraints = re.findall(r'\b(within|under|less than)\s+(\d+)\s*(kb|mb|gb)\s+(memory|ram)\b', text, re.IGNORECASE)
        for constraint in memory_constraints:
            constraints.append(f"Memory constraint: {constraint[1]} {constraint[2]}")

        # Other common constraints
        if 'concurrent' in text.lower() or 'parallel' in text.lower():
            constraints.append('Concurrency/parallelism required')
        if 'realtime' in text.lower() or 'real-time' in text.lower():
            constraints.append('Real-time processing required')
        if 'cross-platform' in text.lower():
            constraints.append('Cross-platform compatibility required')
        if 'backward compatible' in text.lower() or 'backwards compatible' in text.lower():
            constraints.append('Backward compatibility required')

        return list(set(constraints))

    def _detect_problem_domain(self, text: str) -> Optional[str]:
        """Detect the problem domain."""
        scores = {}

        for domain, pattern in self.domain_patterns.items():
            matches = len(pattern.findall(text))
            if matches > 0:
                scores[domain] = matches

        if scores:
            return max(scores, key=scores.get)

        return ProblemDomain.OTHER.value

    def _estimate_complexity(self, text: str) -> Optional[str]:
        """Estimate the complexity level of the problem."""
        scores = {}

        for level, pattern in self.complexity_patterns.items():
            matches = len(pattern.findall(text))
            if matches > 0:
                scores[level] = matches

        # Check for complexity indicators in code snippets
        code_snippets = self._extract_code_snippets(text)
        for snippet in code_snippets:
            if len(snippet.split('\n')) > 50:
                scores['advanced'] = scores.get('advanced', 0) + 1
            if 'class' in snippet and 'def' in snippet and len(snippet) > 1000:
                scores['expert'] = scores.get('expert', 0) + 1

        if scores:
            return max(scores, key=scores.get)

        return ComplexityLevel.BEGINNER.value

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from the description."""
        keywords = []

        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'with', 'by', 'from', 'as', 'of', 'it', 'this', 'that', 'is', 'are',
                     'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had'}

        # Extract words with specific patterns
        word_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Proper nouns
            r'\b[a-z]{4,}\b',  # Words with 4+ letters
            r'\b[A-Z]{2,}\b',  # Acronyms
        ]

        for pattern in word_patterns:
            matches = re.findall(pattern, text)
            keywords.extend([match.lower() for match in matches if match.lower() not in stop_words])

        # Remove duplicates and sort
        keywords = list(set(keywords))
        keywords.sort(key=lambda x: len(x), reverse=True)

        # Limit to top 20 keywords
        return keywords[:20]

    def _generate_suggested_tags(self, analysis: ProblemAnalysis) -> List[str]:
        """Generate suggested tags for the problem."""
        tags = []

        if analysis.language:
            tags.append(analysis.language)

        if analysis.error_type:
            tags.append(f"error:{analysis.error_type}")

        if analysis.problem_domain:
            tags.append(analysis.problem_domain)

        if analysis.complexity:
            tags.append(f"complexity:{analysis.complexity}")

        # Add library tags
        for lib in analysis.libraries[:5]:  # Limit to top 5 libraries
            tags.append(lib)

        # Add common problem type tags
        if analysis.functionality:
            functionality_lower = analysis.functionality.lower()
            if any(word in functionality_lower for word in ['api', 'rest', 'endpoint']):
                tags.append('api-development')
            if any(word in functionality_lower for word in ['database', 'sql', 'query']):
                tags.append('database')
            if any(word in functionality_lower for word in ['test', 'unit test', 'testing']):
                tags.append('testing')

        return list(set(tags))

    def _estimate_solution_time(self, analysis: ProblemAnalysis) -> Optional[int]:
        """Estimate solution time in minutes."""
        base_time = 30  # Base time in minutes

        # Adjust based on complexity
        complexity_multipliers = {
            ComplexityLevel.BEGINNER.value: 0.5,
            ComplexityLevel.INTERMEDIATE.value: 1.0,
            ComplexityLevel.ADVANCED.value: 2.0,
            ComplexityLevel.EXPERT.value: 3.0,
        }

        multiplier = complexity_multipliers.get(analysis.complexity, 1.0)
        estimated_time = int(base_time * multiplier)

        # Adjust based on number of libraries
        if analysis.libraries:
            estimated_time += len(analysis.libraries) * 10

        # Adjust based on constraints
        if analysis.constraints:
            estimated_time += len(analysis.constraints) * 5


