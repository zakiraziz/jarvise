"""
Problem Parser Module
Analyzes natural language problem descriptions to extract key information.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

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

    def __post_init__(self):
        if self.libraries is None:
            self.libraries = []
        if self.error_messages is None:
            self.error_messages = []
        if self.constraints is None:
            self.constraints = []
        if self.clarification_questions is None:
            self.clarification_questions = []

class ProblemParser:
    """Parses natural language coding problem descriptions."""

    def __init__(self):
        # Language detection patterns
        self.language_patterns = {
            'python': re.compile(r'\b(python|py|django|flask|pandas|numpy|tensorflow|keras|scikit|pytorch)\b', re.IGNORECASE),
            'javascript': re.compile(r'\b(javascript|js|node|react|vue|angular|express|jquery|typescript|ts)\b', re.IGNORECASE),
            'java': re.compile(r'\b(java|spring|maven|gradle|hibernate|android)\b', re.IGNORECASE),
            'cpp': re.compile(r'\b(c\+\+|cpp|qt|opencv|boost)\b', re.IGNORECASE),
            'c': re.compile(r'\b(c\b|glibc|posix)', re.IGNORECASE),
            'go': re.compile(r'\b(go|golang|gin|echo)\b', re.IGNORECASE),
            'rust': re.compile(r'\b(rust|cargo|tokio|actix)\b', re.IGNORECASE),
            'html': re.compile(r'\b(html|css|dom|jquery)\b', re.IGNORECASE),
            'sql': re.compile(r'\b(sql|mysql|postgresql|sqlite|oracle|mongodb)\b', re.IGNORECASE),
        }

        # Error type patterns
        self.error_patterns = {
            'syntax': re.compile(r'\b(syntax error|parse error|invalid syntax|unexpected token|compilation error)\b', re.IGNORECASE),
            'logic': re.compile(r'\b(logic error|wrong output|incorrect result|bug|not working|infinite loop)\b', re.IGNORECASE),
            'dependency': re.compile(r'\b(import error|module not found|dependency|package|pip|npm|install)\b', re.IGNORECASE),
            'runtime': re.compile(r'\b(runtime error|exception|crash|segmentation fault|null pointer)\b', re.IGNORECASE),
            'ssl': re.compile(r'\b(ssl|certificate|tls|https|connection|verification)\b', re.IGNORECASE),
        }

        # Common libraries/frameworks
        self.library_keywords = {
            'python': ['requests', 'beautifulsoup', 'selenium', 'pandas', 'numpy', 'matplotlib', 'django', 'flask', 'fastapi'],
            'javascript': ['react', 'vue', 'angular', 'express', 'jquery', 'axios', 'lodash', 'moment'],
            'java': ['spring', 'hibernate', 'maven', 'junit', 'log4j'],
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

        # Detect programming language
        analysis.language = self._detect_language(description)

        # Extract libraries/frameworks
        analysis.libraries = self._extract_libraries(description, analysis.language)

        # Extract error messages
        analysis.error_messages = self._extract_errors(description)

        # Determine error type
        analysis.error_type = self._classify_error(description)

        # Extract functionality requirements
        analysis.functionality = self._extract_functionality(description)

        # Extract constraints
        analysis.constraints = self._extract_constraints(description)

        # Determine if clarification is needed
        analysis.needs_clarification, analysis.clarification_questions = self._check_clarification_needed(description, analysis)

        # Calculate confidence score
        analysis.confidence = self._calculate_confidence(analysis)

        logger.info(f"Parsed problem: language={analysis.language}, error_type={analysis.error_type}, confidence={analysis.confidence:.2f}")

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
        if re.search(r'\.py\b', text):
            return 'python'
        elif re.search(r'\.js\b', text):
            return 'javascript'
        elif re.search(r'\.java\b', text):
            return 'java'
        elif re.search(r'\.cpp|\.cc|\.cxx', text):
            return 'cpp'

        return None

    def _extract_libraries(self, text: str, language: Optional[str]) -> List[str]:
        """Extract mentioned libraries and frameworks."""
        libraries = []

        # Language-specific libraries
        if language and language in self.library_keywords:
            for lib in self.library_keywords[language]:
                if lib.lower() in text.lower():
                    libraries.append(lib)

        # General library extraction
        lib_pattern = re.compile(r'\b(using|with|import|from|require)\s+([a-zA-Z_][a-zA-Z0-9_]*)\b', re.IGNORECASE)
        matches = lib_pattern.findall(text)
        for match in matches:
            lib = match[1].lower()
            if lib not in ['i', 'a', 'the', 'this', 'that'] and len(lib) > 2:
                libraries.append(lib)

        return list(set(libraries))

    def _extract_errors(self, text: str) -> List[str]:
        """Extract error messages from the description."""
        errors = []

        # Look for quoted error messages
        quoted_errors = re.findall(r'["\']([^"\']*error[^"\']*)["\']', text, re.IGNORECASE)
        errors.extend(quoted_errors)

        # Look for error patterns
        error_sentences = re.findall(r'[^.!?]*error[^.!?]*[.!?]', text, re.IGNORECASE)
        errors.extend([s.strip() for s in error_sentences if len(s.strip()) > 10])

        return list(set(errors))

    def _classify_error(self, text: str) -> Optional[str]:
        """Classify the type of error."""
        scores = {}

        for error_type, pattern in self.error_patterns.items():
            if pattern.search(text):
                scores[error_type] = scores.get(error_type, 0) + 1

        if scores:
            return max(scores, key=scores.get)

        return None

    def _extract_functionality(self, text: str) -> Optional[str]:
        """Extract what the code is supposed to do."""
        # Look for sentences describing functionality
        sentences = re.split(r'[.!?]+', text)

        for sentence in sentences:
            sentence = sentence.strip()
            if any(word in sentence.lower() for word in ['trying to', 'want to', 'need to', 'create', 'build', 'make', 'implement']):
                return sentence

        return None

    def _extract_constraints(self, text: str) -> List[str]:
        """Extract constraints and requirements."""
        constraints = []

        # Look for version constraints
        version_matches = re.findall(r'\b(version|v)?\s*[\d.]+\b', text, re.IGNORECASE)
        constraints.extend(version_matches)

        # Look for platform constraints
        if 'windows' in text.lower():
            constraints.append('Windows platform')
        if 'linux' in text.lower():
            constraints.append('Linux platform')
        if 'mac' in text.lower() or 'osx' in text.lower():
            constraints.append('macOS platform')

        return constraints

    def _check_clarification_needed(self, text: str, analysis: ProblemAnalysis) -> Tuple[bool, List[str]]:
        """Determine if clarification questions are needed."""
        questions = []

        if not analysis.language:
            questions.append("What programming language are you using?")

        if not analysis.error_messages and 'error' in text.lower():
            questions.append("Can you provide the exact error message you're seeing?")

        if not analysis.functionality:
            questions.append("What exactly are you trying to accomplish with this code?")

        # Check for ambiguous terms
        if 'it' in text.lower() and not analysis.functionality:
            questions.append("What does 'it' refer to in your description?")

        return len(questions) > 0, questions

    def _calculate_confidence(self, analysis: ProblemAnalysis) -> float:
        """Calculate confidence score for the analysis."""
        score = 0.0

        if analysis.language:
            score += 0.3
        if analysis.libraries:
            score += 0.2
        if analysis.error_messages:
            score += 0.2
        if analysis.functionality:
            score += 0.2
        if analysis.error_type:
            score += 0.1

        return min(score, 1.0)