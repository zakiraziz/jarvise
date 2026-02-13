"""
Code Generation Engine
Uses OpenAI to generate code solutions based on problem analysis with enhanced features.
"""

import openai
import logging
import re
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import hashlib
from pathlib import Path
import time
import ast
import builtins
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import subprocess
import tempfile
import os

from .problem_parser import ProblemAnalysis
from .safety_checker import SafetyChecker

logger = logging.getLogger(__name__)


class GenerationMode(Enum):
    """Different code generation modes."""
    FIX = "fix"  # Fix existing code
    OPTIMIZE = "optimize"  # Optimize existing code
    REFACTOR = "refactor"  # Refactor with better patterns
    EXPLAIN = "explain"  # Just explain without generating
    COMPLETE = "complete"  # Generate from scratch
    TEST = "test"  # Generate tests
    DOCUMENT = "document"  # Generate documentation
    REVIEW = "review"  # Code review mode
    DEBUG = "debug"  # Debug existing code
    CONVERT = "convert"  # Convert between languages


class ComplexityLevel(Enum):
    """Solution complexity levels."""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    PRODUCTION = "production"
    ENTERPRISE = "enterprise"


class CodeQuality(Enum):
    """Code quality assessment levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNSAFE = "unsafe"


@dataclass
class GenerationConfig:
    """Enhanced configuration for code generation."""
    mode: GenerationMode = GenerationMode.COMPLETE
    complexity: ComplexityLevel = ComplexityLevel.PRODUCTION
    include_tests: bool = True
    include_documentation: bool = True
    include_examples: bool = True
    include_comments: bool = True
    include_type_hints: bool = True
    include_error_handling: bool = True
    include_logging: bool = False
    include_performance_metrics: bool = False
    max_tokens: int = 2000
    temperature: float = 0.1
    top_p: float = 0.95
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: int = 30
    num_alternatives: int = 0
    cache_results: bool = True
    stream_output: bool = False
    validate_syntax: bool = True
    validate_security: bool = True
    max_iterations: int = 3
    code_style: str = "pep8"  # pep8, google, black, etc.
    framework: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)


@dataclass
class CodeMetrics:
    """Code quality metrics."""
    lines_of_code: int = 0
    comment_lines: int = 0
    blank_lines: int = 0
    complexity_score: float = 0.0
    maintainability_index: float = 0.0
    cyclomatic_complexity: int = 0
    code_smells: List[str] = field(default_factory=list)
    security_issues: List[str] = field(default_factory=list)
    performance_score: float = 0.0
    test_coverage: Optional[float] = None


class CodeGenerator:
    """Enhanced code generator with advanced features."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4-turbo-preview",
        safety_checker: Optional[SafetyChecker] = None,
        cache_dir: Optional[str] = None,
        max_workers: int = 4,
        enable_validation: bool = True
    ):
        self.api_key = api_key
        self.model = model
        self.client = openai.OpenAI(api_key=api_key)
        self.safety_checker = safety_checker or SafetyChecker()
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_workers = max_workers
        self.enable_validation = enable_validation
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Language-specific configurations
        self.language_configs = {
            'python': {
                'extension': '.py',
                'comment_style': '#',
                'test_frameworks': ['pytest', 'unittest'],
                'linter': 'pylint',
                'formatter': 'black'
            },
            'javascript': {
                'extension': '.js',
                'comment_style': '//',
                'test_frameworks': ['jest', 'mocha'],
                'linter': 'eslint',
                'formatter': 'prettier'
            },
            'typescript': {
                'extension': '.ts',
                'comment_style': '//',
                'test_frameworks': ['jest', 'mocha'],
                'linter': 'eslint',
                'formatter': 'prettier'
            },
            'java': {
                'extension': '.java',
                'comment_style': '//',
                'test_frameworks': ['junit', 'testng'],
                'linter': 'checkstyle',
                'formatter': 'google-java-format'
            },
            'cpp': {
                'extension': '.cpp',
                'comment_style': '//',
                'test_frameworks': ['gtest', 'catch2'],
                'linter': 'cpplint',
                'formatter': 'clang-format'
            },
            'go': {
                'extension': '.go',
                'comment_style': '//',
                'test_frameworks': ['testing'],
                'linter': 'golint',
                'formatter': 'gofmt'
            },
            'rust': {
                'extension': '.rs',
                'comment_style': '//',
                'test_frameworks': ['built-in'],
                'linter': 'clippy',
                'formatter': 'rustfmt'
            },
            'ruby': {
                'extension': '.rb',
                'comment_style': '#',
                'test_frameworks': ['rspec', 'minitest'],
                'linter': 'rubocop',
                'formatter': 'rubocop'
            }
        }

    def generate_solution(
        self,
        problem_description: str,
        analysis: ProblemAnalysis,
        config: Optional[GenerationConfig] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict:
        """
        Generate a complete code solution with enhanced configuration.

        Args:
            problem_description: Original problem text
            analysis: Parsed problem analysis
            config: Optional generation configuration
            context: Additional context for generation

        Returns:
            Dict containing solution, explanation, and metadata
        """
        config = config or GenerationConfig()
        context = context or {}

        try:
            # Check cache if enabled
            if config.cache_results:
                cached = self._check_cache(problem_description, analysis, config)
                if cached:
                    logger.info("Returning cached solution")
                    return cached

            # Special handling for different modes
            if config.mode == GenerationMode.REVIEW:
                return self._review_code(problem_description, analysis, config)
            elif config.mode == GenerationMode.DEBUG:
                return self._debug_code(problem_description, analysis, config)
            elif config.mode == GenerationMode.CONVERT:
                return self._convert_code(problem_description, analysis, config)

            # Build the prompt based on configuration
            prompt = self._build_generation_prompt(
                problem_description, 
                analysis, 
                config,
                context
            )

            # Generate code with retry logic
            solution = self._generate_with_iterative_improvement(
                prompt, analysis, config
            )

            # Safety check
            if analysis.language and solution.get('code'):
                solution = self._perform_safety_checks(solution, analysis, config)

            # Validate syntax if enabled
            if config.validate_syntax and solution.get('code'):
                solution = self._validate_syntax(solution, analysis.language)

            # Generate additional components
            if config.include_tests:
                solution['tests'] = self._generate_test_cases(
                    solution['code'], 
                    analysis.language,
                    config.framework
                )

            if config.include_documentation:
                solution['documentation'] = self._generate_documentation(
                    solution['code'],
                    analysis.language
                )

            # Calculate code metrics
            solution['metrics'] = self._calculate_metrics(
                solution['code'],
                analysis.language
            )

            # Generate alternatives if requested
            if config.num_alternatives > 0:
                solution['alternatives'] = self._generate_alternatives(
                    problem_description, 
                    analysis, 
                    config
                )

            # Add enhanced metadata
            solution['metadata'] = self._build_metadata(analysis, config)
            solution['quality_assessment'] = self._assess_code_quality(solution)

            # Cache the result
            if config.cache_results:
                self._cache_result(problem_description, analysis, config, solution)

            logger.info(f"Generated solution for {analysis.language} problem in {config.mode.value} mode")
            return solution

        except Exception as e:
            logger.error(f"Error generating solution: {e}")
            return self._create_error_response(str(e))

    def _generate_with_iterative_improvement(
        self,
        prompt: str,
        analysis: ProblemAnalysis,
        config: GenerationConfig
    ) -> Dict:
        """Generate code with iterative improvements."""
        best_solution = None
        best_score = -1
        
        for iteration in range(config.max_iterations):
            try:
                response = self._generate_with_retry(prompt, config)
                solution_text = response.choices[0].message.content.strip()
                solution = self._parse_solution_response(solution_text, config)
                
                # Score the solution
                score = self._score_solution(solution, analysis.language)
                
                if score > best_score:
                    best_score = score
                    best_solution = solution
                    
                # If we found a good solution, break early
                if score > 0.8:
                    break
                    
                # Update prompt with feedback
                if iteration < config.max_iterations - 1:
                    prompt = self._add_feedback_to_prompt(
                        prompt, solution, analysis.language
                    )
                    
            except Exception as e:
                logger.warning(f"Iteration {iteration} failed: {e}")
                continue
        
        return best_solution or solution

    def _score_solution(self, solution: Dict, language: str) -> float:
        """Score the quality of a solution."""
        score = 0.5  # Base score
        
        code = solution.get('code', '')
        
        # Check for completeness
        if len(code) > 100:
            score += 0.1
            
        # Check for error handling
        if 'try' in code or 'except' in code or 'catch' in code:
            score += 0.1
            
        # Check for comments
        if '#' in code or '//' in code or '/*' in code:
            score += 0.05
            
        # Check for type hints (Python specific)
        if language == 'python' and ':' in code and '->' in code:
            score += 0.05
            
        # Check for docstrings
        if '"""' in code or "'''" in code:
            score += 0.05
            
        # Check for input validation
        if 'if' in code and ('None' in code or 'null' in code or 'undefined' in code):
            score += 0.05
            
        return min(1.0, score)

    def _perform_safety_checks(
        self,
        solution: Dict,
        analysis: ProblemAnalysis,
        config: GenerationConfig
    ) -> Dict:
        """Perform comprehensive safety checks."""
        code = solution.get('code', '')
        language = analysis.language
        
        # Basic safety check
        is_safe, issues = self.safety_checker.check_code(code, language)
        solution['safety_check'] = {
            'is_safe': is_safe,
            'issues': issues,
            'warning': self.safety_checker.generate_safety_warning(issues) if not is_safe else ""
        }
        
        # Security scan if enabled
        if config.validate_security:
            security_issues = self._scan_security(code, language)
            if security_issues:
                solution['security_check'] = {
                    'issues': security_issues,
                    'warning': "Security issues detected. Review carefully."
                }
        
        # Try to fix unsafe code
        if not is_safe and config.mode == GenerationMode.FIX:
            solution = self._fix_unsafe_code(solution, analysis, issues)
            
        return solution

    def _scan_security(self, code: str, language: str) -> List[str]:
        """Scan code for security issues."""
        issues = []
        
        # Common security patterns to check
        security_patterns = {
            'python': [
                (r'eval\(', 'Use of eval() can be dangerous'),
                (r'exec\(', 'Use of exec() can be dangerous'),
                (r'__import__\(', 'Dynamic imports can be unsafe'),
                (r'pickle\.loads?', 'Pickle can execute arbitrary code'),
                (r'subprocess\.Popen\(.*shell=True', 'Shell=True can be dangerous'),
                (r'os\.system\(', 'Use subprocess instead of os.system'),
                (r'sqlite3\.execute\(.*\%', 'Possible SQL injection vulnerability'),
                (r'\.format\(.*__', 'Template injection possible')
            ],
            'javascript': [
                (r'eval\(', 'Use of eval() can be dangerous'),
                (r'Function\(', 'Function constructor can be dangerous'),
                (r'document\.write\(', 'Can lead to XSS vulnerabilities'),
                (r'innerHTML\s*=', 'Can lead to XSS vulnerabilities'),
                (r'localStorage\.', 'Sensitive data in localStorage')
            ]
        }
        
        patterns = security_patterns.get(language.lower(), [])
        for pattern, message in patterns:
            if re.search(pattern, code, re.IGNORECASE):
                issues.append(message)
                
        return issues

    def _validate_syntax(self, solution: Dict, language: str) -> Dict:
        """Validate code syntax."""
        code = solution.get('code', '')
        
        if language.lower() == 'python':
            try:
                ast.parse(code)
                solution['syntax_valid'] = True
            except SyntaxError as e:
                solution['syntax_valid'] = False
                solution['syntax_error'] = str(e)
                
        elif language.lower() in ['javascript', 'typescript']:
            # Basic validation for JS/TS
            try:
                # Use a simple validation approach
                if 'function' in code or 'class' in code:
                    solution['syntax_valid'] = True
                else:
                    solution['syntax_valid'] = False
                    solution['syntax_error'] = "Could not validate syntax"
            except Exception as e:
                solution['syntax_valid'] = False
                solution['syntax_error'] = str(e)
                
        return solution

    def generate_test_cases(
        self,
        code: str,
        language: str,
        framework: Optional[str] = None
    ) -> Dict:
        """Enhanced test case generation with multiple frameworks."""
        
        # Detect appropriate test framework
        if not framework:
            framework = self._detect_test_framework(language)
        
        prompt = self._build_test_prompt(code, language, framework)
        
        try:
            response = self._generate_with_retry(prompt, GenerationConfig())
            test_text = response.choices[0].message.content.strip()
            
            # Parse test cases
            tests = self._parse_test_response(test_text, language)
            
            # Validate tests
            validated_tests = self._validate_tests(tests, code, language)
            
            return {
                'framework': framework,
                'code': validated_tests.get('code', ''),
                'test_cases': validated_tests.get('cases', []),
                'coverage_estimate': self._estimate_test_coverage(code, validated_tests),
                'execution_instructions': self._get_test_instructions(language, framework)
            }
            
        except Exception as e:
            logger.error(f"Error generating tests: {e}")
            return {'error': str(e)}

    def _build_test_prompt(self, code: str, language: str, framework: str) -> str:
        """Build prompt for test generation."""
        return f"""
Generate comprehensive test cases for the following {language} code using {framework}:

```{language}
{code}
