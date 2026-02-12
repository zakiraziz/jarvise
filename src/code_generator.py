"""
Code Generation Engine
Uses OpenAI to generate code solutions based on problem analysis.
"""

import openai
import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib
from pathlib import Path

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


class ComplexityLevel(Enum):
    """Solution complexity levels."""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    PRODUCTION = "production"


@dataclass
class GenerationConfig:
    """Configuration for code generation."""
    mode: GenerationMode = GenerationMode.COMPLETE
    complexity: ComplexityLevel = ComplexityLevel.PRODUCTION
    include_tests: bool = True
    include_documentation: bool = True
    include_examples: bool = True
    max_tokens: int = 2000
    temperature: float = 0.1
    timeout: int = 30
    num_alternatives: int = 0
    cache_results: bool = True
    stream_output: bool = False


class CodeGenerator:
    """Generates code solutions using OpenAI with enhanced features."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4-turbo-preview",
        safety_checker: Optional[SafetyChecker] = None,
        cache_dir: Optional[str] = None
    ):
        self.api_key = api_key
        self.model = model
        self.client = openai.OpenAI(api_key=api_key)
        self.safety_checker = safety_checker or SafetyChecker()
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def generate_solution(
        self,
        problem_description: str,
        analysis: ProblemAnalysis,
        config: Optional[GenerationConfig] = None
    ) -> Dict:
        """
        Generate a complete code solution with enhanced configuration.

        Args:
            problem_description: Original problem text
            analysis: Parsed problem analysis
            config: Optional generation configuration

        Returns:
            Dict containing solution, explanation, and metadata
        """
        config = config or GenerationConfig()

        try:
            # Check cache if enabled
            if config.cache_results:
                cached = self._check_cache(problem_description, analysis, config)
                if cached:
                    logger.info("Returning cached solution")
                    return cached

            # Build the prompt based on configuration
            prompt = self._build_generation_prompt(
                problem_description, 
                analysis, 
                config
            )

            # Generate code
            response = self._generate_with_retry(prompt, config)

            solution_text = response.choices[0].message.content.strip()

            # Parse the response
            solution = self._parse_solution_response(solution_text, config)

            # Safety check
            if analysis.language and solution.get('code'):
                is_safe, issues = self.safety_checker.check_code(
                    solution['code'], 
                    analysis.language
                )
                solution['safety_check'] = {
                    'is_safe': is_safe,
                    'issues': issues,
                    'warning': self.safety_checker.generate_safety_warning(issues) if not is_safe else ""
                }

                # Try to fix unsafe code
                if not is_safe and config.mode == GenerationMode.FIX:
                    solution = self._fix_unsafe_code(solution, analysis, issues)

            # Add enhanced metadata
            solution['metadata'] = self._build_metadata(analysis, config)

            # Generate alternatives if requested
            if config.num_alternatives > 0:
                solution['alternatives'] = self._generate_alternatives(
                    problem_description, 
                    analysis, 
                    config
                )

            # Cache the result
            if config.cache_results:
                self._cache_result(problem_description, analysis, config, solution)

            logger.info(f"Generated solution for {analysis.language} problem in {config.mode.value} mode")
            return solution

        except Exception as e:
            logger.error(f"Error generating solution: {e}")
            return self._create_error_response(str(e))

    def generate_test_cases(
        self,
        code: str,
        language: str,
        framework: Optional[str] = None
    ) -> Dict:
        """Generate test cases for the provided code."""
        prompt = f"""
Generate comprehensive test cases for the following {language} code:

```{language}
{code}
