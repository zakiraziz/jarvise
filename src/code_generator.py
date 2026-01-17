"""
Code Generation Engine
Uses OpenAI to generate code solutions based on problem analysis.
"""

import openai
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from .problem_parser import ProblemAnalysis
from .safety_checker import SafetyChecker

logger = logging.getLogger(__name__)

class CodeGenerator:
    """Generates code solutions using OpenAI."""

    def __init__(self, api_key: str, model: str = "gpt-4-turbo-preview", safety_checker: Optional[SafetyChecker] = None):
        self.api_key = api_key
        self.model = model
        self.client = openai.OpenAI(api_key=api_key)
        self.safety_checker = safety_checker or SafetyChecker()

    def generate_solution(self, problem_description: str, analysis: ProblemAnalysis) -> Dict:
        """
        Generate a complete code solution.

        Args:
            problem_description: Original problem text
            analysis: Parsed problem analysis

        Returns:
            Dict containing solution, explanation, and metadata
        """
        try:
            # Build the prompt
            prompt = self._build_generation_prompt(problem_description, analysis)

            # Generate code
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.1,  # Low temperature for consistent code
                timeout=30
            )

            solution_text = response.choices[0].message.content.strip()

            # Parse the response
            solution = self._parse_solution_response(solution_text)

            # Safety check
            if analysis.language and solution.get('code'):
                is_safe, issues = self.safety_checker.check_code(solution['code'], analysis.language)
                solution['safety_check'] = {
                    'is_safe': is_safe,
                    'issues': issues,
                    'warning': self.safety_checker.generate_safety_warning(issues) if not is_safe else ""
                }

            # Add metadata
            solution['metadata'] = {
                'generated_at': datetime.now().isoformat(),
                'model': self.model,
                'language': analysis.language,
                'confidence': analysis.confidence,
                'problem_type': analysis.error_type
            }

            logger.info(f"Generated solution for {analysis.language} problem")
            return solution

        except Exception as e:
            logger.error(f"Error generating solution: {e}")
            return {
                'error': str(e),
                'code': None,
                'explanation': "Failed to generate solution due to an error.",
                'metadata': {'error': True}
            }

    def _build_generation_prompt(self, description: str, analysis: ProblemAnalysis) -> str:
        """Build the code generation prompt."""
        prompt_parts = []

        # Problem description
        prompt_parts.append(f"Problem Description: {description}")

        # Technical details
        if analysis.language:
            prompt_parts.append(f"Programming Language: {analysis.language}")

        if analysis.libraries:
            prompt_parts.append(f"Libraries/Frameworks: {', '.join(analysis.libraries)}")

        if analysis.error_messages:
            prompt_parts.append(f"Error Messages: {'; '.join(analysis.error_messages)}")

        if analysis.error_type:
            prompt_parts.append(f"Error Type: {analysis.error_type}")

        if analysis.constraints:
            prompt_parts.append(f"Constraints: {'; '.join(analysis.constraints)}")

        # Requirements
        prompt_parts.append("""
Please provide a complete, working solution with:
1. The corrected code
2. Detailed explanation of the fix
3. Comments in the code explaining key parts
4. Best practices recommendations
5. Prevention tips for similar issues

Format your response as:
## Code Solution
```language
[code here]
```

## Explanation
[detailed explanation]

## Best Practices
[list of recommendations]

## Prevention
[tips to avoid similar issues]
""")

        return "\n\n".join(prompt_parts)

    def _get_system_prompt(self) -> str:
        """Get the system prompt for code generation."""
        return """You are an expert software engineer and coding assistant. Your task is to analyze coding problems and provide complete, working solutions.

Guidelines:
- Always provide production-ready code
- Include proper error handling
- Add comprehensive comments
- Follow language-specific best practices
- Suggest security improvements when relevant
- Explain the root cause and solution clearly
- Provide alternative approaches when beneficial
- Include testing recommendations

Safety: Never generate malicious code, exploits, or insecure solutions. Always prioritize security and best practices.

Be thorough but concise in explanations."""

    def _parse_solution_response(self, response: str) -> Dict:
        """Parse the AI response into structured components."""
        solution = {
            'code': None,
            'explanation': '',
            'best_practices': [],
            'prevention': [],
            'alternatives': []
        }

        # Extract code blocks
        import re
        code_blocks = re.findall(r'```(\w+)?\n(.*?)\n```', response, re.DOTALL)

        if code_blocks:
            # Use the first code block as the main solution
            language, code = code_blocks[0]
            solution['code'] = code.strip()

        # Extract sections
        sections = re.split(r'^##\s+', response, flags=re.MULTILINE)

        for section in sections:
            if section.strip():
                lines = section.strip().split('\n', 1)
                if len(lines) == 2:
                    section_title, content = lines
                    section_title = section_title.lower().strip()

                    if 'explanation' in section_title:
                        solution['explanation'] = content.strip()
                    elif 'best practices' in section_title:
                        solution['best_practices'] = [line.strip('- ').strip() for line in content.split('\n') if line.strip()]
                    elif 'prevention' in section_title:
                        solution['prevention'] = [line.strip('- ').strip() for line in content.split('\n') if line.strip()]
                    elif 'alternative' in section_title:
                        solution['alternatives'] = [line.strip('- ').strip() for line in content.split('\n') if line.strip()]

        return solution

    def generate_followup(self, original_problem: str, solution: Dict, user_question: str) -> Dict:
        """Generate a response to a follow-up question about the solution."""
        context = f"""
Original Problem: {original_problem}

Previous Solution: {solution.get('code', 'N/A')}

User Question: {user_question}
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a coding assistant answering follow-up questions about previous solutions. Be helpful, clear, and provide additional code examples when needed."},
                    {"role": "user", "content": context}
                ],
                max_tokens=1000,
                temperature=0.2
            )

            return {
                'answer': response.choices[0].message.content.strip(),
                'type': 'followup'
            }

        except Exception as e:
            return {
                'error': str(e),
                'answer': "Sorry, I couldn't generate a response to your follow-up question."
            }