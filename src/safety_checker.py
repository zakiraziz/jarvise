"""
Safety Checker Module
Ensures generated code is safe and doesn't contain malicious content.
"""

import re
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

class SafetyChecker:
    """Checks code for safety and security issues."""

    def __init__(self, blocked_keywords: Optional[List[str]] = None):
        self.blocked_keywords = blocked_keywords or [
            'malware', 'virus', 'trojan', 'worm', 'ransomware',
            'hack', 'exploit', 'crack', 'keylogger', 'backdoor',
            'rootkit', 'spyware', 'adware', 'botnet',
            'ddos', 'bruteforce', 'sql injection', 'xss',
            'csrf', 'buffer overflow', 'format string',
            'eval', 'exec', 'system', 'subprocess', 'os.system',
            'pickle.loads', 'yaml.load', 'marshal.loads'
        ]

        # Dangerous patterns
        self.dangerous_patterns = [
            r'\bos\.system\s*\(',
            r'\bsubprocess\.(call|Popen|run)\s*\(',
            r'\beval\s*\(',
            r'\bexec\s*\(',
            r'\b__import__\s*\(',
            r'\bopen\s*\(\s*.*\s*[\'"]w[\'"]',
            r'\binput\s*\(\s*.*\)\s*:\s*exec',
            r'\bpickle\.loads?\s*\(',
            r'\byaml\.load\s*\(',
            r'\bmarshal\.loads?\s*\(',
            r'\b shelve\.open\s*\(',
        ]

        # Suspicious imports
        self.suspicious_imports = [
            'socket', 'ftplib', 'smtplib', 'telnetlib',
            'http.client', 'urllib.request', 'urllib.parse',
            'ssl', 'cryptography', 'hashlib', 'hmac'
        ]

    def check_code(self, code: str, language: str) -> Tuple[bool, List[str]]:
        """
        Check if generated code is safe.

        Args:
            code: The code to check
            language: Programming language

        Returns:
            Tuple of (is_safe, list_of_issues)
        """
        issues = []

        # Check for blocked keywords
        for keyword in self.blocked_keywords:
            if keyword.lower() in code.lower():
                issues.append(f"Blocked keyword detected: {keyword}")

        # Check for dangerous patterns
        for pattern in self.dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                issues.append(f"Dangerous pattern detected: {pattern}")

        # Language-specific checks
        if language.lower() == 'python':
            issues.extend(self._check_python_code(code))
        elif language.lower() in ['javascript', 'typescript']:
            issues.extend(self._check_javascript_code(code))

        # Check for suspicious network/file operations
        if self._has_suspicious_operations(code, language):
            issues.append("Code contains potentially unsafe file or network operations")

        is_safe = len(issues) == 0

        if not is_safe:
            logger.warning(f"Unsafe code detected: {issues}")

        return is_safe, issues

    def _check_python_code(self, code: str) -> List[str]:
        """Python-specific safety checks."""
        issues = []

        # Check for dangerous imports
        for imp in self.suspicious_imports:
            if f"import {imp}" in code or f"from {imp}" in code:
                issues.append(f"Potentially unsafe import: {imp}")

        # Check for file operations without proper validation
        if 'open(' in code and 'with ' not in code:
            issues.append("File operations should use 'with' statement for proper resource management")

        # Check for hardcoded credentials
        if re.search(r'password\s*=\s*[\'"][^\'"]*[\'"]', code, re.IGNORECASE):
            issues.append("Hardcoded password detected")

        if re.search(r'api_key\s*=\s*[\'"][^\'"]*[\'"]', code, re.IGNORECASE):
            issues.append("Hardcoded API key detected")

        return issues

    def _check_javascript_code(self, code: str) -> List[str]:
        """JavaScript-specific safety checks."""
        issues = []

        # Check for dangerous DOM manipulation
        if 'document.write(' in code or 'innerHTML' in code:
            issues.append("Potentially unsafe DOM manipulation")

        # Check for eval usage
        if 'eval(' in code:
            issues.append("Use of eval() is dangerous")

        # Check for hardcoded secrets
        if re.search(r'apiKey\s*:\s*[\'"][^\'"]*[\'"]', code):
            issues.append("Hardcoded API key detected")

        return issues

    def _has_suspicious_operations(self, code: str, language: str) -> bool:
        """Check for suspicious file/network operations."""
        suspicious_indicators = [
            'socket', 'connect', 'listen', 'bind', 'accept',
            'open(', 'write(', 'read(', 'close(',
            'http', 'ftp', 'ssh', 'telnet',
            'encrypt', 'decrypt', 'hash',
        ]

        code_lower = code.lower()
        count = sum(1 for indicator in suspicious_indicators if indicator in code_lower)

        return count > 3  # If more than 3 suspicious operations, flag it

    def sanitize_code(self, code: str, language: str) -> str:
        """
        Attempt to sanitize potentially unsafe code.
        Note: This is not foolproof and should not be relied upon for security.
        """
        # Remove dangerous imports
        if language.lower() == 'python':
            dangerous_imports = [
                r'^import\s+(socket|ftplib|smtplib|telnetlib)',
                r'^from\s+(socket|ftplib|smtplib|telnetlib)',
            ]
            for pattern in dangerous_imports:
                code = re.sub(pattern, '# DANGEROUS IMPORT REMOVED', code, flags=re.MULTILINE)

        return code

    def generate_safety_warning(self, issues: List[str]) -> str:
        """Generate a safety warning message."""
        if not issues:
            return ""

        warning = "⚠️  SAFETY WARNING:\n"
        warning += "The generated code may contain security risks:\n"
        for issue in issues:
            warning += f"• {issue}\n"
        warning += "\nPlease review the code carefully before execution.\n"
        warning += "Consider using this code only in a safe, isolated environment.\n"

        return warning