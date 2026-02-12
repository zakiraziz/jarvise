"""
Safety Checker Module
Ensures generated code is safe and doesn't contain malicious content.
"""

import re
import logging
import ast
import json
from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import hashlib
from datetime import datetime

logger = logging.getLogger(__name__)


class SeverityLevel(Enum):
    """Severity levels for safety issues."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IssueCategory(Enum):
    """Categories of safety issues."""
    MALICIOUS_CODE = "malicious_code"
    INSECURE_FUNCTION = "insecure_function"
    HARDCODED_SECRETS = "hardcoded_secrets"
    DANGEROUS_IMPORT = "dangerous_import"
    UNSAFE_FILE_OP = "unsafe_file_operation"
    UNSAFE_NETWORK = "unsafe_network"
    CODE_INJECTION = "code_injection"
    CRYPTO_WEAKNESS = "crypto_weakness"
    DEPENDENCY_RISK = "dependency_risk"
    BEST_PRACTICE = "best_practice"
    PERFORMANCE = "performance"
    MEMORY_SAFETY = "memory_safety"
    CONCURRENCY = "concurrency"


@dataclass
class SafetyIssue:
    """Represents a safety issue found in code."""
    category: IssueCategory
    severity: SeverityLevel
    message: str
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None
    suggestion: Optional[str] = None
    cwe_id: Optional[str] = None  # Common Weakness Enumeration ID


class SafetyChecker:
    """Checks code for safety and security issues with enhanced capabilities."""

    def __init__(
        self,
        blocked_keywords: Optional[List[str]] = None,
        config_path: Optional[str] = None,
        enable_ast_analysis: bool = True,
        max_file_size: int = 1024 * 1024  # 1MB
    ):
        # Load configuration if provided
        self.config = self._load_config(config_path) if config_path else {}
        
        # Initialize blocklists and patterns
        self.blocked_keywords = blocked_keywords or self._get_default_blocked_keywords()
        self.dangerous_patterns = self._get_default_dangerous_patterns()
        self.suspicious_imports = self._get_default_suspicious_imports()
        
        # Enhanced tracking
        self.enable_ast_analysis = enable_ast_analysis
        self.max_file_size = max_file_size
        self.issue_cache = {}
        
        # Language-specific checkers
        self.language_checkers = {
            'python': self._check_python_code,
            'javascript': self._check_javascript_code,
            'typescript': self._check_javascript_code,
            'java': self._check_java_code,
            'cpp': self._check_cpp_code,
            'c': self._check_cpp_code,
            'csharp': self._check_csharp_code,
            'go': self._check_go_code,
            'rust': self._check_rust_code,
            'ruby': self._check_ruby_code,
            'php': self._check_php_code,
            'sql': self._check_sql_code,
            'bash': self._check_bash_code,
            'powershell': self._check_powershell_code,
        }

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return {}

    def _get_default_blocked_keywords(self) -> List[str]:
        """Get default blocked keywords with categories."""
        return [
            # Malware-related
            'malware', 'virus', 'trojan', 'worm', 'ransomware',
            'hack', 'exploit', 'crack', 'keylogger', 'backdoor',
            'rootkit', 'spyware', 'adware', 'botnet',
            
            # Attack-related
            'ddos', 'bruteforce', 'sql injection', 'xss',
            'csrf', 'buffer overflow', 'format string',
            'privilege escalation', 'arbitrary code',
            
            # Dangerous functions
            'eval', 'exec', 'system', 'subprocess', 'os.system',
            'pickle.loads', 'yaml.load', 'marshal.loads',
            'php://input', 'allow_url_include', 'base64_decode',
            
            # Cryptography misuse
            'md5(', 'sha1(', 'weak_encryption', 'broken_crypto',
            
            # Hardcoded credentials
            'password=', 'passwd=', 'pwd=', 'secret_key=',
            'api_key=', 'aws_secret', 'private_key'
        ]

    def _get_default_dangerous_patterns(self) -> List[str]:
        """Get default dangerous patterns with regex."""
        return [
            r'\bos\.system\s*\(',
            r'\bsubprocess\.(call|Popen|run|check_output)\s*\(',
            r'\beval\s*\(',
            r'\bexec\s*\(',
            r'\b__import__\s*\(',
            r'\bcompile\s*\(',
            r'\bopen\s*\(\s*.*\s*[\'"]w[\'"]',
            r'\binput\s*\(\s*.*\)\s*:\s*exec',
            r'\bpickle\.loads?\s*\(',
            r'\byaml\.load\s*\(\s*(?!.*Loader=SafeLoader)',
            r'\bmarshal\.loads?\s*\(',
            r'\bshelve\.open\s*\(',
            r'\bglobals\(\)\.update\(',
            r'\blocals\(\)\.update\(',
            r'\b__builtins__\s*\[\s*[\'"]eval[\'"]\s*\]',
            r'\bpty\.spawn\(',
            r'\bpexpect\.spawn\(',
            r'\bsqlite3\.connect\s*\([^,)]*\)',  # SQL injection risk
            r'\bexecScript\b',  # JavaScript
            r'\bsetTimeout\s*\(\s*[\'"].*[\'"]\s*,\s*0\s*\)',  # Potential infinite loop
        ]

    def _get_default_suspicious_imports(self) -> Dict[str, str]:
        """Get default suspicious imports with risk descriptions."""
        return {
            'socket': 'Network socket operations - verify destination and port',
            'ftplib': 'FTP operations - consider using SFTP/FTPS',
            'smtplib': 'SMTP email operations - verify email content',
            'telnetlib': 'Insecure telnet protocol - use SSH instead',
            'http.client': 'HTTP operations - validate URLs and use HTTPS',
            'urllib.request': 'URL requests - validate URLs and use HTTPS',
            'urllib.parse': 'URL parsing - validate input',
            'ssl': 'SSL/TLS - verify certificate validation is enabled',
            'cryptography': 'Cryptography - ensure proper key management',
            'hashlib': 'Hashing - avoid weak algorithms (MD5, SHA1)',
            'hmac': 'HMAC - ensure secure key storage',
            'pickle': 'Unsafe deserialization - avoid untrusted data',
            'subprocess': 'System command execution - validate input',
            'os': 'Operating system operations - validate paths',
            'shutil': 'File operations - validate paths',
            'tempfile': 'Temporary files - ensure secure permissions',
            'ctypes': 'C interface - memory safety risks',
            'winreg': 'Windows registry - modify with caution',
            'win32api': 'Windows API - validate usage',
        }

    def check_code(
        self, 
        code: str, 
        language: str,
        context: Optional[Dict] = None,
        detailed: bool = False
    ) -> Tuple[bool, List[SafetyIssue]]:
        """
        Enhanced code safety check with detailed issue reporting.

        Args:
            code: The code to check
            language: Programming language
            context: Additional context (e.g., 'web_app', 'cli_tool')
            detailed: Return detailed issue objects

        Returns:
            Tuple of (is_safe, list_of_issues)
        """
        if len(code) > self.max_file_size:
            logger.warning(f"Code exceeds max size ({len(code)} > {self.max_file_size})")
            return False, [SafetyIssue(
                category=IssueCategory.BEST_PRACTICE,
                severity=SeverityLevel.MEDIUM,
                message=f"Code size exceeds maximum allowed ({self.max_file_size} bytes)",
                suggestion="Split code into smaller modules"
            )]

        issues = []
        context = context or {}

        # Check for blocked keywords with context
        for keyword in self.blocked_keywords:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            for match in re.finditer(pattern, code, re.IGNORECASE):
                line_number = code[:match.start()].count('\n') + 1
                issues.append(SafetyIssue(
                    category=IssueCategory.MALICIOUS_CODE,
                    severity=SeverityLevel.HIGH,
                    message=f"Blocked keyword detected: {keyword}",
                    line_number=line_number,
                    code_snippet=self._get_code_snippet(code, line_number),
                    suggestion=f"Remove or replace '{keyword}' with safe alternatives"
                ))

        # Check for dangerous patterns
        for pattern in self.dangerous_patterns:
            for match in re.finditer(pattern, code, re.IGNORECASE):
                line_number = code[:match.start()].count('\n') + 1
                issues.append(SafetyIssue(
                    category=IssueCategory.INSECURE_FUNCTION,
                    severity=SeverityLevel.CRITICAL,
                    message=f"Dangerous pattern detected: {match.group()}",
                    line_number=line_number,
                    code_snippet=self._get_code_snippet(code, line_number),
                    suggestion=self._get_suggestion_for_pattern(match.group()),
                    cwe_id=self._get_cwe_for_pattern(match.group())
                ))

