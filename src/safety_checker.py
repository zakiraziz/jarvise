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
        # Language-specific AST analysis
        if self.enable_ast_analysis and language.lower() == 'python':
            issues.extend(self._analyze_python_ast(code))

        # Language-specific checks
        if language.lower() in self.language_checkers:
            issues.extend(self.language_checkers[language.lower()](code, context))

        # Secret detection
        issues.extend(self._detect_hardcoded_secrets(code))

        # Dependency analysis
        issues.extend(self._analyze_dependencies(code, language))

        # Environment-specific checks
        if context.get('environment'):
            issues.extend(self._check_environment_specific(code, context['environment']))

        # Remove duplicates
        issues = self._deduplicate_issues(issues)

        # Determine if safe based on severity
        critical_issues = [i for i in issues if i.severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH]]
        is_safe = len(critical_issues) == 0

        if issues:
            logger.warning(f"Found {len(issues)} safety issues in {language} code")

        if detailed:
            return is_safe, issues
        else:
            # Return simplified issues for backward compatibility
            simplified_issues = [f"[{i.severity.value}] {i.message}" for i in issues]
            return is_safe, simplified_issues

    def _analyze_python_ast(self, code: str) -> List[SafetyIssue]:
        """Perform AST analysis on Python code."""
        issues = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                # Check for try-except without specific exceptions
                if isinstance(node, ast.Try):
                    for handler in node.handlers:
                        if handler.type is None:
                            issues.append(SafetyIssue(
                                category=IssueCategory.BEST_PRACTICE,
                                severity=SeverityLevel.MEDIUM,
                                message="Bare except clause catches all exceptions",
                                line_number=handler.lineno,
                                code_snippet=self._get_code_snippet(code, handler.lineno),
                                suggestion="Specify specific exceptions to catch"
                            ))
                
                # Check for mutable default arguments
                if isinstance(node, ast.FunctionDef):
                    for arg in node.args.defaults:
                        if isinstance(arg, (ast.List, ast.Dict, ast.Set)):
                            issues.append(SafetyIssue(
                                category=IssueCategory.BEST_PRACTICE,
                                severity=SeverityLevel.LOW,
                                message="Mutable default argument detected",
                                line_number=node.lineno,
                                code_snippet=self._get_code_snippet(code, node.lineno),
                                suggestion="Use None as default and initialize inside function"
                            ))
                
                # Check for assert statements in production code
                if isinstance(node, ast.Assert):
                    issues.append(SafetyIssue(
                        category=IssueCategory.BEST_PRACTICE,
                        severity=SeverityLevel.INFO,
                        message="Assert statement may be disabled in optimized mode",
                        line_number=node.lineno,
                        code_snippet=self._get_code_snippet(code, node.lineno),
                        suggestion="Use proper error handling instead of asserts"
                    ))
                
                # Check for eval/exec usage
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in ['eval', 'exec', 'compile']:
                            issues.append(SafetyIssue(
                                category=IssueCategory.CODE_INJECTION,
                                severity=SeverityLevel.CRITICAL,
                                message=f"Use of {node.func.id}() allows arbitrary code execution",
                                line_number=node.lineno,
                                code_snippet=self._get_code_snippet(code, node.lineno),
                                suggestion="Avoid dynamic code execution, use safe alternatives",
                                cwe_id="CWE-95" if node.func.id == 'eval' else "CWE-94"
                            ))
        
        except SyntaxError as e:
            issues.append(SafetyIssue(
                category=IssueCategory.BEST_PRACTICE,
                severity=SeverityLevel.LOW,
                message=f"Syntax error in code: {str(e)}",
                suggestion="Fix syntax errors before deployment"
            ))
        
        return issues

    def _detect_hardcoded_secrets(self, code: str) -> List[SafetyIssue]:
        """Detect hardcoded secrets, API keys, and credentials."""
        issues = []
        
        patterns = {
            'api_key': r'api[_-]?key\s*[=:]\s*[\'"]([^\'"]+)[\'"]',
            'password': r'password\s*[=:]\s*[\'"]([^\'"]+)[\'"]',
            'aws_key': r'AKIA[0-9A-Z]{16}',
            'private_key': r'-----BEGIN (?:RSA|DSA|EC|OPENSSH) PRIVATE KEY-----',
            'oauth_token': r'oauth[_-]?token\s*[=:]\s*[\'"]([^\'"]+)[\'"]',
            'auth_token': r'auth[_-]?token\s*[=:]\s*[\'"]([^\'"]+)[\'"]',
            'jwt_token': r'eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*',
        }
        
        for secret_type, pattern in patterns.items():
            for match in re.finditer(pattern, code, re.IGNORECASE):
                line_number = code[:match.start()].count('\n') + 1
                severity = SeverityLevel.HIGH
                
                # Check if it's likely a placeholder
                value = match.group(1) if len(match.groups()) > 0 else match.group()
                if any(placeholder in value.lower() for placeholder in ['your_', 'xxxx', 'example', 'changeme']):
                    severity = SeverityLevel.MEDIUM
                
                issues.append(SafetyIssue(
                    category=IssueCategory.HARDCODED_SECRETS,
                    severity=severity,
                    message=f"Hardcoded {secret_type.replace('_', ' ')} detected",
                    line_number=line_number,
                    code_snippet=self._get_code_snippet(code, line_number),
                    suggestion="Store secrets in environment variables or secure vault",
                    cwe_id="CWE-798"
                ))
        
        return issues

    def _analyze_dependencies(self, code: str, language: str) -> List[SafetyIssue]:
        """Analyze dependencies for security risks."""
        issues = []
        
        if language.lower() == 'python':
            # Check for outdated/insecure packages
            import_patterns = [
                r'^\s*import\s+(\w+)',
                r'^\s*from\s+(\w+)\s+import',
            ]
            
            for pattern in import_patterns:
                for match in re.finditer(pattern, code, re.MULTILINE):
                    module = match.group(1)
                    if module in self.suspicious_imports:
                        line_number = code[:match.start()].count('\n') + 1
                        issues.append(SafetyIssue(
                            category=IssueCategory.DEPENDENCY_RISK,
                            severity=SeverityLevel.MEDIUM,
                            message=f"Suspicious import: {module} - {self.suspicious_imports[module]}",
                            line_number=line_number,
                            code_snippet=self._get_code_snippet(code, line_number),
                            suggestion=f"Review usage of {module} and consider alternatives"
                        ))
        
        return issues

    def _check_java_code(self, code: str, context: Dict) -> List[SafetyIssue]:
        """Java-specific safety checks."""
        issues = []
        
        # Check for insecure deserialization
        if 'ObjectInputStream' in code and 'readObject' in code:
            issues.append(SafetyIssue(
                category=IssueCategory.UNSAFE_FILE_OP,
                severity=SeverityLevel.HIGH,
                message="Insecure deserialization detected",
                suggestion="Use safe deserialization methods or validate input",
                cwe_id="CWE-502"
            ))
        
        # Check for SQL injection
        if 'Statement' in code and 'executeQuery' in code:
            if '?' not in code and 'PreparedStatement' not in code:
                issues.append(SafetyIssue(
                    category=IssueCategory.CODE_INJECTION,
                    severity=SeverityLevel.CRITICAL,
                    message="Potential SQL injection - use PreparedStatement",
                    suggestion="Replace Statement with PreparedStatement and use parameterized queries",
                    cwe_id="CWE-89"
                ))
        
        return issues

    def _check_cpp_code(self, code: str, context: Dict) -> List[SafetyIssue]:
        """C/C++-specific safety checks."""
        issues = []
        
        # Check for buffer overflows
        if 'strcpy' in code or 'strcat' in code:
            issues.append(SafetyIssue(
                category=IssueCategory.MEMORY_SAFETY,
                severity=SeverityLevel.CRITICAL,
                message="Unsafe string function detected",
                suggestion="Use strncpy/strncat or safer alternatives",
                cwe_id="CWE-120"
            ))
        
        # Check for format string vulnerabilities
        if 'printf' in code and '%n' in code:
            issues.append(SafetyIssue(
                category=IssueCategory.MEMORY_SAFETY,
                severity=SeverityLevel.CRITICAL,
                message="Format string vulnerability",
                suggestion="Avoid using %n format specifier",
                cwe_id="CWE-134"
            ))
        
        return issues

    def _check_csharp_code(self, code: str, context: Dict) -> List[SafetyIssue]:
        """C#-specific safety checks."""
        issues = []
                # Check for insecure deserialization
        if 'BinaryFormatter' in code or 'SoapFormatter' in code:
            issues.append(SafetyIssue(
                category=IssueCategory.UNSAFE_FILE_OP,
                severity=SeverityLevel.HIGH,
                message="Insecure deserializer detected",
                suggestion="Use XmlSerializer or DataContractSerializer with proper validation",
                cwe_id="CWE-502"
            ))
        
        return issues

    def _check_go_code(self, code: str, context: Dict) -> List[SafetyIssue]:
        """Go-specific safety checks."""
        issues = []
        
        # Check for goroutine leaks
        if 'go ' in code and 'sync.WaitGroup' not in code:
            issues.append(SafetyIssue(
                category=IssueCategory.CONCURRENCY,
                severity=SeverityLevel.MEDIUM,
                message="Potential goroutine leak",
                suggestion="Use WaitGroup or channels to manage goroutine lifecycle"
            ))
        
        return issues

    def _check_rust_code(self, code: str, context: Dict) -> List[SafetyIssue]:
        """Rust-specific safety checks."""
        issues = []
        
        # Check for unsafe blocks
        if 'unsafe {' in code:
            issues.append(SafetyIssue(
                category=IssueCategory.MEMORY_SAFETY,
                severity=SeverityLevel.MEDIUM,
                message="Unsafe block detected",
                suggestion="Minimize unsafe code and verify invariants"
            ))
        
        return issues

    def _check_ruby_code(self, code: str, context: Dict) -> List[SafetyIssue]:
        """Ruby-specific safety checks."""
        issues = []
        
        # Check for unsafe YAML loading
        if 'YAML.load' in code and 'YAML.safe_load' not in code:
            issues.append(SafetyIssue(
                category=IssueCategory.UNSAFE_FILE_OP,
                severity=SeverityLevel.HIGH,
                message="Unsafe YAML loading detected",
                suggestion="Use YAML.safe_load instead",
                cwe_id="CWE-502"
            ))
        
        return issues

    def _check_php_code(self, code: str, context: Dict) -> List[SafetyIssue]:
        """PHP-specific safety checks."""
        issues = []
        
        # Check for dangerous functions
        dangerous_php = ['eval', 'system', 'exec', 'shell_exec', 'passthru', 'popen']
        for func in dangerous_php:
            if func + '(' in code:
                issues.append(SafetyIssue(
                    category=IssueCategory.CODE_INJECTION,
                    severity=SeverityLevel.CRITICAL,
                    message=f"Dangerous PHP function: {func}()",
                    suggestion=f"Avoid using {func}() with user input"
                ))
        
        return issues

    def _check_sql_code(self, code: str, context: Dict) -> List[SafetyIssue]:
        """SQL-specific safety checks."""
        issues = []
        
        # Check for SQL injection patterns
        if '--' in code:
            issues.append(SafetyIssue(
                category=IssueCategory.CODE_INJECTION,
                severity=SeverityLevel.CRITICAL,
                message="Comment syntax detected - potential SQL injection",
                suggestion="Use parameterized queries"
            ))
        
        return issues

    def _check_bash_code(self, code: str, context: Dict) -> List[SafetyIssue]:
        """Bash-specific safety checks."""
        issues = []
        
        # Check for command injection
        if '$((' not in code and '$(' in code:
            issues.append(SafetyIssue(
                category=IssueCategory.CODE_INJECTION,
                severity=SeverityLevel.HIGH,
                message="Command substitution detected",
                suggestion="Be careful with command substitution and validate input"
            ))
        
        return issues

    def _check_powershell_code(self, code: str, context: Dict) -> List[SafetyIssue]:
        """PowerShell-specific safety checks."""
        issues = []
        
        # Check for execution policy bypass
        if 'Bypass' in code and 'ExecutionPolicy' in code:
            issues.append(SafetyIssue(
                category=IssueCategory.MALICIOUS_CODE,
                severity=SeverityLevel.HIGH,
                message="Execution policy bypass detected",
                suggestion="Avoid bypassing security policies"
            ))
        
        return issues

    def _check_environment_specific(self, code: str, environment: str) -> List[SafetyIssue]:
        """Environment-specific safety checks."""
        issues = []
                    # Production-specific checks
            if 'print(' in code or 'console.log(' in code:
                issues.append(SafetyIssue(
                    category=IssueCategory.BEST_PRACTICE,
                    severity=SeverityLevel.LOW,
                    message="Debug print statement in production code",
                    suggestion="Remove or replace with proper logging"
                ))
        
        elif environment == 'web':
            # Web-specific checks
            if 'input' in code and 'htmlspecialchars' not in code:
                issues.append(SafetyIssue(
                    category=IssueCategory.CODE_INJECTION,
                    severity=SeverityLevel.HIGH,
                    message="Missing XSS protection",
                    suggestion="Escape output with htmlspecialchars() or template engine",
                    cwe_id="CWE-79"
                ))
        
        return issues

    def _get_code_snippet(self, code: str, line_number: int, context_lines: int = 2) -> str:
        """Extract code snippet around a specific line."""
        lines = code.split('\n')
        start = max(0, line_number - context_lines - 1)
        end = min(len(lines), line_number + context_lines)
        
        snippet = []
        for i in range(start, end):
            prefix = '-> ' if i == line_number - 1 else '   '
            snippet.append(f"{prefix}{i+1}: {lines[i]}")
        
        return '\n'.join(snippet)

    def _get_suggestion_for_pattern(self, pattern: str) -> str:
        """Get remediation suggestion for a dangerous pattern."""
        suggestions = {
            'os.system': 'Use subprocess.run() with argument lists instead of shell=True',
            'eval': 'Avoid eval() - use safer alternatives like ast.literal_eval() for simple data',
            'exec': 'Avoid exec() - use modules and functions instead',
            'pickle': 'Use JSON or other serialization formats for untrusted data',
            'yaml.load': 'Use yaml.safe_load() instead',
            'input.*exec': 'Never pass user input to exec()',
        }
        
        for key, suggestion in suggestions.items():
            if key in pattern.lower():
                return suggestion
        
        return "Replace with a secure alternative and validate all inputs"

    def _get_cwe_for_pattern(self, pattern: str) -> Optional[str]:
        """Get CWE ID for a dangerous pattern."""
        cwe_mapping = {
            'os.system': 'CWE-78',
            'subprocess': 'CWE-78',
            'eval': 'CWE-95',
            'exec': 'CWE-94',
            'pickle': 'CWE-502',
            'yaml.load': 'CWE-502',
            'marshal': 'CWE-502',
            'input.*exec': 'CWE-94',
            'strcpy': 'CWE-120',
            'printf.*%n': 'CWE-134',
        }
        
        for key, cwe_id in cwe_mapping.items():
            if key in pattern.lower():
                return cwe_id
        
        return None

    def _deduplicate_issues(self, issues: List[SafetyIssue]) -> List[SafetyIssue]:
        """Remove duplicate issues based on message and line number."""
        seen = set()
        unique_issues = []
        
        for issue in issues:
            key = (issue.message, issue.line_number)
            if key not in seen:
                seen.add(key)
                unique_issues.append(issue)
        
        return unique_issues

    def sanitize_code(
        self, 
        code: str, 
        language: str,
        aggressive: bool = False
    ) -> Tuple[str, List[str]]:
        """
        Attempt to sanitize potentially unsafe code with tracking.

        Args:
            code: Original code
            language: Programming language
            aggressive: Remove all suspicious patterns (may break functionality)

        Returns:
            Tuple of (sanitized_code, list_of_removed_patterns)
        """
        removed_patterns = []
        sanitized = code
        
        if language.lower() == 'python':
            # Remove dangerous imports
            for imp in ['socket', 'ftplib', 'smtplib', 'telnetlib']:
                pattern = rf'^.*(import\s+{imp}|from\s+{imp}\s+import).*$'
                if re.search(pattern, sanitized, re.MULTILINE):
                    sanitized = re.sub(pattern, f'# SAFETY: Removed dangerous import: {imp}', sanitized, flags=re.MULTILINE)
                    removed_patterns.append(f"Removed import: {imp}")
            
            # Replace dangerous functions
            dangerous_funcs = {
                'eval': 'ast.literal_eval',  # Note: only works for literals
                'exec': '# EXEC_REMOVED',
                'os.system': 'subprocess.run',
            }
            
            for old, new in dangerous_funcs.items():
                if old in sanitized:
                    if aggressive:
                        sanitized = sanitized.replace(old, new)
                        removed_patterns.append(f"Replaced {old}() with {new}")
                    else:
                        sanitized = sanitized.replace(old, f'# WARNING: {old}() used here\n            #{old}')
                        removed_patterns.append(f"Commented out {old}()")
        
        return sanitized, removed_patterns

    def generate_safety_report(self, issues: List[SafetyIssue]) -> Dict:
        """Generate a comprehensive safety report."""
        report = {
            'summary': {
                'total_issues': len(issues),
                'by_severity': {},
                'by_category': {},
                'critical_count': 0,
                'high_count': 0,
                'medium_count': 0,
                'low_count': 0,
                'info_count': 0,
            },
            'issues': [],
            'recommendations': [],
            'generated_at': datetime.now().isoformat(),
        }
        
        # Count by severity
        for issue in issues:
            severity = issue.severity.value
            report['summary']['by_severity'][severity] = report['summary']['by_severity'].get(severity, 0) + 1
            report['summary'][f'{severity}_count'] = report['summary'].get(f'{severity}_count', 0) + 1
            
            # Count by category
            category = issue.category.value
            report['summary']['by_category'][category] = report['summary']['by_category'].get(category, 0) + 1
            
            # Add issue details
            report['issues'].append({
                'severity': issue.severity.value,
                'category': issue.category.value,
                'message': issue.message,
                'line_number': issue.line_number,
                'suggestion': issue.suggestion,
                'cwe_id': issue.cwe_id,
            })
            
            # Add unique recommendations
            if issue.suggestion and issue.suggestion not in report['recommendations']:
                report['recommendations'].append(issue.suggestion)
        
        return report

    def generate_safety_warning(self, issues: List[SafetyIssue]) -> str:
        """Generate a formatted safety warning message."""
        if not issues:
            return ""
        
        # Group issues by severity
        critical = [i for i in issues if i.severity == SeverityLevel.CRITICAL]
        high = [i for i in issues if i.severity == SeverityLevel.HIGH]
        medium = [i for i in issues if i.severity == SeverityLevel.MEDIUM]
        low = [i for i in issues if i.severity == SeverityLevel.LOW]
        
        warning = []
        warning.append("⚠️  SAFETY WARNING:")
        warning.append("=" * 60)
        
        if critical:
            warning.append(f"\n🔴 CRITICAL ISSUES ({len(critical)}):")
            warning.append("These issues MUST be fixed before deployment:")
            for issue in critical[:5]:  # Show top 5
                line_info = f" (line {issue.line_number})" if issue.line_number else ""
                warning.append(f"  • [{issue.category.value}] {issue.message}{line_info}")
                if issue.suggestion:
                    warning.append(f"    Suggestion: {issue.suggestion}")
        
        if high:
            warning.append(f"\n🟠 HIGH PRIORITY ({len(high)}):")
            for issue in high[:5]:
                line_info = f" (line {issue.line_number})" if issue.line_number else ""
                warning.append(f"  • {issue.message}{line_info}")
        
        if medium:
            warning.append(f"\n🟡 MEDIUM PRIORITY ({len(medium)}):")
            for issue in medium[:5]:
                warning.append(f"  • {issue.message}")
        
        warning.append("\n📋 RECOMMENDATIONS:")
        recommendations = list(set([i.suggestion for i in issues[:10] if i.suggestion]))
        for i, rec in enumerate(recommendations[:5], 1):
            warning.append(f"  {i}. {rec}")
        
        warning.append("\n⚠️  Please review the code carefully before execution.")
        warning.append("Use this code only in a safe, isolated environment.")
        
        return '\n'.join(warning)

    def is_whitelisted(self, code: str, language: str) -> bool:
        """Check if code is whitelisted for certain operations."""
        # Add whitelist logic here
        return False
        if environment == 'production':


