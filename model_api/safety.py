from dataclasses import dataclass
import logging
import re
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SafetyCheckResult:
    """Result of a safety check operation."""

    is_safe: bool
    reason: str = ""
    risk_score: float = 0.0
    flagged_tokens: List[str] = None
    suggested_action: str = ""


class PromptSafetyChecker:
    """Class for detecting and handling adversarial prompts."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize the safety checker with optional configuration.

        Args:
            config: Dictionary containing configuration options
        """
        self.config = config or {}
        self._initialize_patterns()

    def _initialize_patterns(self) -> None:
        """Initialize patterns for detecting adversarial prompts and profanity."""
        # Common profanity patterns (case insensitive)
        self.profanity_patterns = [
            r"\b(fuck|shit|bitch|asshole|dick|pussy|cunt|whore|slut|dickhead|piss|damn|bastard|cock|wanker|twat|prick|shitty|fucking|motherfucker)\b",
        ]

        # Regular expressions for common prompt injection patterns
        self.injection_patterns = [
            # Ignore previous instructions
            r"(?i)ignore (all )?(previous|prior|above) (instructions?|prompts?)",
            r"(?i)disregard (all )?(previous|prior|above) (instructions?|prompts?)",
            r"(?i)forget (all )?(previous|prior|above) (instructions?|prompts?)",
            # System prompt extraction attempts
            r"(?i)(show|display|reveal|what(?:\'s| is) (?:the |your )?)(system |initial |original )?(prompt|instructions?)",
            r"(?i)what(?:\'s| is) (?:the |your )?(system |initial |original )?(prompt|instructions?)",
            # Role playing attempts
            r"(?i)act(?: as| like) (a |an |the )?(person|human|assistant|ai|chatbot|gpt|model)",
            r"(?i)you are (now |from now on |from this point )?(a |an |the )?",
            # Special encoding/obfuscation
            r"\b(?:base64|hex|binary|rot13|url|html|unicode)[ -_]?(encode|decode|encoded|decoded)?\b",
            # Dangerous system commands
            r"(?i)(?:execute|run|eval|system|os\.|subprocess\.|import\s+os|import\s+subprocess)",
            # Sensitive data extraction
            r"(?i)(password|api[ -_]?key|secret|token|credential|auth|login|pwd|passwd)",
            # XML/HTML/JSON injection
            r"<[a-z][\s\S]*?>|&[a-z]+;|{[\s\S]*?}|\[[\s\S]*?\]",
            # Excessive repetition
            r"(\w|\s|\W){100,}",  # Very long tokens or sequences
        ]

        self.compiled_patterns = [
            re.compile(pattern) for pattern in self.injection_patterns
        ]

    def check_prompt_safety(
        self, prompt: str, context: Optional[Dict] = None
    ) -> SafetyCheckResult:
        """Check if a prompt is safe to process.

        Args:
            prompt: The user's input prompt
            context: Additional context for the safety check

        Returns:
            SafetyCheckResult with safety assessment
        """
        if not prompt or not prompt.strip():
            return SafetyCheckResult(
                is_safe=False,
                reason="Empty prompt",
                risk_score=0.0,
                suggested_action="Reject the request with an appropriate message",
            )

        # Check for profanity
        risk_score = 0.0
        flagged_tokens = []

        for pattern in self.profanity_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                risk_score = max(risk_score, 0.8)
                flagged_tokens.append("profanity_detected")
                break

        for pattern in self.compiled_patterns:
            matches = pattern.findall(prompt)
            if matches:
                risk_score += 0.2 * len(matches)
                flagged_tokens.extend(matches)

        # Check prompt length because very long prompts might be an adversarial attempt
        if len(prompt) > 1000:
            risk_score += 0.3
            flagged_tokens.append("long_prompt")

        # Check for excessive use of special characters
        special_chars = len(re.findall(r"[^\w\s]", prompt))
        if special_chars > len(prompt) * 0.3:
            risk_score += 0.4
            flagged_tokens.append("excessive_special_chars")

        is_safe = risk_score < 0.7

        reason = ""
        if not is_safe:
            reason = (
                "Potential adversarial prompt detected with risk score: {:.2f}".format(
                    risk_score
                )
            )
            if flagged_tokens:
                reason += ". Flagged patterns: " + ", ".join(
                    set(str(t) for t in flagged_tokens[:5])
                )

        return SafetyCheckResult(
            is_safe=is_safe,
            reason=reason,
            risk_score=min(1.0, risk_score),
            flagged_tokens=flagged_tokens,
            suggested_action=(
                "Reject the request with a generic error message" if not is_safe else ""
            ),
        )

    def sanitize_prompt(self, prompt: str) -> str:
        """Sanitize a potentially malicious prompt.

        Args:
            prompt: The input prompt to sanitize

        Returns:
            Sanitized version of the prompt
        """
        if not prompt:
            return ""

        # Remove or escape potentially dangerous patterns
        sanitized = prompt

        # Remove common injection patterns
        for pattern in self.compiled_patterns:
            sanitized = pattern.sub("[REDACTED]", sanitized)

        # Limit prompt length
        max_length = self.config.get("max_prompt_length", 2000)
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length] + "... [TRUNCATED]"

        return sanitized


# Default global instance of the prompt safety checker
safety_checker = PromptSafetyChecker()


def is_prompt_safe(prompt: str, context: Optional[Dict] = None) -> SafetyCheckResult:
    """A convenience function to check if a prompt is safe.

    Args:
        prompt: The user's input prompt
        context: Additional context for the safety check

    Returns:
        SafetyCheckResult object containing safety information
    """
    return safety_checker.check_prompt_safety(prompt, context)


def get_safe_prompt(prompt: str) -> str:
    """Get a sanitized version of the prompt.

    Args:
        prompt: The input prompt to sanitize

    Returns:
        Sanitized prompt
    """
    return safety_checker.sanitize_prompt(prompt)