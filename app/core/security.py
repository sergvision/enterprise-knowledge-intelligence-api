
"""
Security utilities for Enterprise Knowledge Intelligence V2.

Provides secure token generation and constant-time token verification.
"""

import hmac
import secrets


def generate_secure_token(length: int = 32) -> str:
    """
    Generate a cryptographically secure URL-safe token.

    Args:
        length: Number of random bytes used to generate the token.

    Returns:
        A URL-safe secret token.
    """
    if length < 16:
        raise ValueError("Token length must be at least 16 bytes.")

    return secrets.token_urlsafe(length)


def verify_token(
    provided_token: str,
    expected_token: str
) -> bool:
    """
    Verify a token using constant-time comparison.

    Args:
        provided_token: Token received from the requester.
        expected_token: Expected secret token.

    Returns:
        True if both tokens match, otherwise False.
    """
    if not provided_token or not expected_token:
        return False

    return hmac.compare_digest(
        provided_token.encode("utf-8"),
        expected_token.encode("utf-8")
    )
