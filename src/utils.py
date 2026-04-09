import random


def word_count(text: str) -> int:
    return len(text.split())


def should_respond_probabilistically(probability: float) -> bool:
    return random.random() < probability


def is_mentioned(message_text: str, names: list[str], bot_username: str) -> bool:
    """Return True if any of the character's names or @handle appear in the message."""
    text_lower = message_text.lower()
    if f"@{bot_username.lower()}" in text_lower:
        return True
    return any(name.lower() in text_lower for name in names if name)
