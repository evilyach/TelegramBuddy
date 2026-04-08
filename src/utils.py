import random


def word_count(text: str) -> int:
    return len(text.split())


def should_respond_probabilistically(probability: float) -> bool:
    return random.random() < probability


def is_mentioned(message_text: str, char_name: str, bot_username: str) -> bool:
    text_lower = message_text.lower()
    return (
        char_name.lower() in text_lower
        or f"@{bot_username.lower()}" in text_lower
    )
