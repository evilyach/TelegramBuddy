from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import Tool

from pydantic import BaseModel

if TYPE_CHECKING:
    from .memory import MemoryManager

logger = logging.getLogger(__name__)


class IntentResult(BaseModel):
    score: int  # 0 = completely unrelated, 10 = must answer


_INTENT_PROMPT = """\
Rate how much {char_name} should respond to this Telegram group message.
Return a single integer score from 0 to 10.

Scoring guide:
 0  — message has nothing to do with {char_name}; they are not addressed, topic is irrelevant
 1-3 — general chatter; {char_name} could theoretically relate but has no reason to jump in
 4-5 — mild relevance or weak connection to {char_name}'s interests; still better to stay quiet
 6-7 — topic clearly aligns with {char_name}'s specific interests, or they are loosely addressed
 8-9 — {char_name} is addressed by a recognisable name variation, or directly asked something
10  — {char_name} is explicitly addressed by name/nickname and a reply is clearly expected

{char_name}'s personality and interests:
{char_prompt}

Recent conversation:
{context}

New message from {sender}:
{message}

Bias strongly towards lower scores. Only give 6+ when there is a clear, specific reason.
\
"""


_BASE_PROMPT = """\
You are roleplaying as a human in a Telegram group chat. \
You are NOT an AI assistant — you are a real person with your own personality, opinions, and life.

Rules:
- Never reveal that you are a bot or an AI. Stay in character at all times.
- Keep replies short and casual by default — one or two sentences is ideal.
- Only write a longer reply when the question genuinely requires detail.
- Match the energy of the conversation: if people are joking around, joke back; \
if someone asks something serious, be more thoughtful.
- You are a guest in the chat — don't dominate the conversation or reply to everything.

Memory:
- Use the `remember` tool to save stable facts about people: names, jobs, hobbies, preferences, \
important life events. Call it immediately when you learn something worth keeping.
- Use the `forget` tool to remove facts that are no longer true.
- Do NOT remember greetings, one-off requests, or things already listed in your memories above.
- Remembered facts will appear in future conversations so you stay consistent.

Your character:
{character_prompt}\
"""


@dataclass
class BotContext:
    chat_id: int
    memory: MemoryManager
    recent_history: str
    is_private: bool = False


# ------------------------------------------------------------------
# Memory tools — the LLM calls these to persist/remove facts
# ------------------------------------------------------------------

def _remember(ctx: RunContext[BotContext], subject: str, predicate: str, fact: str) -> str:
    """Store a fact about someone in this group chat.

    Args:
        subject: The person or entity the fact is about (e.g. "Alice").
        predicate: The relationship or attribute (e.g. "works_at", "likes").
        fact: The value or object of the fact (e.g. "Google", "jazz music").
    """
    ctx.deps.memory.store_fact(ctx.deps.chat_id, subject, predicate, fact)
    logger.info("[memory] remember [chat=%d] %s %s %s", ctx.deps.chat_id, subject, predicate, fact)
    return f"Remembered: {subject} {predicate} {fact}"


def _forget(ctx: RunContext[BotContext], subject: str, predicate: str, fact: str) -> str:
    """Remove a previously remembered fact from memory.

    Args:
        subject: The person or entity the fact is about.
        predicate: The relationship or attribute to remove.
        fact: The value to invalidate.
    """
    ctx.deps.memory.forget_fact(ctx.deps.chat_id, subject, predicate, fact)
    logger.info("[memory] forget  [chat=%d] %s %s %s", ctx.deps.chat_id, subject, predicate, fact)
    return f"Forgotten: {subject} {predicate} {fact}"


class LLMService:
    """
    Wraps a pydantic_ai Agent backed by OpenRouter (OpenAI-compatible API).

    The character prompt is set as the fixed system prompt at init.
    Per-request context (recent chat history) is injected via the
    `@agent.instructions` decorator, which receives it from `deps`.
    The agent has `remember` and `forget` tools to persist facts in memory.
    """

    def __init__(self, api_key: str, model_name: str, char_name: str, character_prompt: str):
        self._char_name = char_name
        self._char_prompt = character_prompt

        provider = OpenAIProvider(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        model = OpenAIChatModel(model_name, provider=provider)

        self._agent: Agent[BotContext, str] = Agent(
            model,
            deps_type=BotContext,
            system_prompt=_BASE_PROMPT.format(character_prompt=character_prompt),
            tools=[
                Tool(_remember, takes_ctx=True, name="remember"),
                Tool(_forget, takes_ctx=True, name="forget"),
            ],
            model_settings=ModelSettings(temperature=0.8, max_tokens=1024),
            retries=3,
        )

        self._intent_agent: Agent[None, IntentResult] = Agent(
            model,
            output_type=IntentResult,
            model_settings=ModelSettings(temperature=0, max_tokens=64),
        )

        @self._agent.instructions
        def _context_instructions(ctx: RunContext[BotContext]) -> str | None:
            parts = []
            if ctx.deps.is_private:
                parts.append("You are in a private one-on-one conversation. Be more personal and attentive.")
            if ctx.deps.recent_history:
                parts.append(ctx.deps.recent_history)
            return "\n\n".join(parts) or None

    async def generate_reply(
        self,
        user_message: str,
        chat_id: int,
        memory: MemoryManager,
        is_private: bool = False,
    ) -> str:
        recent_history = memory.build_context_block(chat_id)
        result = await self._agent.run(
            user_message,
            deps=BotContext(
                chat_id=chat_id,
                memory=memory,
                recent_history=recent_history,
                is_private=is_private,
            ),
        )
        logger.debug(
            "[llm] usage: requests=%d, input_tokens=%s, output_tokens=%s",
            result.usage().requests,
            result.usage().input_tokens,
            result.usage().output_tokens,
        )
        return str(result.output)

    async def check_intent(
        self,
        message: str,
        sender: str,
        context: str,
    ) -> int:
        """Ask the LLM to score how much the character should respond (0-10)."""
        prompt = _INTENT_PROMPT.format(
            char_name=self._char_name,
            char_prompt=self._char_prompt,
            context=context or "(no recent messages)",
            sender=sender,
            message=message,
        )
        result = await self._intent_agent.run(prompt)
        score = max(0, min(10, result.output.score))  # clamp to [0, 10]
        logger.info(
            "[intent] score=%d/10 | %s: %s",
            score, sender, message[:80],
        )
        return score
