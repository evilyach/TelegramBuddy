from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider


class LLMService:
    """
    Wraps a pydantic_ai Agent backed by OpenRouter (OpenAI-compatible API).

    The character_prompt is set as the agent's system prompt at init time.
    Per-request context (recent chat history, memories) is appended via the
    `instructions` kwarg on `agent.run()`, which is sent as a system message.
    """

    def __init__(self, api_key: str, model_name: str, character_prompt: str):
        provider = OpenAIProvider(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

        model = OpenAIModel(model_name, provider=provider)

        self._agent = Agent(
            model=model,
            system_prompt=character_prompt,
        )

    async def generate_reply(self, user_message: str, context_block: str) -> str:
        instructions = context_block if context_block else None
        result = await self._agent.run(user_message, instructions=instructions)
        return result.output
