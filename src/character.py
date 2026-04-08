import logging

from aiogram import Bot, Dispatcher
from aiogram.types import FSInputFile, Message

from .config import AppConfig, CharacterConfig
from .llm import LLMService
from .memory import MemoryManager
from .tts import TTSService
from .utils import is_mentioned, should_respond_probabilistically, word_count

logger = logging.getLogger(__name__)


class CharacterBot:
    """
    Orchestrates one Telegram character. Runs in its own process with its
    own asyncio event loop.
    """

    def __init__(
        self,
        char_name: str,
        char_cfg: CharacterConfig,
        app_cfg: AppConfig,
    ):
        self._name = char_name
        self._cfg = char_cfg
        self._bot = Bot(token=char_cfg.tg_bot_token)
        self._dp = Dispatcher()
        self._memory = MemoryManager(char_name=char_name, db_path=f"{char_name}_memory.db")
        model_name = char_cfg.openrouter_model or app_cfg.openrouter_model
        self._llm = LLMService(
            api_key=app_cfg.openrouter_api_key,
            model_name=model_name,
            character_prompt=char_cfg.prompt,
        )
        self._tts = TTSService(
            ref_audio=str(char_cfg.ref_audio),
            ref_text=char_cfg.ref_text,
        )
        self._bot_username: str = ""  # populated in run() before polling starts

        self._register_handlers()

    def _register_handlers(self) -> None:
        @self._dp.message()
        async def on_message(message: Message) -> None:
            await self._handle_message(message)

    async def _handle_message(self, message: Message) -> None:
        if not message.text or not message.from_user:
            return

        chat_id = message.chat.id
        sender = message.from_user.full_name
        text = message.text

        # Always record the incoming message for context
        self._memory.add_message(chat_id, sender, text)

        # Decide whether to reply
        mentioned = is_mentioned(text, self._name, self._bot_username)
        if not mentioned:
            if not should_respond_probabilistically(self._cfg.start_conversation_probability):
                return

        context_block = self._memory.build_context_block(chat_id)
        reply_text = await self._llm.generate_reply(text, context_block)

        if word_count(reply_text) > self._cfg.voice_word_count_threshold:
            audio_path = await self._tts.synthesize(reply_text)
            try:
                await message.reply_voice(FSInputFile(audio_path))
            finally:
                audio_path.unlink(missing_ok=True)
        else:
            await message.reply(reply_text)

    async def run(self) -> None:
        me = await self._bot.get_me()
        self._bot_username = me.username or ""
        logger.info("Character '%s' started as @%s", self._name, self._bot_username)
        await self._dp.start_polling(self._bot)
