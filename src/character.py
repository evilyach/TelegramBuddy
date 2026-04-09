import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.enums import ChatAction, ChatType
from aiogram.types import FSInputFile, Message

from .config import AppConfig, CharacterConfig
from .llm import LLMService
from .memory import MemoryManager
from .tts import TTSService
from .utils import is_mentioned, word_count

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
            char_name=char_cfg.name,
            character_prompt=char_cfg.prompt,
        )
        self._tts = TTSService(
            ref_audio=str(char_cfg.ref_audio),
            ref_text=char_cfg.ref_text,
            device=app_cfg.tts_device,
            denoise=char_cfg.tts_denoise,
        )
        self._bot_username: str = ""   # populated in run()
        self._bot_id: int = 0          # populated in run()

        self._register_handlers()

    def _register_handlers(self) -> None:
        @self._dp.message()
        async def on_message(message: Message) -> None:
            await self._handle_message(message)

    async def _keep_action(self, chat_id: int, action: ChatAction) -> None:
        """Repeatedly send a chat action every 4 s until cancelled."""
        while True:
            try:
                await self._bot.send_chat_action(chat_id=chat_id, action=action)
            except Exception:
                pass
            await asyncio.sleep(4)

    def _is_reply_to_us(self, message: Message) -> bool:
        """True if this message is a reply to one of our own messages."""
        reply = message.reply_to_message
        return (
            reply is not None
            and reply.from_user is not None
            and reply.from_user.id == self._bot_id
        )

    async def _handle_message(self, message: Message) -> None:
        if not message.text or not message.from_user:
            return

        chat_id = message.chat.id
        chat_type = message.chat.type
        sender = message.from_user.full_name
        text = message.text
        is_private = chat_type == ChatType.PRIVATE

        logger.debug(
            "[%s] [chat=%d %s] %s: %s",
            self._name, chat_id, chat_type, sender, text[:120],
        )

        # Always record the incoming message for context
        self._memory.add_message(chat_id, sender, text)

        # Decide whether to reply
        if is_private:
            logger.debug("[%s] [chat=%d] responding — private chat", self._name, chat_id)
        else:
            # Tier 1 — 100%: @handle, reply to our message, or name directly in text
            if self._is_reply_to_us(message):
                logger.debug("[%s] [chat=%d] responding (100%%) — reply to our message", self._name, chat_id)
            elif f"@{self._bot_username.lower()}" in text.lower():
                logger.debug("[%s] [chat=%d] responding (100%%) — @handle mention", self._name, chat_id)
            elif is_mentioned(text, [self._name, self._cfg.name], self._bot_username):
                logger.debug("[%s] [chat=%d] responding (100%%) — name mention", self._name, chat_id)
            else:
                # Tier 2 — LLM intent score + probability roll
                context = self._memory.build_context_block(chat_id)
                score = await self._llm.check_intent(text, sender, context)
                threshold = self._cfg.answer_threshold
                if score < threshold:
                    logger.debug(
                        "[%s] [chat=%d] skipping — intent score %d/10 < threshold %d",
                        self._name, chat_id, score, threshold,
                    )
                    return
                logger.debug(
                    "[%s] [chat=%d] responding — intent score %d/10 >= threshold %d",
                    self._name, chat_id, score, threshold,
                )

        logger.info("[%s] [chat=%d] generating reply to: %s", self._name, chat_id, text[:80])

        typing_task = asyncio.create_task(self._keep_action(chat_id, ChatAction.TYPING))
        try:
            reply_text = await self._llm.generate_reply(
                text, chat_id, self._memory, is_private=is_private
            )
        finally:
            typing_task.cancel()

        wc = word_count(reply_text)
        if wc > self._cfg.voice_word_count_threshold:
            logger.info(
                "[%s] [chat=%d] sending voice (%d words): %s",
                self._name, chat_id, wc, reply_text[:80],
            )
            logger.debug("[%s] [chat=%d] synthesizing TTS...", self._name, chat_id)
            recording_task = asyncio.create_task(
                self._keep_action(chat_id, ChatAction.RECORD_VOICE)
            )
            try:
                audio_path = await self._tts.synthesize(reply_text)
            finally:
                recording_task.cancel()
            try:
                await message.reply_voice(FSInputFile(audio_path))
                logger.debug("[%s] [chat=%d] voice sent", self._name, chat_id)
            finally:
                audio_path.unlink(missing_ok=True)
        else:
            logger.info(
                "[%s] [chat=%d] sending text (%d words): %s",
                self._name, chat_id, wc, reply_text[:80],
            )
            await message.reply(reply_text)

    async def run(self) -> None:
        me = await self._bot.get_me()
        self._bot_username = me.username or ""
        self._bot_id = me.id
        logger.info(
            "Character '%s' (%s) started as @%s id=%d (token ends ...%s)",
            self._name, self._cfg.name, self._bot_username, self._bot_id,
            self._cfg.tg_bot_token[-6:],
        )
        await self._dp.start_polling(self._bot)
