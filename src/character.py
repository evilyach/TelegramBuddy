import asyncio
import io
import logging

from aiogram import Bot, Dispatcher
from aiogram.enums import ChatAction, ChatType
from aiogram.types import FSInputFile, Message, ReactionTypeEmoji

from .config import AppConfig, CharacterConfig
from .llm import LLMService
from .memory import MemoryManager
from .message_cache import MessageCache
from .stt import STTService
from .tts import TTSService
from .utils import bot_name_var, is_mentioned, word_count

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
        message_cache: MessageCache,
    ):
        self._name = char_name
        self._cfg = char_cfg
        self._app_cfg = app_cfg
        self._bot = Bot(token=char_cfg.tg_bot_token)
        self._dp = Dispatcher()
        self._cache = message_cache
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
            runpod_api_key=app_cfg.runpod_api_key,
            runpod_endpoint_id=app_cfg.runpod_tts_endpoint,
        )
        self._stt = STTService(
            api_key=app_cfg.openrouter_api_key,
            model=app_cfg.stt_model,
        )
        self._bot_username: str = ""   # populated in run()
        self._bot_id: int = 0          # populated in run()
        self._chat_locks: dict[int, asyncio.Lock] = {}

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

    async def _download_image(self, message: Message) -> tuple[bytes, str] | None:
        """Return (image_bytes, media_type) for photo or image document, or None."""
        if message.photo:
            # Telegram sends multiple sizes; take the largest one
            photo = message.photo[-1]
            buf = io.BytesIO()
            await self._bot.download(photo.file_id, destination=buf)
            return buf.getvalue(), "image/jpeg"
        if message.document and message.document.mime_type and message.document.mime_type.startswith("image/"):
            buf = io.BytesIO()
            await self._bot.download(message.document.file_id, destination=buf)
            return buf.getvalue(), message.document.mime_type
        return None

    async def _try_react(self, message: Message, text: str, sender: str, chat_id: int) -> None:
        """Check react intent and send emoji reaction if score meets threshold."""
        try:
            context = self._memory.build_context_block(chat_id)
            score, emoji = await self._llm.check_react(text, sender, context)
            threshold = self._cfg.react_threshold
            if threshold is not None and score >= threshold:
                logger.info(
                    "[%s] [chat=%d] reacting with %s — react score %d/10 >= threshold %d",
                    self._name, chat_id, emoji, score, threshold,
                )
                await message.react([ReactionTypeEmoji(emoji=emoji)])
            else:
                logger.debug(
                    "[%s] [chat=%d] no reaction — react score %d/10 < threshold %d",
                    self._name, chat_id, score, threshold,
                )
        except Exception as e:
            logger.warning("[%s] [chat=%d] react check failed: %s", self._name, chat_id, e)

    def _chat_lock(self, chat_id: int) -> asyncio.Lock:
        if chat_id not in self._chat_locks:
            self._chat_locks[chat_id] = asyncio.Lock()
        return self._chat_locks[chat_id]

    async def _handle_message(self, message: Message) -> None:
        if not message.from_user:
            return

        chat_id = message.chat.id
        async with self._chat_lock(chat_id):
            await self._process_message(message)

    async def _process_message(self, message: Message) -> None:
        assert message.from_user is not None  # guaranteed by _handle_message

        image: bytes | None = None
        image_media_type: str = "image/jpeg"

        # Resolve text: prefer actual text, fall back to caption (for photos) or cached voice transcript
        if message.text:
            text = message.text
        elif message.photo or (message.document and message.document.mime_type and message.document.mime_type.startswith("image/")):
            text = message.caption or "[sent an image]"
            img = await self._download_image(message)
            if img:
                image, image_media_type = img
                logger.debug(
                    "[%s] [chat=%d] received image (%s, %d bytes), caption: %s",
                    self._name, message.chat.id, image_media_type, len(image), text[:80],
                )
        elif message.voice:
            if not self._app_cfg.stt_enabled:
                return
            if message.from_user.id in self._app_cfg.stt_user_blacklist:
                logger.debug(
                    "[%s] [chat=%d] skipping STT — user %d is blacklisted",
                    self._name, message.chat.id, message.from_user.id,
                )
                return
            lock = self._cache.transcription_lock(message.chat.id, message.message_id)
            async with lock:
                text = self._cache.lookup(message.chat.id, message.message_id)
                if text is None:
                    logger.info(
                        "[%s] [chat=%d] transcribing voice msg=%d via STT",
                        self._name, message.chat.id, message.message_id,
                    )
                    try:
                        buf = io.BytesIO()
                        await self._bot.download(message.voice.file_id, destination=buf)
                        text = await self._stt.transcribe(buf.getvalue())
                        self._cache.store(message.chat.id, message.message_id, text)
                        logger.info(
                            "[%s] [chat=%d] transcribed voice msg=%d: %s",
                            self._name, message.chat.id, message.message_id, text[:80],
                        )
                    except Exception as e:
                        logger.warning(
                            "[%s] [chat=%d] STT failed for voice msg=%d: %s",
                            self._name, message.chat.id, message.message_id, e,
                        )
                        return
                else:
                    logger.info(
                        "[%s] [chat=%d] resolved voice msg=%d from cache: %s",
                        self._name, message.chat.id, message.message_id, text[:80],
                    )
        else:
            return

        chat_id = message.chat.id
        chat_type = message.chat.type
        sender = message.from_user.full_name
        sender_is_bot = message.from_user.is_bot
        is_private = chat_type == ChatType.PRIVATE

        logger.debug(
            "[%s] [chat=%d %s] %s: %s",
            self._name, chat_id, chat_type, sender, text[:120],
        )

        # Always record the incoming message for context
        self._memory.add_message(chat_id, sender, text)
        bot_streak = self._cache.record_message(chat_id, is_bot=sender_is_bot)

        # Hard stop: too many consecutive bot messages — break the loop
        if sender_is_bot and bot_streak >= self._app_cfg.bot_consecutive_limit:
            logger.debug(
                "[%s] [chat=%d] skipping — bot streak %d >= limit %d",
                self._name, chat_id, bot_streak, self._app_cfg.bot_consecutive_limit,
            )
            return

        # Fire react check as a background task (independent of reply decision)
        if self._cfg.react_threshold is not None:
            asyncio.create_task(self._try_react(message, text, sender, chat_id))

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
                # Tier 2 — LLM intent score; raise threshold when talking to another bot
                context = self._memory.build_context_block(chat_id)
                score = await self._llm.check_intent(text, sender, context)
                threshold = 9 if sender_is_bot else self._cfg.answer_threshold
                if score < threshold:
                    logger.debug(
                        "[%s] [chat=%d] skipping — intent score %d/10 < threshold %d%s",
                        self._name, chat_id, score, threshold,
                        " (bot sender)" if sender_is_bot else "",
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
                text, sender, chat_id, self._bot, self._memory,
                is_private=is_private,
                image=image, image_media_type=image_media_type,
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
                sent = await message.reply_voice(FSInputFile(audio_path))
                self._cache.store(sent.chat.id, sent.message_id, reply_text)
                logger.debug("[%s] [chat=%d] voice sent (msg=%d cached)", self._name, chat_id, sent.message_id)
            finally:
                audio_path.unlink(missing_ok=True)
        else:
            logger.info(
                "[%s] [chat=%d] sending text (%d words): %s",
                self._name, chat_id, wc, reply_text[:80],
            )
            await message.reply(reply_text)

    async def stop(self) -> None:
        await self._dp.stop_polling()

    async def run(self) -> None:
        bot_name_var.set(self._name)
        me = await self._bot.get_me()
        self._bot_username = me.username or ""
        self._bot_id = me.id
        logger.info(
            "Character '%s' (%s) started as @%s id=%d (token ends ...%s)",
            self._name, self._cfg.name, self._bot_username, self._bot_id,
            self._cfg.tg_bot_token[-6:],
        )
        await self._dp.start_polling(self._bot)
