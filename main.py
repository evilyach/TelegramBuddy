import asyncio
import logging
import signal

from src.utils import BotNameFilter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(bot_name)s] %(levelname)s %(name)s: %(message)s",
)
_bot_filter = BotNameFilter()
for _handler in logging.getLogger().handlers:
    _handler.addFilter(_bot_filter)


async def async_main() -> None:
    from src.character import CharacterBot
    from src.config import load_config
    from src.message_cache import MessageCache

    app_cfg = load_config("config.yaml")
    cache = MessageCache("message_cache.db")
    bots = [
        CharacterBot(char_name=char_name, char_cfg=char_cfg, app_cfg=app_cfg, message_cache=cache)
        for char_name, char_cfg in app_cfg.characters.items()
    ]

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _request_stop() -> None:
        if not stop_event.is_set():
            logging.info("Shutdown requested, stopping bots...")
            stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _request_stop)

    bot_tasks = [asyncio.create_task(bot.run(), name=name) for name, bot in zip(app_cfg.characters, bots)]

    # Wait until a stop signal arrives or all tasks finish on their own
    stop_task = asyncio.create_task(stop_event.wait())
    done, _ = await asyncio.wait([stop_task, *bot_tasks], return_when=asyncio.FIRST_COMPLETED)

    # If a bot task finished first (e.g. crashed), still shut everything down cleanly
    if stop_task not in done:
        stop_task.cancel()

    for bot in bots:
        await bot.stop()

    await asyncio.gather(*bot_tasks, return_exceptions=True)
    logging.info("All bots stopped.")


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
