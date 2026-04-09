import asyncio
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


async def async_main() -> None:
    from src.character import CharacterBot
    from src.config import load_config

    app_cfg = load_config("config.yaml")
    bots = [
        CharacterBot(char_name=char_name, char_cfg=char_cfg, app_cfg=app_cfg)
        for char_name, char_cfg in app_cfg.characters.items()
    ]
    await asyncio.gather(*(bot.run() for bot in bots))


def main() -> None:
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        logging.info("Shutting down...")


if __name__ == "__main__":
    main()
