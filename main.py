import asyncio
import logging
import multiprocessing

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(processName)s] %(levelname)s %(name)s: %(message)s",
)


def run_character(char_name: str, app_cfg_dict: dict) -> None:
    """
    Subprocess entry point. Reconstructs AppConfig from a plain dict because
    macOS uses `spawn` for multiprocessing — the parent's objects are not
    inherited, so we must re-create them from picklable data.
    """
    from src.character import CharacterBot
    from src.config import AppConfig

    app_cfg = AppConfig(**app_cfg_dict)
    char_cfg = app_cfg.characters[char_name]
    bot = CharacterBot(char_name=char_name, char_cfg=char_cfg, app_cfg=app_cfg)
    asyncio.run(bot.run())


def main() -> None:
    from src.config import load_config

    app_cfg = load_config("config.yaml")
    processes: list[multiprocessing.Process] = []

    for char_name in app_cfg.characters:
        p = multiprocessing.Process(
            target=run_character,
            args=(char_name, app_cfg.model_dump()),
            name=f"bot-{char_name}",
            daemon=True,
        )
        p.start()
        logging.info("Spawned process for character '%s' (pid=%s)", char_name, p.pid)
        processes.append(p)

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        logging.info("Shutting down...")
        for p in processes:
            p.terminate()


if __name__ == "__main__":
    main()
