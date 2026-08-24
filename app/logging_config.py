import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


def _make_file_handler(filename: str) -> RotatingFileHandler:
    handler = RotatingFileHandler(
        LOG_DIR / filename,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )

    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )
    )

    return handler


def setup_logging() -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    if not root_logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
            )
        )

        root_logger.addHandler(console_handler)
        root_logger.addHandler(
            _make_file_handler("sable.log")
        )

    homework = logging.getLogger("sable.homework")
    homework.setLevel(logging.INFO)

    if not homework.handlers:
        homework.addHandler(
            _make_file_handler("homework.log")
        )

    homework.propagate = True

    haiku = logging.getLogger("sable.haiku")
    haiku.setLevel(logging.INFO)

    if not haiku.handlers:
        haiku.addHandler(
            _make_file_handler("haiku.log")
        )

    haiku.propagate = True


api_logger = logging.getLogger("sable.api")
llm_logger = logging.getLogger("sable.llm")
memory_logger = logging.getLogger("sable.memory")
prompt_logger = logging.getLogger("sable.prompt")

homework_logger = logging.getLogger("sable.homework")
haiku_logger = logging.getLogger("sable.haiku")
