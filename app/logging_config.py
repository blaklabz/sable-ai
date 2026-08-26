import logging
import os
from pathlib import Path


LOG_DIR = Path(
    os.getenv(
        "SABLE_LOG_DIR",
        "/var/log/sable",
    )
)


LOG_FORMAT = (
    "%(asctime)s | "
    "%(levelname)-8s | "
    "%(name)s | "
    "%(message)s"
)


def _make_file_handler(
    filename: str,
) -> logging.FileHandler:
    handler = logging.FileHandler(
        LOG_DIR / filename,
        mode="a",
        encoding="utf-8",
    )

    handler.setFormatter(
        logging.Formatter(
            LOG_FORMAT
        )
    )

    return handler


def setup_logging() -> None:
    """
    Configure Sable logging.

    Log files are append-only from Python's perspective.
    Linux logrotate is responsible for rotation,
    compression, and local retention.
    """

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    root_logger = (
        logging.getLogger()
    )

    root_logger.setLevel(
        logging.INFO
    )

    if not root_logger.handlers:
        console_handler = (
            logging.StreamHandler()
        )

        console_handler.setFormatter(
            logging.Formatter(
                LOG_FORMAT
            )
        )

        root_logger.addHandler(
            console_handler
        )

        root_logger.addHandler(
            _make_file_handler(
                "sable.log"
            )
        )

    # -------------------------------------------------------------
    # Dedicated component logs.
    #
    # propagate=True means these events also appear in sable.log,
    # giving us both component-specific files and one master timeline.
    # -------------------------------------------------------------

    component_logs = {
        "sable.api": "api.log",
        "sable.llm": "llm.log",
        "sable.memory": "memory.log",
        "sable.prompt": "prompt.log",
        "sable.homework": "homework.log",
        "sable.haiku": "haiku.log",
    }

    for logger_name, filename in (
        component_logs.items()
    ):
        logger = logging.getLogger(
            logger_name
        )

        logger.setLevel(
            logging.INFO
        )

        if not logger.handlers:
            logger.addHandler(
                _make_file_handler(
                    filename
                )
            )

        logger.propagate = True


api_logger = logging.getLogger(
    "sable.api"
)

llm_logger = logging.getLogger(
    "sable.llm"
)

memory_logger = logging.getLogger(
    "sable.memory"
)

prompt_logger = logging.getLogger(
    "sable.prompt"
)

homework_logger = logging.getLogger(
    "sable.homework"
)

haiku_logger = logging.getLogger(
    "sable.haiku"
)
