from logging import basicConfig, FileHandler, Formatter, getLogger, Logger, NullHandler, StreamHandler

from typing import Optional

from .settings import LOG_FORMAT, LOG_FORMATTER, LOG_LEVEL, LOG_TO_FILE


default_handlers = [NullHandler()]
basicConfig(format=LOG_FORMAT, level=LOG_LEVEL, handlers=default_handlers)


def setup_logger(
    name: str,
    console_log_level: int = LOG_LEVEL,
    log_to_file: bool = LOG_TO_FILE,
    log_formatter: Optional[Formatter] = LOG_FORMATTER,
    file_handler: Optional[FileHandler] = None,
) -> Logger:
    logger = getLogger(name)

    stream_handler = StreamHandler()
    stream_handler.setLevel(console_log_level)
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)

    if log_to_file and file_handler is not None:
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)

    return logger
