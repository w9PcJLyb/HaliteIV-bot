import os
import logging

FILE = "halite.log"
LEVEL = logging.DEBUG
LOGGER = None
IS_KAGGLE = False


class _FileHandler(logging.FileHandler):
    def emit(self, record):
        if IS_KAGGLE:
            print(self.format(record))
        else:
            super().emit(record)


def _get_logger():
    global LOGGER

    if not LOGGER:
        if not IS_KAGGLE:
            if os.path.exists(FILE):
                os.remove(FILE)

        LOGGER = logging.getLogger("halite")
        LOGGER.setLevel(LEVEL)
        ch = _FileHandler(FILE)
        ch.setLevel(LEVEL)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%H-%M-%S"
        )
        ch.setFormatter(formatter)
        LOGGER.addHandler(ch)

    return LOGGER


logger = _get_logger()
