from ultralytics.utils import LOGGER


class Logger:
    def __init__(self, verbose):
        self.__verbose = verbose

    def log(self, msg):
        if self.__verbose:
            LOGGER.info(msg)

    @staticmethod
    def warn(msg):
        LOGGER.warning(msg)
