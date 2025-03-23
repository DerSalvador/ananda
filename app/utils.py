import logging
import os
from sys import stdout


cur_logger = None
def get_logger(module = "api-engineer"):
    global cur_logger
    if cur_logger:
        return cur_logger
    logging.basicConfig(format="%(asctime)s %(levelname)-2s %(filename)s:%(funcName)s:%(lineno)d %(message)s")
    logger = logging.getLogger(module)
    log_level = logging.INFO
    logger.setLevel(log_level)

    logFormatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(filename)s:%(funcName)s %(message)s"
    )
    consoleHandler = logging.StreamHandler(stdout)
    consoleHandler.setFormatter(logFormatter)

    logpath = os.getenv("LOG_PATH", "/var/log/cryptoendpoints.log")
    fileHandler = logging.FileHandler(filename=logpath)
    fileHandler.setFormatter(logFormatter)
    fileHandler.setLevel(log_level)
    logger.addHandler(fileHandler)

    cur_logger = logger
    return logger


