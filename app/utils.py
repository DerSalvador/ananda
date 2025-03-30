import logging
import os
from sys import stdout
from collections import deque
from time import time, sleep


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


class TimeBasedDeque:
    def __init__(self, max_age=3600):
        self.max_age = max_age
        self.queue = deque()

    def add(self, item):
        """Insert a new item with the current timestamp."""
        timestamp = time()
        self.queue.append((timestamp, item))
        self.cleanup()

    def cleanup(self):
        now = time()
        while self.queue and (now - self.queue[0][0] > self.max_age):
            self.queue.popleft()

    def get_items_last_x_seconds(self, seconds):
        """Retrieve items from the last X seconds."""
        threshold = time() - (seconds)
        return [item for t, item in self.queue if t >= threshold]
