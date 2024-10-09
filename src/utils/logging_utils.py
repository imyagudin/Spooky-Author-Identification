import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logging(log_file='log.log', log_level=logging.INFO, max_bytes=10*1024*1024, backup_count=5):
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
    handler.setLevel(log_level)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(log_level)
    logger.addHandler(handler)

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(handler)

def get_logger(name):
    return logging.getLogger(name)
