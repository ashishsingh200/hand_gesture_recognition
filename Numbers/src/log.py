# log.py
import logging
import os
from datetime import datetime

# Create logs directory if not exists
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Log file name with timestamp
log_file = os.path.join(LOG_DIR, f"{datetime.now().strftime('%Y-%m-%d')}.log")

# Logging format
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG,  # Change to INFO or WARNING in production
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()  # Also log to console
    ]
)

# Function to get a logger for any module
def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger with the specified name.
    Example:
        from log import get_logger
        logger = get_logger(__name__)
        logger.info("This is a log message")
    """
    return logging.getLogger(name)
