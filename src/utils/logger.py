import logging
import time
import os
from functools import wraps
from logging.handlers import RotatingFileHandler

# Ensure the logs folder exists
log_folder = "logs"
os.makedirs(log_folder, exist_ok=True)
log_file = os.path.join(log_folder, "app.log")

# Configure file logging with rotation (max 5 MB per file, keep 3 backups)
handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
handler.setFormatter(formatter)

logger_instance = logging.getLogger("AppLogger")
logger_instance.setLevel(logging.INFO)
logger_instance.addHandler(handler)

def logger(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            logger_instance.info(f"Started function '{func.__name__}' with args={args}, kwargs={kwargs}")
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            logger_instance.info(f"Finished function '{func.__name__}' in {elapsed_time:.4f} seconds")
            return result
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger_instance.error(f"Error in function '{func.__name__}' after {elapsed_time:.4f} seconds: {e}", exc_info=True)
            return f"Error in function '{func.__name__}' after {elapsed_time:.4f} seconds: {e}"
              # Keep raising so calling code knows
    return wrapper
