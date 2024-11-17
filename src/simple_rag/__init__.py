import os
import logging
from logging.handlers import RotatingFileHandler
from simple_rag.common import common
from dotenv import load_dotenv

load_dotenv()


def setup_logging():
    log_file = os.path.join(common.get_log_dir(), "app.log")
    log_level_str = os.getenv('LOG_LEVEL', 'INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=20, encoding='utf-8')  # 10MB
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s',
        handlers=[
            handler,
            logging.StreamHandler()
        ]
    )
    
    lib_with_decreased_logging = [
        'openai',
        'urllib3',
        "httpcore",
        "httpx",
    ]
    for lib in lib_with_decreased_logging:
        logging.getLogger(lib).setLevel(logging.INFO)
        
    # Separate logger for openai debug logs
    openai_log_file = os.path.join(common.get_log_dir(), "openai_debug.log")
    openai_handler = RotatingFileHandler(openai_log_file, maxBytes=10 * 1024 * 1024, backupCount=20, encoding='utf-8')  # 10MB
    openai_handler.setLevel(logging.DEBUG)
    openai_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s'))
    
    openai_logger = logging.getLogger('openai')
    openai_logger.setLevel(logging.DEBUG)
    openai_logger.addHandler(openai_handler)


setup_logging()
