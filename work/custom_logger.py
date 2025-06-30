import logging

def get_logger():
    logger = logging.getLogger('thyroid')
    logger.setLevel(logging.INFO)
    
    if (logger.hasHandlers()):
        logger.handlers.clear()
          
    stdout_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s(%(name)s) %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)
    
    file_handler = logging.FileHandler('logs/runlog.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
        
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    
    return logger
