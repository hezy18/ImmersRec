import logging

logger_loss = logging.getLogger('logger_loss')

def setup_logger(log_file):
    handler = logging.FileHandler(log_file.split('.txt')[0]+'_loss.txt')
    handler.setFormatter(logging.Formatter('%(levelname)s %(message)s'))
    logger_loss.setLevel(logging.INFO)
    logger_loss.addHandler(handler)
    
