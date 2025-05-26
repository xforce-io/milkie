import logging, sys

MaxLenLog = 4096

def setup_logging():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    root_logger.handlers = []
    
    file_handler = logging.FileHandler('log/milkie.log')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(file_handler)

setup_logging()

def INFO(logger :logging.Logger, log :str): logger.info(extractLogExpr(log))
def DEBUG(logger :logging.Logger, log :str): logger.debug(extractLogExpr(log))
def ERROR(logger :logging.Logger, log :str): logger.error(extractLogExpr(log))
def WARNING(logger :logging.Logger, log :str): logger.warning(extractLogExpr(log))
def CRITICAL(logger :logging.Logger, log :str): logger.critical(extractLogExpr(log))

def extractLogExpr(log :str) :
    if len(log) > MaxLenLog:
        log = log[:int(MaxLenLog*2/3)] + " ...... " + log[-int(MaxLenLog/3):]
    return log.replace("\n", "")