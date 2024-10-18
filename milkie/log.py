import logging, sys

MaxLenLog = 4096

logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('log/milkie.log')])

def INFO(logger :logging.Logger, log :str): logger.info(extractLogExpr(log))
def DEBUG(logger :logging.Logger, log :str): logger.debug(extractLogExpr(log))
def ERROR(logger :logging.Logger, log :str): logger.error(extractLogExpr(log))
def WARNING(logger :logging.Logger, log :str): logger.warning(extractLogExpr(log))
def CRITICAL(logger :logging.Logger, log :str): logger.critical(extractLogExpr(log))

def extractLogExpr(log :str) :
    if len(log) > MaxLenLog:
        log = log[:int(MaxLenLog*2/3)] + " ...... " + log[-int(MaxLenLog/3):]
    return log.replace("\n", "")