import logging, sys

MaxLenLog = 1000

logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('log/example.log'),
                logging.StreamHandler(sys.stdout)])

def INFO(logger :logging.Logger, log :str): logger.info(extractLogExpr(log))
def DEBUG(logger :logging.Logger, log :str): logger.debug(extractLogExpr(log))
def ERROR(logger :logging.Logger, log :str): logger.error(extractLogExpr(log))
def WARNING(logger :logging.Logger, log :str): logger.warning(extractLogExpr(log))
def CRITICAL(logger :logging.Logger, log :str): logger.critical(extractLogExpr(log))

def extractLogExpr(log :str) :
    if len(log) < MaxLenLog:
        return log
    else :
        return log[:int(MaxLenLog*2/3)] + " ...... " + log[-int(MaxLenLog/3):]