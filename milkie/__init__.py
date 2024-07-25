import logging, sys

logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('log/example.log'),
                logging.StreamHandler(sys.stdout)])
