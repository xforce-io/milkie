import logging, sys

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('log/example.log'),
                logging.StreamHandler(sys.stdout)])
