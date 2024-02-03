import logging, sys

logging.basicConfig(filename='log/example.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))