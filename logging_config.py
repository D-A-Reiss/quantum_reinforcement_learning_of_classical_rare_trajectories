import logging


# configure logger
logging.basicConfig(
    filename='script.log',
    filemode='a',  # append to existing log
    level=logging.INFO,  # general info; can change to DEBUG for more details
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)