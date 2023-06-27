import logging

def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    # logging.basicConfig(
    #     level=logging.DEBUG,
    #     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    #     datefmt="%Y-%m-%d %H:%M:%S",
    # )

    return logger
