import logging

FORMAT = '%(asctime)s %(message)s'
formatter = logging.Formatter(FORMAT)

ch = logging.StreamHandler()
ch.setFormatter(formatter)

logger = logging.getLogger('pyBlackScholesAnalytics')
logger.addHandler(ch)
