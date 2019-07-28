import logging

logging.basicConfig()
logger = logging.getLogger('gym-duckietown')
logger.setLevel(logging.DEBUG)

logger.info('gym-duckietown %s\n' % __version__)
