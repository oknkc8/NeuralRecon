from tqdm import tqdm
from loguru import logger
import time

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""))

logger.info("Initializing")

for x in tqdm(range(100)):
    logger.info("Iterating #{}", x)
    time.sleep(0.1)
