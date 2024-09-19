import logging
from config import *

# Get the current date and time
current_time = datetime.now()
timestamp = current_time.strftime("%Y%m%d_%H%M%S")

# Configure logging level
log_file_base_name = f"{os.path.basename(__file__).replace('.py', '')}_{timestamp}.log"
log_file_path = os.path.join(LOG_DIR, log_file_base_name)
level = getattr(logging, config.LOGGING_LEVEL)
logging.basicConfig(filename=log_file_path,
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=level)

import os
import logging
from datetime import datetime
import config 
def configure_logging(LOG_DIR):
    # Get the current date and time
    current_time = datetime.now()
    timestamp = current_time.strftime("%Y%m%d_%H%M%S")

    # Configure logging level
    log_file_base_name = f"{os.path.basename(__file__).replace('.py', '')}_{timestamp}.log"
    log_file_path = os.path.join(LOG_DIR, log_file_base_name)
    level = getattr(logging, config.LOGGING_LEVEL)
    logging.basicConfig(filename=log_file_path,
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=level)