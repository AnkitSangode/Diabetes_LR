import logging, os
from datetime import datetime

LOG_FILE = datetime.now().strftime("%m%d%Y_%H%M%S") + ".log"
log_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(log_dir, exist_ok=True)

log_file_path = os.path.join(log_dir, LOG_FILE)

logging.basicConfig(
    filename=log_file_path,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
