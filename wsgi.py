import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from webapp import server as app
from loguru import logger

logger.add(os.path.join("home", "prefltlf", "webapp", "assets", "out", "app.log"), rotation="20 MB")
# logger.info(sys.path)
# logger.info(app.get_asset_url("app.log"))

if __name__ == "__main__":
    app.run(port=5080, debug=True)
