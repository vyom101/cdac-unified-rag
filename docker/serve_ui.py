"""Container-only UI entry point.

The server listens on the container interface; Compose publishes it exclusively
on the host loopback interface.
"""

from rag.config import load_config
from rag.ui import launch_ui


if __name__ == "__main__":
    launch_ui(load_config(), host="0.0.0.0", port=8765)
