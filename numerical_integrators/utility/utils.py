import os

def ensure_directory_exists(directory="plots"):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
