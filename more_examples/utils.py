import os
from dotenv import load_dotenv

def load_environment_variables() -> str:
    """Load environment variables from a .env file."""
    load_dotenv()
    return os.getenv("APIKEY_NIXTLA")