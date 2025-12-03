import sys
import os
from pathlib import Path
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

def setup(silent=False):
    """
    Configures the environment to import quartopy using the path defined in .env
    """
    # Find the project root (assuming this file is in utils/)
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    
    # Load .env file
    env_path = project_root / '.env'
    if env_path.exists():
        load_dotenv(env_path)
    else:
        if not silent:
            print(f"Warning: .env file not found at {env_path}")

    # Get QUARTOPY_PATH
    quartopy_path = os.getenv("QUARTOPY_PATH")
    
    if quartopy_path:
        # Normalize path
        quartopy_path = str(Path(quartopy_path).resolve())
        
        if quartopy_path not in sys.path:
            sys.path.insert(0, quartopy_path)
            if not silent:
                print(f"Added quartopy path: {quartopy_path}")
    else:
        if not silent:
            print("Warning: QUARTOPY_PATH not set in .env")

    try:
        import quartopy
        if not silent:
            print("✅ Quartopy imported successfully")
        return quartopy
    except ImportError as e:
        if not silent:
            print(f"❌ Failed to import quartopy: {e}")
        raise
