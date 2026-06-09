import sys
from pathlib import Path

# Make `from app.rag.loaders import ...` work when pytest is run from backend/
sys.path.insert(0, str(Path(__file__).parent))
