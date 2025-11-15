import os
from dotenv import load_dotenv
from pathlib import Path

# Ruta base del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent

# Cargar variables del .env
load_dotenv(BASE_DIR / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Datos
DATA_RAW_PATH = BASE_DIR / "data" / "raw" / "conversations_sample.csv"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = DATA_PROCESSED_DIR / "results_conversations.csv"
