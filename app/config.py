import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

weather_api_key = ""

SECRET_KEY = os.environ.get("SECRET_KEY", "fallback-secret-key")
MONGO_URI = os.environ.get("MONGO_URI")
