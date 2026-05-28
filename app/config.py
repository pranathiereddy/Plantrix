import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

weather_api_key = "171e97242c9b52994b1924b718af4b85"

SECRET_KEY = os.environ.get("SECRET_KEY", "fallback-secret-key")
MONGO_URI = os.environ.get("MONGO_URI")
