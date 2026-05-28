"""
MongoDB client initialisation and collection accessors.
"""
import urllib.parse

from pymongo import MongoClient, ASCENDING
from pymongo.errors import ConnectionFailure

from config import MONGO_URI

if not MONGO_URI:
    raise RuntimeError(
        "MONGO_URI is not set. Add it to your .env file or environment variables."
    )

try:
    _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    # Force a connection check at import time
    _client.admin.command("ping")
except ConnectionFailure as exc:
    raise RuntimeError(f"Failed to connect to MongoDB: {exc}") from exc

# Extract database name from the URI path, defaulting to "agri_db"
_parsed = urllib.parse.urlparse(MONGO_URI)
_db_name = _parsed.path.lstrip("/").split("?")[0] or "agri_db"

_db = _client[_db_name]

_history_index_created = False


def get_users():
    """Return the 'users' collection."""
    return _db.users


def get_history():
    """Return the 'history' collection, creating the user_id index on first use."""
    global _history_index_created
    if not _history_index_created:
        _db.history.create_index([("user_id", ASCENDING)])
        _history_index_created = True
    return _db.history
