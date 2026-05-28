"""
Authentication utilities: LoginManager, User model, and password helpers.
"""
from bson import ObjectId
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin

from utils.db import get_users

# Exportable instances — initialised via init_app() in app.py
login_manager = LoginManager()
login_manager.login_view = "login"
login_manager.login_message = "Please log in to access this page."

bcrypt = Bcrypt()


class User(UserMixin):
    """Thin wrapper around a MongoDB users document."""

    def __init__(self, doc):
        self._id = doc["_id"]
        self.email = doc["email"]

    @property
    def id(self):
        # Flask-Login requires id to be a string
        return str(self._id)

    @classmethod
    def get(cls, user_id):
        """Fetch a User by _id string; returns None if not found."""
        try:
            doc = get_users().find_one({"_id": ObjectId(user_id)})
        except Exception:
            return None
        return cls(doc) if doc else None


@login_manager.user_loader
def load_user(user_id):
    """Session restoration callback for Flask-Login."""
    return User.get(user_id)


def hash_password(plain: str) -> str:
    """Return a bcrypt hash of *plain*."""
    return bcrypt.generate_password_hash(plain).decode("utf-8")


def check_password(plain: str, hashed: str) -> bool:
    """Return True if *plain* matches *hashed*."""
    return bcrypt.check_password_hash(hashed, plain)
