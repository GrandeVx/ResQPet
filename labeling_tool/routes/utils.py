"""
Route utilities and decorators
"""

from functools import wraps
from flask import session, redirect, url_for, g
from labeling_tool import config


def require_user(f):
    """Decorator to require user selection before accessing a route."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user_id = session.get('current_user_id')
        if not user_id or user_id not in config.USERS:
            return redirect(url_for('auth.select_user'))
        g.current_user_id = user_id
        g.current_user_name = config.USERS[user_id]
        return f(*args, **kwargs)
    return decorated_function


def get_current_user_id():
    """Get the current user ID from session or g object."""
    return getattr(g, 'current_user_id', None) or session.get('current_user_id')


def get_current_user_name():
    """Get the current user name from session."""
    user_id = get_current_user_id()
    return config.USERS.get(user_id, None) if user_id else None
