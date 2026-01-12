"""
Database initialization and session management
"""

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


def init_db(app):
    """Initialize database with Flask app."""
    from labeling_tool import config

    app.config['SQLALCHEMY_DATABASE_URI'] = config.SQLALCHEMY_DATABASE_URI
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    db.init_app(app)

    # Create tables
    with app.app_context():
        # Import models to register them
        from labeling_tool.database import models
        db.create_all()
        print(f"[Database] Initialized at {config.DATABASE_PATH}")
