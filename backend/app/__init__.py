"""
ResQPet Backend Application
Sistema di identificazione automatizzata dello stato di abbandono nei cani
"""

from flask import Flask, send_from_directory
from flask_socketio import SocketIO
from flask_cors import CORS
import os

socketio = SocketIO()


def create_app(config_name='development'):
    """Application factory pattern"""
    app = Flask(__name__, static_folder='static', static_url_path='/static')

    # Configuration
    app.config['SECRET_KEY'] = 'resqpet-secret-key-change-in-production'
    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max upload

    # Enable CORS for React frontend
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Initialize SocketIO
    socketio.init_app(app, cors_allowed_origins="*", async_mode='eventlet')

    # Register blueprints
    from app.routes.api import api_bp
    from app.routes.websocket import register_socketio_handlers

    app.register_blueprint(api_bp, url_prefix='/api')
    register_socketio_handlers(socketio)

    return app
