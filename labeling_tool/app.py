"""
Flask Application Factory for ResQPet Labeling Tool
"""

from flask import Flask, session
from labeling_tool import config
from labeling_tool.database import init_db


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)

    # Configuration
    app.config['SECRET_KEY'] = config.SECRET_KEY
    app.config['DEBUG'] = config.DEBUG

    # Store config paths for access in routes
    app.config['DATASETS'] = config.DATASETS
    app.config['BACKBONE_MODEL'] = str(config.BACKBONE_MODEL)
    app.config['COLLAR_MODEL'] = str(config.COLLAR_MODEL)
    app.config['EXPORTS_DIR'] = config.EXPORTS_DIR
    app.config['PROJECT_ROOT'] = config.PROJECT_ROOT

    # Ensure labeling_data directory exists
    config.LABELING_DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize database
    init_db(app)

    # Register blueprints
    from labeling_tool.routes.labeling import labeling_bp
    from labeling_tool.routes.api import api_bp
    from labeling_tool.routes.dashboard import dashboard_bp
    from labeling_tool.routes.auth import auth_bp

    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(labeling_bp)
    app.register_blueprint(api_bp, url_prefix='/api')
    app.register_blueprint(dashboard_bp, url_prefix='/dashboard')

    # Context processor to make current user available in all templates
    @app.context_processor
    def inject_user():
        user_id = session.get('current_user_id')
        if user_id and user_id in config.USERS:
            return {
                'current_user_id': user_id,
                'current_user_name': config.USERS[user_id]
            }
        return {'current_user_id': None, 'current_user_name': None}

    print(f"[LabelingTool] App created")
    print(f"[LabelingTool] Project root: {config.PROJECT_ROOT}")
    print(f"[LabelingTool] Database: {config.DATABASE_PATH}")

    return app


def main():
    """Run the labeling tool server."""
    app = create_app()
    print("\n" + "=" * 50)
    print("ResQPet Labeling Tool")
    print("=" * 50)
    print("Open http://localhost:5001 in your browser")
    print("=" * 50 + "\n")
    app.run(host='0.0.0.0', port=5001, debug=True)


if __name__ == '__main__':
    main()
