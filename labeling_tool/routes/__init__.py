"""
Routes for the labeling tool
"""

# Import utilities first (no circular dependency)
from labeling_tool.routes.utils import require_user, get_current_user_id, get_current_user_name

# Then import blueprints
from labeling_tool.routes.labeling import labeling_bp
from labeling_tool.routes.api import api_bp
from labeling_tool.routes.dashboard import dashboard_bp

__all__ = ['labeling_bp', 'api_bp', 'dashboard_bp', 'require_user', 'get_current_user_id', 'get_current_user_name']
