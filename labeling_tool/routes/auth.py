"""
Authentication routes for user selection
"""

from flask import Blueprint, render_template, redirect, url_for, session, request
from labeling_tool import config

auth_bp = Blueprint('auth', __name__)


@auth_bp.route('/select-user')
def select_user():
    """Display user selection page."""
    return render_template('auth/select_user.html', users=config.USERS)


@auth_bp.route('/set-user', methods=['POST'])
def set_user():
    """Set the current user in session."""
    user_id = request.form.get('user_id', type=int)
    if user_id and user_id in config.USERS:
        session['current_user_id'] = user_id
        return redirect(url_for('labeling.index'))
    return redirect(url_for('auth.select_user'))


@auth_bp.route('/logout')
def logout():
    """Clear user session and redirect to selection page."""
    session.pop('current_user_id', None)
    return redirect(url_for('auth.select_user'))
