"""
ResQPet - Main Entry Point
"""

from app import create_app, socketio

app = create_app()

if __name__ == '__main__':
    print("=" * 50)
    print("  ResQPet CCTV Monitoring System")
    print("  Starting server on http://localhost:5000")
    print("=" * 50)

    socketio.run(
        app,
        host='0.0.0.0',
        port=5000,
        debug=True
    )
