from flask import Flask
import os

def create_app():
    app = Flask(__name__, 
               template_folder='templates',
               static_folder='static')
    
    # Import and register blueprints/routes
    from app.routes import main_bp
    app.register_blueprint(main_bp)
    
    return app