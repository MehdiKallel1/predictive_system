from flask import Flask
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def create_app():
    app = Flask(__name__, 
               template_folder='templates',
               static_folder='static')
    
    # Import and register blueprints/routes
    from app.routes import main_bp
    app.register_blueprint(main_bp)
    
    return app