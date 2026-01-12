from flask import Flask
from config import Config
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager
from flask_bootstrap import Bootstrap
import pandas as pd
import os
from pathlib import Path
from app.ml.loaders import load_best_model

db = SQLAlchemy()
migrate = Migrate()
login = LoginManager()
login.login_view = 'auth.login' 
bootstrap = Bootstrap()

# model for prediction
MODEL_FILENAME = "model_v2.pkl"
APP_DATA_PATH = Path(os.getcwd()) / "app" / "static" / "data"

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    db.init_app(app)
    migrate.init_app(app, db)
    login.init_app(app)
    bootstrap.init_app(app)

    # load model : tuple with best_model, selected_features
    app.best_model, app.selected_features = load_best_model(APP_DATA_PATH, MODEL_FILENAME)

    from app.auth import bp as auth_bp
    app.register_blueprint(auth_bp)

    from app.main import bp as main_bp
    app.register_blueprint(main_bp)

    return app 

from app import models 