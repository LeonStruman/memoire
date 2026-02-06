import argparse
import os

from dotenv import load_dotenv

from app import create_app, db
from app.models import Log, User
from config import configs

parser = argparse.ArgumentParser(description="Well-being predictor")
parser.add_argument(
    "--config", type=str, default="debug", help="Configuration name, see config.py"
)
args, _ = parser.parse_known_args()
config_name = args.config

app = create_app(config_class=configs[config_name])


@app.shell_context_processor
def make_shell_context():
    return {"db": db, "User": User, "Log": Log}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
