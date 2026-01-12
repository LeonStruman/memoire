ifeq ($(OS),Windows_NT)
    SEP = \\
else
    SEP = /
endif

include .env

clean-python:
	poetry run isort .
	poetry run black .

create-features:
	poetry run python -c "from app.ml.feature_engineering import feature_engineering; feature_engineering()"

score-features:
	poetry run python -c "from app.ml.feature_selection import feature_selection; feature_selection()"

build-sandbox:
	poetry run python app$(SEP)ml$(SEP)sandbox.py

generate-example:
	poetry run python -c "from app.ml.deploy import deploy; deploy('generate_questionnaire_and_example')"

predict-for-example:
	poetry run python -c "from app.ml.deploy import deploy; deploy('predict_for_example')"

run-crossval:
	poetry run python -c "from app.ml.crossval import run_crossval; run_crossval()"

eda:
	poetry run python -c "from app.ml.eda import score_eda; score_eda()"

launch-visuals:
	poetry run python -c "import os, sys; project_root = os.path.abspath(os.getcwd()); os.chdir(project_root); from streamlit.web.cli import main; sys.argv = ['streamlit', 'run', 'app/ml/visuals.py', '--client.showSidebarNavigation=False']; main()"

train-best-model:
	poetry run python -c "from app.ml.deploy import deploy; deploy('train_best_model')"

assign-conditions:
	poetry run python -c "from scripts.assign_conditions import assign_conditions; assign_conditions()"

cloud-connect:
	gcloud compute ssh datagotchi

create-model:
	poetry run python -c "from scripts.train_recommender_model import create_recommender_model; create_recommender_model()"

display-users:
	poetry run python -c "from scripts.view_databases import display_users; display_users()"

display-userpiis:
	poetry run python -c "from scripts.view_databases import display_userpiis; display_userpiis()"

display-logs:
	poetry run python -c "from scripts.view_databases import display_logs; display_logs()"

display-questions:
	poetry run python -c "from scripts.view_databases import display_questions; display_questions()"

display-answers:
	poetry run python -c "from scripts.view_databases import display_answers; display_answers()"

display-activity:
	poetry run python -c "from scripts.view_databases import display_activity; display_activity()"

reload-experiment:
	poetry run python -c "from scripts.reload_experiment import reload_databases; reload_databases()"

update-deploy: 
	git pull
	sudo supervisorctl stop microapp
	flask db upgrade
	sudo supervisorctl start microapp

sup-start:
	sudo supervisorctl start microapp

sup-stop:
	sudo supervisorctl stop microapp

start-website:
	poetry run python microapp.py --config=default

debug:
	poetry run python microapp.py --config=debug

send-files:
	gcloud compute scp --recurse $(DATA_WEBAPP_PATH) $(vm):datagotchi_sante/deploy/data

download-files:
	gcloud compute scp --recurse $(vm):datagotchi_sante/deploy/data/experiment $(DATA_EXPERIMENT_PATH) 

dump-database:
	poetry run python -c "from scripts.export_db import export_database_to_csv; export_database_to_csv()"
