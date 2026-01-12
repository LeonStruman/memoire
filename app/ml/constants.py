import os
from pathlib import Path

from dotenv import load_dotenv

# Load the .env file
load_dotenv()


class Constants:
    # Folders
    RAW_FOLDER_NAME = "raw"
    ML_FOLDER_NAME = "ml"
    SANDBOX_FOLDER_NAME = "sandbox"
    REAL_FOLDER_NAME = "real"
    CODEBOOK_FOLDER_NAME = "codebooks"
    FEATURE_LIBRARIES_FOLDER_NAME = "feature_libraries"
    FEATURE_SELECTION_FOLDER_NAME = "feature_selection"
    EXPERIMENTS_FOLDER_NAME = "experiments"
    EXPERIMENTS_ARTIFACTS_FOLDER_NAME = "artifacts"
    DEPLOY_FOLDER_NAME = "deploy"

    # Paths
    LOGGING_PATH = Path(os.getcwd()) / "app" / "ml"
    DATA_PATH = Path(os.getenv("DATA_PATH"))
    RAW_PATH = DATA_PATH / RAW_FOLDER_NAME
    ML_PATH = DATA_PATH / ML_FOLDER_NAME
    CODEBOOK_PATH = ML_PATH / CODEBOOK_FOLDER_NAME

    # Filenames
    RAW_FILENAME = "data_raw.sav"
    ATTRIBUTES_FILENAME = "attributes.csv"
    FEATURE_LIBRARY_FILENAME = "feature_library.csv"
    FEATURE_SELECTION_FILENAME = "feature_selection_{}.csv"
    TARGETS_FILENAME = "targets.csv"
    PREDICTIONS_FILENAME = "predictions.csv"
    METRICS_FILENAME = "metrics.csv"
    LOGGING_CONFIG_FILENAME = "logging.conf"
    DASHBOARD_FILENAME = "dashboard.csv"
    ARTIFACTS_CONFIG_FILENAME = "config.json"
    ARTIFACTS_HP_FILENAME = "best_hyperparameters.json"
    ARTIFACTS_PREDICTIONS_FILENAME = "predictions.csv"
    FEATURE_LOOKUP_FILENAME = "feature_lookup.csv"
    QUESTIONNAIRE_FILENAME = "questionnaire.csv"
    EXAMPLE_ANSWERS_FILENAME = "example_answers.csv"
    DEPLOY_FEATURE_NAMES = "deploy_feature_names.csv"
    BEST_MODEL_FILENAME = "model.pkl"
    BEST_MODEL_COEFFICIENT_FILENAME = "model_coefficients.json"
    BEST_PARAMS_FILENAME = "model_details.json"
    EXAMPLE_PREDICTION_FILENAME = "example_predictions.csv"

    # Codebook fields
    CODEBOOK_ID_COL = "id"
    CODEBOOK_NAME_COL = "raw_variable_name"
    CODEBOOK_TYPE_COL = "raw_variable_type"
    CODEBOOK_QUESTION_COL = "Questions"
    CODEBOOK_CHOICE_COL = "Choix de réponse "
    CODEBOOK_OBSERVABILITY_COL = "observability"
    CODEBOOK_OBSERVABILITY_COL_V2 = "observability_v2"
    CODEBOOK_COLS = [
        CODEBOOK_ID_COL,
        CODEBOOK_NAME_COL,
        CODEBOOK_TYPE_COL,
        CODEBOOK_OBSERVABILITY_COL,
        CODEBOOK_OBSERVABILITY_COL_V2,
        CODEBOOK_QUESTION_COL,
        CODEBOOK_CHOICE_COL,
    ]
    CODEBOOK_TYPE_INTEGER_LABEL = "integer"
    CODEBOOK_TYPE_FLOAT_LABEL = "float"
    CODEBOOK_TYPE_ORDINAL_LABEL = "ordinal"
    CODEBOOK_TYPE_NOMINAL_SINGLE_LABEL = "nominal_single"
    CODEBOOK_TYPE_NOMINAL_MULTIPLE_LABEL = "nominal_multiple"

    CODEBOOK_OBSERVABILITY_OBSERVABLE_LABEL = "observable"
    CODEBOOK_OBSERVABILITY_PSEUDO_OBSERVABLE_LABEL = "pseudo-observable"
    CODEBOOK_OBSERVABILITY_PERCEPTIF_LABEL = "perceptif"
    OBSERVABILITY_LEVEL = [
        CODEBOOK_OBSERVABILITY_OBSERVABLE_LABEL,
        CODEBOOK_OBSERVABILITY_PSEUDO_OBSERVABLE_LABEL,
    ]

    # Attribute fields
    ATTRIBUTE_ID_COL = "ResponseId"

    TARGET_EMOTIONAL_HEALTH = "emotional_health"
    TARGET_POSITIVE_FUNCTIONING = "positive_functioning"
    TARGET_SCORE_TOT = "score_tot"
    TARGET_HEALTH_INDICATOR = "health_indicator"
    TARGET_COLS = [
        TARGET_EMOTIONAL_HEALTH,
        TARGET_POSITIVE_FUNCTIONING,
        TARGET_SCORE_TOT,
        TARGET_HEALTH_INDICATOR,
    ]

    # Feature Lookup table col
    LOOKUP_FEATURE_NAME_COL = "feature_names"

    # Metrics fields
    METRICS_RUN_ID_FIELD = "run_id"
    METRICS_TIMESTAMP_FIELD = "timestamp"
