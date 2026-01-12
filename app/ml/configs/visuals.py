class VisualsConfig:
    # Versioning
    RUN_TYPE = "REAL_FOLDER_NAME"  # SANDBOX_FOLDER_NAME or REAL_FOLDER_NAME
    METRIC_LIST = [
        "mse",
        "mae",
    ]
    FEATURE_SELECTION_METHOD_NAME = "xgboost"
    CODEBOOK_VERSION = "frozen_codebook_july_25.csv"
