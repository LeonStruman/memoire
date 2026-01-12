class ScoreFeatureConfig:
    # Versioning
    RUN_TYPE = "REAL_FOLDER_NAME"  # SANDBOX_FOLDER_NAME or REAL_FOLDER_NAME
    FEATURE_LIBRARY_VERSION = "feature_library_v10"

    # Feature Selection Grid
    FEATURE_SELECTION_METHOD = ("xgboost", {"k": 20})

    # Target choice
    TARGET_NAME = "C.TARGET_SCORE_TOT"  # TARGET_SCORE_TOT, TARGET_POSITIVE_FUNCTIONING, TARGET_EMOTIONAL_HEALTH
