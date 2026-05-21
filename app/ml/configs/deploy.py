from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from lightgbm import LGBMRegressor

class DeployConfig:
    RUN_TYPE = "REAL_FOLDER_NAME"  # SANDBOX_FOLDER_NAME or REAL_FOLDER_NAME
    CODEBOOK_VERSION = "frozen_codebook_october_15.csv"

    # Best feature selection
    FEATURE_LIBRARY_VERSION = "feature_library_v10"
    FEATURE_SELECTION_METHOD = ("xgboost", {"k": 20})

    # Best Model
    BEST_MODEL = {
        "model_name": "lightgbm_regressor",
        "param_grid": {
            "imputer": [SimpleImputer(strategy="mean"), KNNImputer()],
            "regressor__n_estimators": [50, 200],
            "regressor__learning_rate": [0.01, 0.1],
            "regressor__num_leaves": [31, 50],
            "regressor__min_child_samples": [20, 40],
        },
    }
    MIN_TRAIN_SIZE = 30
    TARGET_NAME = "C.TARGET_SCORE_TOT"  # TARGET_SCORE_TOT, TARGET_POSITIVE_FUNCTIONING, TARGET_EMOTIONAL_HEALTH
    TARGET_BORNE_SUP = (
        70  # 70 for score_tot, 55 for positive_functioning, 15 for emotional_health
    )
    KFOLD_HP = 3

    # Example generation
    N_USERS = 3
