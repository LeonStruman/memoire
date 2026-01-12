import logging
import os

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import StackingRegressor  # Import ajouté

from app.ml.configs.run_crossval import CrossvalConfig as Config
from app.ml.constants import Constants as C
from app.ml.loaders import load_df_X_y, load_selected_features
from app.ml.metrics import available_metrics_dict
from app.ml.models import available_models_dict

from app.ml.tracking import track_results
from app.ml.utils import configure_main_logger

logger = logging.getLogger(__name__)


# Filling missing values in X
def fill_nan_with_value(X, values):
    indices_nan = np.where(np.isnan(X))
    X[indices_nan] = np.take(values, indices_nan[1])
    return X


def preallocate_pipeline(model_name, param_grid):
    pipeline_steps = []
    if "imputer" in param_grid:
        pipeline_steps.append(("imputer", None))
    if "scaler" in param_grid:
        pipeline_steps.append(("scaler", None))
    
    if model_name == "stacking_regressor":
        estimators = param_grid.get("regressor__estimators", [])[0]  # Récupère les estimators
        final_estimator = param_grid.get("regressor__final_estimator", [None])[0]
        cv = param_grid.get("regressor__cv", 5)
        pipeline_steps.append((
            "regressor",
            StackingRegressor(estimators=estimators, final_estimator=final_estimator, cv=cv)
        ))
    else:
        pipeline_steps.append(("regressor", available_models_dict[model_name]()))

    return Pipeline(pipeline_steps)


def scaling_y(y, target_borne_sup):
    y_transformed = y / target_borne_sup * 100
    return y_transformed


def crossval(X, y, index, model_list):
    """Cross-validation (evaluation realized for each fold of each model)

    Attributes
    ----------
    - X : a numpy array containing the features
    - y : a vector containing the target

    Returns
    -------
    """

    # Transform target between 0 and 100
    y_transformed = scaling_y(y, Config.TARGET_BORNE_SUP)

    # Split in folds
    logger.info(f"Crossvalidation : split dataset in {Config.KFOLD} folds")
    kf = KFold(
        n_splits=Config.KFOLD, shuffle=True, random_state=Config.RANDOM_STATE_SPLITTING
    )

    # Lists for predictions results
    fold_id_predict_list = []
    model_name_predict_list = []
    y_test_list = []
    y_predict_list = []
    index_test_list = []

    # Lists for metrics results
    fold_id_metric_list = []
    model_name_metric_list = []
    metric_name_list = []
    metric_value_list = []

    # List of hyperpameter choices
    best_hyperparameters = {}

    # Loop on folds
    for fold_index, (train_index, test_index) in enumerate(kf.split(X)):

        # Prepare dictionary of hyperparameters
        best_hyperparameters[fold_index] = {}

        # Variables X,y train and test
        X_train = X[train_index, :].copy()
        X_test = X[test_index, :].copy()
        y_train = y_transformed[train_index].copy()
        y_test = y_transformed[test_index].copy()
        index_test = index[test_index].copy()

        # Assert minimal number of targets
        indices_nan_y_train = np.isnan(y_train)
        assert sum(~indices_nan_y_train) >= Config.MIN_TRAIN_SIZE
        logger.info(
            f"Fold {fold_index} : keeping {sum(~indices_nan_y_train)}/{len(indices_nan_y_train)} non-missing values"
        )

        # Remove y_train missing
        y_train = y_train[~indices_nan_y_train]
        X_train = X_train[~indices_nan_y_train]

        # Loop on models
        for model in model_list:

            # Prepare a pipeline
            model_name = model["model_name"]
            param_grid = model["param_grid"]
            logger.info(f"- Training for: {model_name}")
            pipeline = preallocate_pipeline(model_name, param_grid)

            # Perform HP optimization
            grid_search = GridSearchCV(pipeline, param_grid, cv=Config.KFOLD_HP)
            logger.debug(
                f"- Performing hyperparameter optimization with {Config.KFOLD_HP} splits"
            )
            grid_search.fit(X_train, y_train)
            best_hyperparameters[fold_index][model_name] = grid_search.best_params_

            # Fit the best model
            logger.debug(
                f"- Fitting with: {grid_search.best_params_}",
            )
            best_model = grid_search.best_estimator_
            best_model.fit(X_train, y_train)

            # Predict
            y_predict = best_model.predict(X_test)

            # For logging : Add prediction to lists
            fold_id_predict_list += [fold_index] * len(y_test)
            model_name_predict_list += [model_name] * len(y_test)
            y_test_list += y_test.tolist()
            y_predict_list += y_predict.tolist()
            index_test_list += index_test.tolist()

            # Parameters y for metrics
            metric_param_y = {
                "y_true": y_test,
                "y_pred": y_predict,
            }

            # Loop on metric
            logger.debug(f"Computing metrics for {model_name}...")
            for metric_name in Config.METRIC_LIST:

                # Metric function
                metric_function, metric_param = available_metrics_dict[metric_name]

                # Parameters for metric
                metric_param_extended = {**metric_param_y, **metric_param}
                metric_value = metric_function(**metric_param_extended)

                # Add metric to lists
                fold_id_metric_list.append(fold_index)
                model_name_metric_list.append(model_name)
                metric_name_list.append(metric_name)
                metric_value_list.append(metric_value)

    # Dict for predictions results
    predictions_dict = {
        C.ATTRIBUTE_ID_COL: index_test_list,
        "fold_id": fold_id_predict_list,
        "model_name": model_name_predict_list,
        "y_test": y_test_list,
        "y_predict": y_predict_list,
    }
    predictions_df = pd.DataFrame.from_dict(predictions_dict)
    predictions_df = predictions_df.set_index(C.ATTRIBUTE_ID_COL)
    # Dict for metrics results
    metrics_dict = {
        "fold_id": fold_id_metric_list,
        "model_name": model_name_metric_list,
        "metric_name": metric_name_list,
        "metric_value": metric_value_list,
    }
    metrics_df = pd.DataFrame.from_dict(metrics_dict)
    logger.info("Cross-validation performed with success !")

    return metrics_df, predictions_df, best_hyperparameters


# Run crossval
#if __name__ == "__main__":
def run_crossval():
    logger = configure_main_logger("crossval")
    ml_run_path = C.ML_PATH / eval(f"C.{Config.RUN_TYPE}")
    frozen_library_folder_name = Config.FEATURE_LIBRARY_VERSION
    df_X, df_y = load_df_X_y(
        ml_run_path / C.FEATURE_LIBRARIES_FOLDER_NAME / frozen_library_folder_name,
        C.FEATURE_LIBRARY_FILENAME,
        C.TARGETS_FILENAME,
        eval(Config.TARGET_NAME),
    )
    selected_features = load_selected_features(
        Config.FEATURE_SELECTION_METHOD,
        ml_run_path / C.FEATURE_SELECTION_FOLDER_NAME / frozen_library_folder_name,
        C.FEATURE_SELECTION_FILENAME,
    )
    model_list = Config.MODEL_LIST
    df_X_selected = df_X.loc[:, selected_features]
    metrics_df, predictions_df, best_hyperparameters = crossval(
        df_X_selected.values,
        df_y.values,
        df_X_selected.index,
        model_list,
    )
    track_results(
        metrics_df,
        predictions_df,
        best_hyperparameters,
        Config,
        ml_run_path / C.EXPERIMENTS_FOLDER_NAME / Config.EXPERIMENT_NAME,
        ml_run_path
        / C.EXPERIMENTS_FOLDER_NAME
        / Config.EXPERIMENT_NAME
        / C.EXPERIMENTS_ARTIFACTS_FOLDER_NAME,
        C.METRICS_FILENAME,
        C.ARTIFACTS_CONFIG_FILENAME,
        C.ARTIFACTS_HP_FILENAME,
        C.ARTIFACTS_PREDICTIONS_FILENAME,
    )
