import argparse
import logging

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostRegressor 
from sklearn.linear_model import Ridge
from app.ml.configs.deploy import DeployConfig as Config
from app.ml.constants import Constants as C
from app.ml.crossval import preallocate_pipeline, scaling_y
from app.ml.feature_engineering import create_features
from app.ml.loaders import (
    load_attributes,
    load_best_model,
    load_codebook,
    load_df_X_y, 
    load_feature_lookup_table,
    load_selected_features,
)
from app.ml.tracking import save_example_predictions, write_best_model, write_example
from app.ml.utils import configure_main_logger
import shap
from lime.lime_tabular import LimeTabularExplainer

logger = logging.getLogger(__name__)


# Create questionnaire
def create_questionnaire(
    df_feature_lookup,
    selected_features,
    df_codebook,
):
    questions = (
        df_feature_lookup.loc[
            df_feature_lookup[C.LOOKUP_FEATURE_NAME_COL].isin(selected_features),
            C.CODEBOOK_NAME_COL,
        ]
        .unique()
        .tolist()
    )
    df_questionnaire = df_codebook.loc[
        df_codebook[C.CODEBOOK_NAME_COL].isin(questions),
        [
            C.CODEBOOK_ID_COL,
            C.CODEBOOK_NAME_COL,
            C.CODEBOOK_QUESTION_COL,
            C.CODEBOOK_CHOICE_COL,
        ],
    ]
    #df_questionnaire = df_questionnaire.drop_duplicates(C.CODEBOOK_ID_COL)
    return df_questionnaire


def create_example(df_attributes, raw_variable_names):
    df_users = df_attributes.sample(n=Config.N_USERS, random_state=42)[
        raw_variable_names
    ]
    df_users.index = range(Config.N_USERS)
    df_users.index.name = C.ATTRIBUTE_ID_COL
    return df_users


def generate_questionnaire_and_example(ml_run_path, frozen_library_folder_name):

    # Load codebook
    df_codebook = load_codebook(C.CODEBOOK_PATH, Config.CODEBOOK_VERSION)

    # Load feature lookup-table
    df_feature_lookup = load_feature_lookup_table(
        ml_run_path / C.FEATURE_LIBRARIES_FOLDER_NAME / frozen_library_folder_name,
        C.FEATURE_LOOKUP_FILENAME,
    )

    # Load selected feature
    selected_features = load_selected_features(
        Config.FEATURE_SELECTION_METHOD,
        ml_run_path / C.FEATURE_SELECTION_FOLDER_NAME / frozen_library_folder_name,
        C.FEATURE_SELECTION_FILENAME,
    )

    # Create questionnaire
    df_questionnaire = create_questionnaire(
        df_feature_lookup,
        selected_features,
        df_codebook,
    )

    # Create example
    df_attributes = load_attributes(ml_run_path, C.ATTRIBUTES_FILENAME)
    df_example = create_example(
        df_attributes, df_questionnaire[C.CODEBOOK_NAME_COL].tolist()
    )

    write_example(
        df_questionnaire,
        df_example,
        ml_run_path / C.DEPLOY_FOLDER_NAME,
        C.QUESTIONNAIRE_FILENAME,
        C.EXAMPLE_ANSWERS_FILENAME,
        C.DEPLOY_FEATURE_NAMES,
    )


def train_best_model(ml_run_path, frozen_library_folder_name):
    # Load data
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
    df_X_selected = df_X.loc[:, selected_features]

    # Prepare X_train and y_train
    X_train = df_X_selected.values
    y_train = df_y.values
    y_train = scaling_y(y_train, Config.TARGET_BORNE_SUP)

    indices_nan_y_train = np.isnan(y_train)
    assert sum(~indices_nan_y_train) >= Config.MIN_TRAIN_SIZE

    # Remove y_train missing
    y_train = y_train[~indices_nan_y_train]
    X_train = X_train[~indices_nan_y_train]

    # Prepare a pipeline
    model_name = Config.BEST_MODEL["model_name"]
    param_grid = Config.BEST_MODEL["param_grid"]
    logger.info(f"Training for: {model_name}")
    pipeline = preallocate_pipeline(model_name, param_grid)

    # Perform HP optimization
    grid_search = GridSearchCV(pipeline, param_grid, cv=Config.KFOLD_HP)
    logger.debug(
        f"Performing hyperparameter optimization with {Config.KFOLD_HP} splits"
    )
    grid_search.fit(X_train, y_train)

    # Fit the best model
    logger.debug(
        f"- Fitting with: {grid_search.best_params_}",
    )
    best_model = grid_search.best_estimator_

    best_model.fit(X_train, y_train)

    # Additional info
    # - hyperparameters
    best_hyperparameters = grid_search.best_params_
    best_hyperparameters["model_name"] = model_name

    # - SHAP local explanation for the 10th person (index 9)
    # Preprocess training data (everything before the regressor in the pipeline)
    X_train_preprocessed = best_model[:-1].transform(X_train)
    regressor = best_model.named_steps["regressor"]

    # Choose the 10th sample (use last available if fewer than 10 samples)
    sample_idx = 9
    if X_train_preprocessed.shape[0] <= sample_idx:
        sample_idx = X_train_preprocessed.shape[0] - 1
    X_local = X_train_preprocessed[sample_idx : sample_idx + 1]

    try:
        # Try the unified SHAP explainer (auto-selects best explainer)
        explainer = shap.Explainer(regressor, X_train_preprocessed, feature_names=selected_features)
        shap_out = explainer(X_local)
        shap_values_local = shap_out.values
    except Exception:
        # Fallback to KernelExplainer if auto explainer fails (slower)
        background_size = min(50, X_train_preprocessed.shape[0])
        background = shap.sample(X_train_preprocessed, background_size, random_state=42)
        explainer = shap.KernelExplainer(lambda x: regressor.predict(x), background)
        shap_values_local = np.array(explainer.shap_values(X_local))
        if isinstance(shap_values_local, list):
            shap_values_local = np.array(shap_values_local[0])

    # Normalize shape: want 1D array of length n_features
    if shap_values_local.ndim == 3:
        # (1, n_outputs, n_features) -> take first output
        shap_values_local = shap_values_local[0, 0, :]
    elif shap_values_local.ndim == 2:
        # (1, n_features)
        shap_values_local = shap_values_local[0, :]
    # else ndim==1 already

    # Create dict of signed contributions and sort by absolute contribution (descending)
    feature_local_contrib = dict(zip(selected_features, shap_values_local))
    sorted_feature_coeff_dict = dict(
        sorted(feature_local_contrib.items(), key=lambda item: abs(item[1]), reverse=True)
    )

    write_best_model(
        best_model,
        selected_features,
        grid_search.best_params_,
        sorted_feature_coeff_dict,
        ml_run_path / C.DEPLOY_FOLDER_NAME,
        C.BEST_MODEL_FILENAME,
        C.BEST_PARAMS_FILENAME,
        C.BEST_MODEL_COEFFICIENT_FILENAME,
    )



def create_features_for_example(df_attributes_example, ml_run_path):
    df_attributes = load_attributes(ml_run_path, C.ATTRIBUTES_FILENAME)
    df_codebook = load_codebook(C.CODEBOOK_PATH, Config.CODEBOOK_VERSION)
    assert all([col in df_attributes for col in df_attributes_example])
    df_attributes_with_example = pd.concat(
        (df_attributes[df_attributes_example.columns], df_attributes_example),
        axis=0,
    )
    df_features, _ = create_features(df_attributes_with_example, df_codebook)
    return df_features.iloc[-df_attributes_example.shape[0] :, :]


def predict_for_example(
    df_example,
    model_info,
    is_df_features = False
):
    # a path is given to acces the model
    if isinstance(model_info, Path):
        ml_run_path = model_info
        best_model, selected_features = load_best_model(
            ml_run_path / C.DEPLOY_FOLDER_NAME,
            C.BEST_MODEL_FILENAME,
        )
    # the model tuple is given
    elif isinstance(model_info, tuple):
        best_model, selected_features = model_info
    else:
        print("Error with model_info given in predict_for_example function")

    # if df is attributes, convert it into features
    if not is_df_features:
        df_features_example = create_features_for_example(
            df_example, ml_run_path
        )
    else:
        df_features_example = df_example

    assert all(col in df_features_example.columns for col in selected_features)
    X = df_features_example[selected_features].values
    y = best_model.predict(X)
    df_y = pd.DataFrame(
        y,
        index=df_features_example.index,
        columns=[f"{eval(Config.TARGET_NAME)}_prediction"],
    )
    return df_y


# Run deploy
#if __name__ == "__main__":
def deploy(deploy_type):
    #parser = argparse.ArgumentParser(description="Deploy script")
    #parser.add_argument("function", type=str, help="Function to execute")
    #args = parser.parse_args()

    logger = configure_main_logger("deploy")

    ml_run_path = C.ML_PATH / eval(f"C.{Config.RUN_TYPE}")
    frozen_library_folder_name = Config.FEATURE_LIBRARY_VERSION

    if deploy_type == "generate_questionnaire_and_example":
        generate_questionnaire_and_example(
            ml_run_path,
            frozen_library_folder_name,
        )
    elif deploy_type == "train_best_model":
        train_best_model(
            ml_run_path,
            frozen_library_folder_name,
        )
    elif deploy_type == "predict_for_example":
        # Read attributes
        df_attributes_example = pd.read_csv(
        ml_run_path / C.DEPLOY_FOLDER_NAME / C.EXAMPLE_ANSWERS_FILENAME
        ).set_index(C.ATTRIBUTE_ID_COL)
        
        # Predict
        df_y = predict_for_example(
            df_example=df_attributes_example,
            model_info=ml_run_path,
        )
        
        # Save predictions
        save_example_predictions(
            df_y,
            ml_run_path / C.DEPLOY_FOLDER_NAME,
            C.EXAMPLE_PREDICTION_FILENAME,
        )

