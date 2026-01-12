import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import TargetEncoder


from app.ml.configs.create_feature import CreateFeatureConfig as Config
from app.ml.constants import Constants as C
from app.ml.loaders import load_attributes, load_codebook
from app.ml.tracking import write_feature_library
from app.ml.utils import configure_main_logger

logger = logging.getLogger(__name__)


def features_in_codebook_and_attributes(fields, df_codebook, df_attributes):
    codebook_fields_variables = df_codebook.loc[
        df_codebook[C.CODEBOOK_TYPE_COL].isin(fields), C.CODEBOOK_NAME_COL
    ].values
    codebook_fields_attributes_variables = [
        variable
        for variable in codebook_fields_variables
        if variable in df_attributes.columns
    ]
    return codebook_fields_attributes_variables


def create_nominal_multiple_features(df_codebook, df_attributes):
    # Select nominal_multiple_features
    nominal_multiple_fields = [
        C.CODEBOOK_TYPE_NOMINAL_MULTIPLE_LABEL,
    ]

    nominal_multiple_variables_in_attributes = features_in_codebook_and_attributes(
        nominal_multiple_fields, df_codebook, df_attributes
    )
    logger.info(
        f"{len(nominal_multiple_variables_in_attributes)} variables are nominal multiple."
    )

    # Keep only those columns
    df_nominal_multiple_features = df_attributes[
        nominal_multiple_variables_in_attributes
    ].copy()

    # Approximate to single nominal with categorical encoding
    assert not Config.TARGET_ENCODING_MULTIPLE_NOMINAL  # target encoding obsolete
    if Config.TARGET_ENCODING_MULTIPLE_NOMINAL:
        nominal_multiple_onehot_cols = df_nominal_multiple_features.columns.tolist()
        categories = np.unique(
            [col.split("_")[0] for col in nominal_multiple_onehot_cols]
        )
        for category in categories:
            onehot_cols_in_category = [
                col
                for col in nominal_multiple_onehot_cols
                if col.split("_")[0] == category
            ]
            df_nominal_multiple_features[category] = df_nominal_multiple_features[
                onehot_cols_in_category
            ].idxmax(axis=1)
        df_nominal_multiple_features = df_nominal_multiple_features.drop(
            nominal_multiple_onehot_cols, axis=1
        )

        # remove missing values from target
        target_name = eval(Config.TARGET_NAME)
        df_targets_filled = df_targets[target_name].copy()
        df_targets_filled = df_targets_filled.fillna(df_targets_filled.mean())

        # target encode
        enc_auto = TargetEncoder(smooth="auto")
        column_names, index_names = (
            df_nominal_multiple_features.columns,
            df_nominal_multiple_features.index,
        )
        X_nominal_multiple_features = enc_auto.fit_transform(
            df_nominal_multiple_features, df_targets_filled
        )
        df_nominal_multiple_features = pd.DataFrame(
            data=X_nominal_multiple_features,
            columns=column_names,
            index=index_names,
        )
    else:
        # Convert '1.0' into 1 and Nan into 0
        # TODO: Keep Nan when no answer to the question
        df_nominal_multiple_features = df_nominal_multiple_features.fillna(0).astype(
            int
        )
    return df_nominal_multiple_features


def create_nominal_single_features(df_codebook, df_attributes):
    # Select nominal_single_features
    nominal_single_fields = [
        C.CODEBOOK_TYPE_NOMINAL_SINGLE_LABEL,
    ]
    nominal_single_variables_in_attributes = features_in_codebook_and_attributes(
        nominal_single_fields, df_codebook, df_attributes
    )

    # Keep only those columns
    df_nominal_single_features = df_attributes[
        nominal_single_variables_in_attributes
    ].copy()

    # Encode nominals
    assert not Config.TARGET_ENCODING_SINGLE_NOMINAL  # target encoding obsolete
    if Config.TARGET_ENCODING_SINGLE_NOMINAL:
        # remove missing values from target
        target_name = eval(Config.TARGET_NAME)
        df_targets_filled = df_targets[target_name].copy()
        df_targets_filled = df_targets_filled.fillna(df_targets_filled.mean())

        # target encode
        enc_auto = TargetEncoder(smooth="auto")
        column_names, index_names = (
            df_nominal_single_features.columns,
            df_nominal_single_features.index,
        )
        X_nominal_single_features = enc_auto.fit_transform(
            df_nominal_single_features, df_targets_filled
        )
        df_nominal_single_features = pd.DataFrame(
            data=X_nominal_single_features,
            columns=column_names,
            index=index_names,
        )

        logger.info(
            f"{len(nominal_single_variables_in_attributes)} variables are nominal single and are Target encoded on {target_name}."
        )
    else:
        # Convert it into dummies (one-hot encoding)
        # TODO: Take categories from codebook (num, not text), assert values are in categories, add the categories in getdummies
        df_nominal_single_features = pd.get_dummies(
            df_nominal_single_features,
            columns=nominal_single_variables_in_attributes,
            dtype=int,
            drop_first=True,
            dummy_na=True,
        )

        # Convert into Nan if Nan initially and remove Nan columns added with get_dummies
        for column in nominal_single_variables_in_attributes:
            nan_column = f"{column}_nan"
            if nan_column in df_nominal_single_features.columns:
                # Write Nan in columns concerned
                df_nominal_single_features.loc[
                    df_nominal_single_features[nan_column] == 1,
                    df_nominal_single_features.columns.str.startswith(column),
                ] = np.nan
                # Remove columns _nan
                df_nominal_single_features.drop(columns=[nan_column], inplace=True)

        logger.info(
            f"{len(nominal_single_variables_in_attributes)} variables are nominal single and are converted into {len(df_nominal_single_features.columns)} variables one-hot encoded."
        )
    return df_nominal_single_features


def create_numerical_features(df_codebook, df_attributes):
    numerical_fields = [
        C.CODEBOOK_TYPE_INTEGER_LABEL,
        C.CODEBOOK_TYPE_FLOAT_LABEL,
        C.CODEBOOK_TYPE_ORDINAL_LABEL,
    ]
    numerical_variables_in_attributes = features_in_codebook_and_attributes(
        numerical_fields, df_codebook, df_attributes
    )
    logger.info(f"{len(numerical_variables_in_attributes)} variables are numerical.")
    return df_attributes.loc[:, numerical_variables_in_attributes]


def generate_feature_lookup_table(df_codebook, nominal_single_feature_names):
    nominal_mapping = [
        [feature_name, feature_name.rsplit("_", 1)[0]]
        for feature_name in nominal_single_feature_names
    ]
    df_nominal_mapping = pd.DataFrame(
        nominal_mapping, columns=[C.LOOKUP_FEATURE_NAME_COL, C.CODEBOOK_NAME_COL]
    )
    df_lookup = df_codebook.merge(
        df_nominal_mapping, how="outer", on=C.CODEBOOK_NAME_COL
    )
    df_lookup = df_lookup[df_lookup.id > 0]
    df_lookup[C.LOOKUP_FEATURE_NAME_COL] = df_lookup[C.LOOKUP_FEATURE_NAME_COL].fillna(
        df_lookup[C.CODEBOOK_NAME_COL]
    )
    return df_lookup[
        [C.CODEBOOK_ID_COL, C.CODEBOOK_NAME_COL, C.LOOKUP_FEATURE_NAME_COL]
    ].sort_values(C.CODEBOOK_ID_COL)


def keep_observable(df_codebook, df_attributes):
    df_codebook_observable = df_codebook[
        df_codebook[eval(Config.CODEBOOK_OBSERVABILITY_COL_NAME)].isin(
            C.OBSERVABILITY_LEVEL
        )
    ].copy()
    observables_variables = df_codebook_observable[C.CODEBOOK_NAME_COL].values
    observables_variables_in_attributes = [
        variable
        for variable in observables_variables
        if variable in df_attributes.columns
    ]
    df_attributes_observable = df_attributes[observables_variables_in_attributes]
    logger.info(
        f"Before the observability step, {df_attributes.shape[1]} variables were available. After the observability step, {df_attributes_observable.shape[1]} variables are kept."
    )
    return df_attributes_observable


def create_features(df_candidate_observable, df_codebook):

    # Numerical features
    df_numerical_features = create_numerical_features(
        df_codebook, df_candidate_observable
    )

    # Nominal single features, one-hot encoded
    df_nominal_single_features = create_nominal_single_features(
        df_codebook, df_candidate_observable
    )

    # Nominal multiple features
    df_nominal_multiple_features = create_nominal_multiple_features(
        df_codebook, df_candidate_observable
    )

    # Aggregate here different type of features
    df_features = pd.concat(
        [
            df_numerical_features,
            df_nominal_single_features,
            df_nominal_multiple_features,
        ],
        axis=1,
    )

    return df_features, df_nominal_single_features


# Run feature_engineering
#if __name__ == "__main__":
def feature_engineering():
    logger = configure_main_logger("feature_engineering")
    ml_run_path = C.ML_PATH / eval(f"C.{Config.RUN_TYPE}")

    # Load codebook and attributes
    df_codebook = load_codebook(C.CODEBOOK_PATH, Config.CODEBOOK_VERSION)
    df_attributes = load_attributes(ml_run_path, C.ATTRIBUTES_FILENAME)

    # Create target and features
    df_targets = df_attributes.loc[:, C.TARGET_COLS]

    # Candidates attributes, removing targets
    df_candidate = df_attributes.drop(columns=C.TARGET_COLS)

    # Observable variables
    df_candidate_observable = keep_observable(df_codebook, df_candidate)

    df_features, df_nominal_single_features = create_features(
        df_candidate_observable, df_codebook
    )

    # Generate feature look-up table
    df_feature_lookup = generate_feature_lookup_table(
        df_codebook, df_nominal_single_features.columns
    )

    # Save features and targets to csv
    write_feature_library(
        df_features,
        df_targets,
        df_feature_lookup,
        ml_run_path / C.FEATURE_LIBRARIES_FOLDER_NAME,
        C.FEATURE_LIBRARY_FILENAME,
        C.TARGETS_FILENAME,
        C.FEATURE_LOOKUP_FILENAME,
    )

    logger.info("Features and Targets created with success !! :-)")
