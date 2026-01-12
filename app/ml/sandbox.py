import logging

import numpy as np
import pandas as pd
from app.ml.configs.create_sandbox import SandboxConfig as Config
from app.ml.constants import Constants as C
from app.ml.utils import configure_main_logger

logger = logging.getLogger(__name__)


def create_attributes_sandbox():
    # Step 0 : Load data
    REAL_PATH = C.DATA_PATH / C.ML_FOLDER_NAME / C.REAL_FOLDER_NAME
    df_attributes_real = pd.read_csv(REAL_PATH / C.ATTRIBUTES_FILENAME, index_col=0)
    logger.info(f"Number of participants in real : {df_attributes_real.shape[0]}")
    logger.info(f"Number of columns in real : {df_attributes_real.shape[1]}")

    # Step 1: Select TARGET_COLS and ATTRIBUTE_ID_COL
    must_have_columns = [C.ATTRIBUTE_ID_COL] + C.TARGET_COLS
    df_selected = df_attributes_real[must_have_columns]

    # Step 2: Select SANDBOX_N_ATTRIBUTES random columns
    # Exclude TARGET_COLS from the list of columns available for random selection
    available_columns = [
        col for col in df_attributes_real.columns if col not in must_have_columns
    ]
    np.random.seed(
        Config.SANDBOX_RANDOM_STATE
    )  # Set the random seed for reproducibility
    random_columns = np.random.choice(
        available_columns, Config.SANDBOX_N_ATTRIBUTES, replace=False
    )

    # Step 3: Combine the TARGET_COLS and the randomly selected columns
    selected_columns = must_have_columns + list(random_columns)
    df_selected = df_attributes_real[selected_columns]

    # Step 4: Sample SANDBOX_N_SAMPLE rows randomly
    df_attributes_sandbox = df_selected.sample(
        n=Config.SANDBOX_N_SAMPLE, random_state=Config.SANDBOX_RANDOM_STATE
    )  # random_state for reproducibility
    logger.info(f"Number of participants in sandbox : {df_attributes_sandbox.shape[0]}")
    logger.info(f"Number of columns in sandbox : {df_attributes_sandbox.shape[1]}")

    # Step 5 : Save sandbox
    SANDBOX_PATH = C.DATA_PATH / C.ML_FOLDER_NAME / C.SANDBOX_FOLDER_NAME
    df_attributes_sandbox.to_csv(SANDBOX_PATH / C.ATTRIBUTES_FILENAME)
    logger.info(f"Sandbox attributes saved with success")


if __name__ == "__main__":
    logger = configure_main_logger("crossval")
    create_attributes_sandbox()
