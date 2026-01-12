import pandas as pd

from app.ml.constants import Constants as C
from app.ml.configs.eda import EdaConfig as Config
from app.ml.loaders import load_targets

def score_eda():
    ml_run_path = C.ML_PATH / eval(f"C.{Config.RUN_TYPE}")
    frozen_library_folder_name = Config.FEATURE_LIBRARY_VERSION
    df_y = load_targets(
        ml_run_path / C.FEATURE_LIBRARIES_FOLDER_NAME / frozen_library_folder_name, 
        C.TARGETS_FILENAME,
    )
    # Rescale to have values 0-100 instead of 0-70
    df_y['score_tot_scaled'] = df_y['score_tot'] * (100 / 70)

    # Compute quartiles 
    quartiles = df_y['score_tot_scaled'].quantile([0.25, 0.5, 0.75]).round().astype(int)
    X1 = quartiles[0.25]
    X2 = quartiles[0.5]  
    X3 = quartiles[0.75]
    print(f"The quartiles are: X1 = {X1:.2f}, X2 = {X2:.2f}, X3 = {X3:.2f}.")