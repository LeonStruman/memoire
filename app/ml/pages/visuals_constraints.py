import streamlit as st
from app.ml.configs.visuals import VisualsConfig as Config
from app.ml.constants import Constants as C
from app.ml.loaders import load_codebook
from app.ml.visuals import menu

###################### Functions  ########################


###################### Preloading  ########################

# Derive paths from configs
ml_run_path = C.ML_PATH / eval(f"C.{Config.RUN_TYPE}")
frozen_library_folder_name = Config.FEATURE_LIBRARY_VERSION
feature_library_path = (
    ml_run_path / C.FEATURE_LIBRARIES_FOLDER_NAME / frozen_library_folder_name
)

# Feature lookup table
#df_feature_lookup = 

# Codebook
df_codebook = load_codebook(C.CODEBOOK_PATH, Config.CODEBOOK_VERSION)


###################### Content page ##########################

# Sidebarmenu
menu()

# Text
st.title("Constraints on features")

feature_library_path
df_codebook
