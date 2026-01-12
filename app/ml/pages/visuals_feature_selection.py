import os

import plotly.express as px
import streamlit as st
from app.ml.configs.visuals import VisualsConfig as Config
from app.ml.constants import Constants as C
from app.ml.loaders import load_scores_features
from app.ml.visuals import menu

###################### Functions  ########################

### Select a feature library
def select_feature_library(feature_selection_path):

    # Title of sidebar
    st.sidebar.title("Selection of a feature library")

    # Select the feature library
    feature_library_list = [library for library in os.listdir(feature_selection_path)]
    feature_library_list_sorted = sorted(feature_library_list, reverse=True)
    selected_feature_library = st.sidebar.selectbox(
        "Choose the feature library you want to see", feature_library_list_sorted
    )
    return selected_feature_library

### Function to select the feature selection method
def select_feature_selection_method(feature_selection_library_path):

    # Title of sidebar
    st.sidebar.title("Feature selection")

    # Available feature selection methods
    prefix = C.FEATURE_SELECTION_FILENAME.split("{}")[0]
    suffix = C.FEATURE_SELECTION_FILENAME.split("{}")[1]
    available_feature_selection = [
        feature_selection_method.replace(prefix, "").replace(suffix, "")
        for feature_selection_method in os.listdir(feature_selection_library_path)
    ]

    # Put it in a dictionnary
    selection_methods_dict = {}
    for feature_selection_method in available_feature_selection:
        feature_selection_method_parts = feature_selection_method.split("_")
        method_name = feature_selection_method_parts[0]
        param_dict = {}
        if len(feature_selection_method_parts) > 2:
            param_dict = {
                feature_selection_method_parts[i]: feature_selection_method_parts[i + 1]
                for i in range(1, len(feature_selection_method_parts), 2)
            }
        selection_methods_dict[feature_selection_method] = (method_name, param_dict)

    # List with feature selection methods
    unique_methods_list = list(
        {method[0] for method in selection_methods_dict.values()}
    )

    # Selection of the feature selection method
    selected_feature_selection_method = st.sidebar.selectbox(
        "Choose the feature selection method you want to see features' scores",
        unique_methods_list,
    )

    # Available parameters for this feature selection method
    feature_selection_parameters = {}
    for feature_selection_method in selection_methods_dict:
        feature_selection_method_name, feature_selection_method_param = (
            selection_methods_dict[feature_selection_method]
        )
        if (
            feature_selection_method_name == selected_feature_selection_method
            and len(feature_selection_method_param) > 0
        ):
            feature_selection_parameters[feature_selection_method] = (
                feature_selection_method_param
            )

    # Selection the parameters for the feature selection
    selected_feature_selection_method_parameters = {}
    if len(feature_selection_parameters) > 0:
        selected_feature_selection_method_parameters_name = st.sidebar.selectbox(
            "Choose the feature selection method you want to see features' scores",
            feature_selection_parameters,
        )
        selected_feature_selection_method_parameters = feature_selection_parameters[
            selected_feature_selection_method_parameters_name
        ]

    # Loading features scores
    df_features_scores = load_scores_features(
        (
            selected_feature_selection_method,
            selected_feature_selection_method_parameters,
        ),
        feature_selection_library_path,
        C.FEATURE_SELECTION_FILENAME,
    )

    # Sort features by score
    df_features_scores_sorted = df_features_scores.sort_values(
        by="feature_scores", ascending=False
    )

    return selected_feature_selection_method, df_features_scores_sorted


### Function to plot the features and scores for a feature selection method
def plot_feature_selection_scores(
    selected_feature_selection_method, df_features_scores_sorted
):

    # Selection of the number of feature to show
    number_features = st.sidebar.slider(
        "Select the number of features you want to see",
        min_value=1,
        max_value=200,
        value=20,  # Default value
    )

    # Keep only the first lines
    df_features_scores_sorted_top = df_features_scores_sorted.head(number_features)

    # Map feature_selected values to labels for legend
    df_features_scores_sorted_top["selected_label"] = df_features_scores_sorted_top[
        "feature_selected"
    ].apply(lambda x: "selected" if x == 1 else "not selected")

    # Create plot using plotly
    fig = px.bar(
        df_features_scores_sorted_top,
        x="feature_names",
        y="feature_scores",
        color="selected_label",
        color_discrete_map={"selected": "green", "not selected": "red"},
        labels={"feature_names": "Feature name", "feature_scores": "Feature score"},
        title=f"Feature importance for the feature selection {selected_feature_selection_method}",
    )

    # Rotate x-axis labels
    fig.update_layout(xaxis_tickangle=-90)

    # Plot results
    st.plotly_chart(fig)

###################### Preloading  ########################

# Derive paths from configs
ml_run_path = C.ML_PATH / eval(f"C.{Config.RUN_TYPE}")
feature_selection_path = (ml_run_path / C.FEATURE_SELECTION_FOLDER_NAME)


###################### Content page ##########################

# Sidebarmenu
menu()

# Text
st.title("Feature Selection")

# Selection of feature library
selected_feature_library = select_feature_library(feature_selection_path)
feature_selection_library_path = (feature_selection_path / selected_feature_library)

# Selection of feature selection method and loading scores of the features
selected_feature_selection_method, df_features_scores_sorted = (
    select_feature_selection_method(feature_selection_library_path)
)

# Visualize feature selection scores
plot_feature_selection_scores(
    selected_feature_selection_method, df_features_scores_sorted
)
