import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from app.ml.configs.visuals import VisualsConfig as Config
from app.ml.constants import Constants as C
from app.ml.loaders import load_config, load_results_metrics
from app.ml.visuals import menu

###################### Functions  ########################


### Select an experiment
def select_experiment(experiments_path):

    # Title of sidebar
    st.sidebar.title("Selection of the experiment")

    # Select the experiment
    experiments_list = [experiment for experiment in os.listdir(experiments_path)]
    experiments_list_sorted = sorted(experiments_list, reverse=True)
    selected_experiment = st.sidebar.selectbox(
        "Choose the experiment you want to see", experiments_list_sorted
    )
    return selected_experiment


### Select metric
def select_metric(metrics_df):

    # List with available metrics for this run
    metric_list = Config.METRIC_LIST

    # Select the metric
    metric_choice = st.sidebar.selectbox(
        "Which metric do you want to plot?", metric_list
    )

    # Keep only results for this metric
    selected_metric_df = metrics_df[metrics_df["metric_name"] == metric_choice].copy()

    return selected_metric_df, metric_choice


### Select model
def select_model(metrics_df):

    # Find unique model
    model_list = metrics_df["model_name"].unique().tolist()

    # Select the model
    model_choice = st.sidebar.selectbox("Which model do you want to plot?", model_list)

    # Keep only results for this model
    selected_model_df = metrics_df[metrics_df["model_name"] == model_choice].copy()

    return selected_model_df


### Compute mean of results on folds
def compute_mean_folds(metrics_df):

    # Group by model
    mean_selected_metrics_df = (
        metrics_df.groupby(["model_name", "run_id"])["metric_value"]
        .mean()
        .reset_index()
    )

    return mean_selected_metrics_df


### Identify run for the feature selection and k values
def k_values(mean_selected_metrics_df, experiments_path, selected_experiment):

    # Write in sidebar feature selection method name
    st.sidebar.write("Feature selection method: ", Config.FEATURE_SELECTION_METHOD_NAME)

    # Dictionnary with results and feature selection parameters
    metrics_feature_selection_k = {}

    for index, run in mean_selected_metrics_df.iterrows():
        run_name = run["run_id"]
        metric_value = run["metric_value"]

        # Load config of the run
        selected_run_path = (
            experiments_path
            / selected_experiment
            / C.EXPERIMENTS_ARTIFACTS_FOLDER_NAME
            / run_name
        )
        config_df = load_config(selected_run_path)

        # Keep name of the feature selection method
        feature_selection_run = eval(config_df["FEATURE_SELECTION_METHOD"])
        feature_selection_run_name = feature_selection_run[0]

        # Keep results for the feature selection method
        if feature_selection_run_name == Config.FEATURE_SELECTION_METHOD_NAME:
            if "k" in feature_selection_run[1]:
                k_value = feature_selection_run[1]["k"]

                # Create dictionnary with metric value and k value for the feature selection method
                metrics_feature_selection_k[run_name] = {
                    "k": k_value,
                    "metric_value": metric_value,
                }

    # Convert dictionnary into dataframe
    metrics_feature_selection_k_df = pd.DataFrame.from_dict(
        metrics_feature_selection_k, orient="index"
    )
    return metrics_feature_selection_k_df


### Create a plot where the x-axis is k values and y-axis is metric values
def plot_metrcis_k(metrics_feature_selection_k_df, metric_choice):

    # Sort by k
    metrics_feature_selection_k_df_sorted = metrics_feature_selection_k_df.sort_values(
        by="k"
    )

    # Create the plot
    fig = px.line(
        metrics_feature_selection_k_df_sorted,
        x="k",
        y="metric_value",
        title=f"Metric value on {metric_choice} corresponding to the number of features selected",
        labels={
            "k": "Number of features selected",
            "metric_value": f"Mean value on {metric_choice}",
        },
    )

    # Plot results
    st.plotly_chart(fig)

    return metrics_feature_selection_k_df_sorted


###################### Preloading  ########################

# Derive paths from configs
ml_run_path = C.ML_PATH / eval(f"C.{Config.RUN_TYPE}")
experiments_path = ml_run_path / C.EXPERIMENTS_FOLDER_NAME

###################### Content page ##########################

# Sidebarmenu
menu()

# Text
st.title("Feature Selection Saturation")

# Experiment selection
selected_experiment = select_experiment(experiments_path)

# Load metrics results
metrics_df = load_results_metrics(
    experiments_path / selected_experiment, C.METRICS_FILENAME
)

# Select metric and keep results on it
selected_metric_df, metric_choice = select_metric(metrics_df)

# Select model and keep results on it
selected_model_df = select_model(selected_metric_df)

# Compute mean on folds
mean_selected_metrics_df = compute_mean_folds(selected_model_df)

# Dataframe with metric value and k value for the feature selection method
metrics_feature_selection_k_df = k_values(
    mean_selected_metrics_df, experiments_path, selected_experiment
)

# Plot results
metrics_feature_selection_k_df_sorted = plot_metrcis_k(
    metrics_feature_selection_k_df, metric_choice
)

# Visualize results on a table
metrics_feature_selection_k_df_sorted
