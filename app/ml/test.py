from app.ml.configs.create_feature import CreateFeatureConfig as Config
from app.ml.constants import Constants as C
from app.ml.loaders import load_attributes
from sklearn.decomposition import FactorAnalysis

path = C.ML_PATH / "real"
file_name = C.ATTRIBUTES_FILENAME
df = load_attributes(path, file_name)
def_reduced = df[
    [
        "CSM_QA1_1",
        "CSM_QA1_2",
        "CSM_QA1_3",
        "CSM_QA2_1",
        "CSM_QA2_2",
        "CSM_QA2_3",
        "CSM_QA2_4",
        "CSM_QA2_5",
        "CSM_QA2_6",
        "CSM_QA2_7",
        "CSM_QA2_8",
        "CSM_QA2_9",
        "CSM_QA2_10",
        "CSM_QA2_11",
    ]
]
def_reduced = def_reduced.fillna(def_reduced.mean())
transformer = FactorAnalysis(n_components=1)
X_transformed = transformer.fit_transform(def_reduced)

df_categories = df[["health_indicator"]]
category_counts = df_categories["health_indicator"].value_counts()
breakpoint()
