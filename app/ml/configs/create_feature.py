class CreateFeatureConfig:
    # Versioning
    RUN_TYPE = "REAL_FOLDER_NAME"  # SANDBOX_FOLDER_NAME or REAL_FOLDER_NAME
    CODEBOOK_VERSION = "frozen_codebook_october_15.csv"
    CODEBOOK_OBSERVABILITY_COL_NAME = "C.CODEBOOK_OBSERVABILITY_COL"  # C.CODEBOOK_OBSERVABILITY_COL or C.CODEBOOK_OBSERVABILITY_COL_V2

    # Target encoding
    TARGET_ENCODING_SINGLE_NOMINAL = False  # set fo False, True not supported anymore
    TARGET_ENCODING_MULTIPLE_NOMINAL = False  # set fo False, True not supported anymore
    TARGET_NAME = "C.TARGET_SCORE_TOT"
