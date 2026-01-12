# Packages ---------------------------------------------------------------------
library(dplyr)
library(readr)

# Experimental data analysis

# Data raw
data_raw <- haven::read_sav("_SharedFolder_datagotchi-santé/data/raw/data_raw.sav")  %>% 
  filter(code == "complete")
names_raw_r <- names(data_raw)

# Check missing values ---------------------------------------------------------
missing_values <- colSums(is.na(data_clean))
missing_values[missing_values > 0]

# Remove observations without score (because of NA in the questionnaire)
data <- data_clean %>%
  filter(!is.na(data_clean$score_norm))

# Check missing values
missing_values <- colSums(is.na(data))
missing_values[missing_values > 0]

# Aperçu des valeurs pour chaque variable
summary(data)

# Comparaisons variables raw and codebook --------------------------------------
raw_codebook <- read_csv("codebook_sante.csv")
names_raw_codebook <- raw_codebook[[3]]

diff_raw_r_codebook <- setdiff(names_raw_r, names_raw_codebook)
print("Variables présentes dans data_raw R mais pas dans le codebook :")
print(diff_raw_r_codebook)

diff_raw_codebook_r <- setdiff(names_raw_codebook, names_raw_r)
print("Variables présentes dans le codebook mais pas dans data_raw R :")
print(diff_raw_codebook_r)
