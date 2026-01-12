library(dplyr)

# Read the SPSS data file and filter for respondents with Finished variable set to TRUE
data <- haven::read_sav("_SharedFolder_datagotchi-santé/data/raw/datagotchi_sante/data_raw.sav")  %>% 
  filter(code == "complete")

data_missing_values <- sondr::qualtrics_na_counter(data)

ggplot2::ggsave("_SharedFolder_datagotchi-santé/data/graphs/na_check.png")
