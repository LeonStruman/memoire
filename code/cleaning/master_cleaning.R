# Packages ---------------------------------------------------------------------
library(dplyr)

# Data -------------------------------------------------------------------------

## load raw data here

data_raw <- haven::read_sav("_SharedFolder_datagotchi-santé/data/raw/data_raw.sav")  %>% 
  filter(code == "complete")

# Clean variables ---------------------------------------------------------

data_clean <- data.frame(id = 1:nrow(data_raw))

## ses -------------------------------------------------------------------------

source("code/cleaning/ses.R")

## lifestyle -------------------------------------------------------------------

source("code/cleaning/lifestyle.R")

## Bien-être -------------------------------------------------------------------

source("code/cleaning/bien_etre.R")

## Comp santé --------------------------------------------------------------

source("code/cleaning/comp_sante.R")

## Enviro travail

source("code/cleaning/enviro_travail.R")

## enviro quartier

source("code/cleaning/enviro_quartier.R")

## Services santé ---------------------------------------------------------------

source("code/cleaning/services_sante.R")

## Valeurs ---------------------------------------------------------------------

source("code/cleaning/valeurs.R")

## Comport socio -----------------------------------------------------------

source("code/cleaning/comport_socio.R")

## Discrimination -----------------------------------------------------------

source("code/cleaning/discrimination.R")

## Political behaviour ---------------------------------------------------------

source("code/cleaning/political_behaviour.R")

## AI --------------------------------------------------------------------------

source("code/cleaning/ai.R")

## inter group ------------------------------------------------------------------

source("code/cleaning/inter_group.R")

## exp --------------------------------------------------------------------------

source("code/cleaning/exp.R")

# Filter for attention check ---------------------------------------------------

## ATTENTION CHECK 1: Please select \"often\" for this answer to confirm that you are paying attention."
table(data_raw$autogestion_7)
attributes(data_raw$autogestion_7)
data_clean$attention_check1_ok <- ifelse(data_raw$autogestion_7 == 3, 1, 0)
table(data_clean$attention_check1_ok)

## ATTENTION CHECK 2: multiple answers possible. Respondents need to have check issue_ai_data_2_5

attributes(data_raw$issue_ai_data_2_1)
table(data_raw$issue_ai_data_2_1)
attributes(data_raw$issue_ai_data_2_2)
attributes(data_raw$issue_ai_data_2_3)
attributes(data_raw$issue_ai_data_2_4)
attributes(data_raw$issue_ai_data_2_5)

data_raw_issue_ai_data_2 <- data_raw %>% 
  select(issue_ai_data_2_1, issue_ai_data_2_2, issue_ai_data_2_3, issue_ai_data_2_4, issue_ai_data_2_5) %>% 
  mutate(## change all columns to numeric
          across(everything(), as.numeric),
          id = 1:nrow(.)) %>% 
  tidyr::pivot_longer(
    cols = starts_with("issue_ai_data_2"),
    names_to = "level",
    names_prefix = "issue_ai_data_2_",
    values_to = "value"
  ) %>% 
  tidyr::replace_na(list(value = 0)) %>% 
  mutate(level = case_when(
    level == 1 ~ "Strongly disagree",
    level == 2 ~ "Somewhat disagree",
    level == 3 ~ "Somewhat agree",
    level == 4 ~ "Strongly agree",
    level == 5 ~ "attention_check"
  ))

data_attention_check2 <- data_raw_issue_ai_data_2 %>% 
  mutate(attention_check2_ok = ifelse(level == "attention_check", value, NA)) %>% 
  filter(level == "attention_check") %>% 
  select(id, attention_check2_ok)

### Also, we need to ensure respondents did not check two likert boxes.
data_attention_check2b <- data_raw_issue_ai_data_2 %>% 
  filter(level != "attention_check") %>% 
  group_by(id) %>% 
  mutate(attention_check2b_ok = ifelse(sum(value) != 1, 0, 1)) %>% 
  filter(value == 1 | attention_check2b_ok == 0) %>% 
  distinct(id, .keep_all = TRUE) %>%
  select(id, attention_check2b_ok)

data_clean <- left_join(data_clean, data_attention_check2, by = "id") %>% 
  left_join(., data_attention_check2b, by = "id")

rm(list = c("data_attention_check2", "data_attention_check2b", "data_raw_issue_ai_data_2"))

## ATTENTION CHECK 3: Please select \"Like me\" for this answer to confirm that you are paying attention."

attributes(data_raw$values_inventory_4)
table(data_raw$values_inventory_4)
data_clean$attention_check3_ok <- NA
data_clean$attention_check3_ok[data_raw$values_inventory_4 == 2] <- 1
data_clean$attention_check3_ok[data_raw$values_inventory_4 != 2] <- 0
table(data_clean$attention_check3_ok)

# Computing scores based on ESSAIM scales --------------------------------------

source("code/cleaning/compute_score.R")

# Save it ----------------------------------------------------------------------

saveRDS(data_clean, "_SharedFolder_datagotchi-santé/data/clean/datagotchi-sante_clean.rds")


