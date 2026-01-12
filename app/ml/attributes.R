# Packages ---------------------------------------------------------------------
library(dplyr)

# Main -------------------------------------------------------------------------

## Load raw data here
data_raw <- haven::read_sav("_SharedFolder_datagotchi-santé/data/raw/data_raw.sav")  %>% 
  filter(code == "complete")
data <- data_raw

## Convert to ESSAIM scales
data$qol_heureux <- convert_16_to_05(data_raw$CSM_QA1_1)
data$qol_interet <- convert_16_to_05(data_raw$CSM_QA1_2)
data$qol_satisfa <- convert_16_to_05(data_raw$CSM_QA1_3)
data$qol_soc_apport <- convert_61_to_05(data_raw$CSM_QA2_1)
data$qol_apparten <- convert_61_to_05(data_raw$CSM_QA2_2)
data$qol_soc_meilleure <- convert_61_to_05(data_raw$CSM_QA2_3)
data$qol_gens_bons <- convert_61_to_05(data_raw$CSM_QA2_4)
data$qol_soc_sens <- convert_61_to_05(data_raw$CSM_QA2_5)
data$qol_aime_perso <- convert_61_to_05(data_raw$CSM_QA2_6)
data$qol_responsab <- convert_61_to_05(data_raw$CSM_QA2_7)
data$qol_relations <- convert_61_to_05(data_raw$CSM_QA2_8)
data$ol_exp_grandirq <- convert_61_to_05(data_raw$CSM_QA2_9)
data$qol_exprimer <- convert_61_to_05(data_raw$CSM_QA2_10)
data$qol_vie_but <- convert_61_to_05(data_raw$CSM_QA2_11)

## Compute target
### Compute scores
data <- compute_scores(data)

### Compute binary indicators
data <- compute_indicators(data)

## Attention checks
data <- filter_attention_checks(data, 2)

## Save data
write.csv(data, file = "_SharedFolder_datagotchi-santé/data/ml/attributes.csv")



# Convert to ESSAIM scales -----------------------------------------------------
## Scales where values are between 1 (= Never) to 6 (=Always) convert to values between 0 (=Never) to 5 (=Always)
convert_16_to_05 <- function(variable_16){
  variable_05 <- variable_16 - 1 
  return (variable_05)
}

## Scales where values are between 6 (= Never) to 1 (=Always) convert to values between 0 (=Never) to 5 (=Always)
convert_61_to_05 <- function(variable_61){
  variable_05 <- 6 - variable_61 
  return (variable_05)
}


# Compute score ----------------------------------------------------------------
compute_scores <- function(data) {
  
  ## Emotional mental health (hedonic), varies between 0 and 15 (based on 3 variables)
  data$emotional_health <- data$qol_heureux + data$qol_interet + data$qol_satisfa
  
  ## Positive functioning, varies between 0 and 55 (based on 11 variables)
  data$positive_functioning <- data$qol_soc_apport + data$qol_apparten + data$qol_soc_meilleure + data$qol_gens_bons + data$qol_soc_sens + data$qol_aime_perso + data$qol_responsab + data$qol_relations + data$ol_exp_grandirq + data$qol_exprimer + data$qol_vie_but
  
  ## Total score, varies between 0 and 70
  data$score_tot <- data$emotional_health + data$positive_functioning
  
  return(data)
}

# Compute binary indicators ----------------------------------------------------
compute_indicators <- function(data){
  
  ## Santé mentale florissante, binaire. Une santé mentale florissante exige la réponse « presque tous les jours » ou « tous les jours » à au moins l'une des 3 questions sur le bien-être émotionnel/santé mentale émotionnelle, et à au moins 6 des 11 questions sur le fonctionnement positif.
  data$flourishing <- if_else(
    (data$qol_heureux>=4 | data$qol_interet>=4 |data$qol_satisfa>=4)
    & (rowSums(data[, c('qol_soc_apport', 'qol_apparten', 'qol_soc_meilleure', 'qol_gens_bons', 'qol_soc_sens', 'qol_aime_perso', 'qol_responsab', 'qol_relations', 'ol_exp_grandirq', 'qol_exprimer', 'qol_vie_but')] >= 4) >= 6),
    1,0)
  
  ## Santé mentale languissante, binaire. Une santé mentale languissante exige la réponse « une fois ou deux » ou « jamais » à au moins l'une des 3 questions sur le bien-être émotionnel/santé mentale émotionnelle, et à au moins 6 des 11 questions sur le fonctionnement positif. 
  data$languishing  <- if_else(
    (data$qol_heureux<=1 | data$qol_interet<=1 |data$qol_satisfa<=1)
    & (rowSums(data[, c('qol_soc_apport', 'qol_apparten', 'qol_soc_meilleure', 'qol_gens_bons', 'qol_soc_sens', 'qol_aime_perso', 'qol_responsab', 'qol_relations', 'ol_exp_grandirq', 'qol_exprimer', 'qol_vie_but')] <= 1) >= 6),
    1,0)
  
  ## Santé mentale modérément bonne, binaire. Ce sont ceux qui ne sont ni dans florissante ni dans languissante.
  data$moderate <- if_else(
    (data$flourishing==0)
    & (data$languishing==0),
    1,0)
  
  ## One variable categorizing the 3 categories
  data <- data %>%
    mutate(health_indicator = case_when(
      data$flourishing == 1 ~ "flourishing",
      data$languishing == 1 ~ "languishing",
      data$moderate == 1 ~ "moderate",
      TRUE ~ NA_character_ 
    )) %>%
    select(-flourishing, -languishing, -moderate)
  
  return(data)
}


# Attention checks -------------------------------------------------------------
filter_attention_checks <- function(data, attention_check_nb){
  
  ## ATTENTION CHECK 1: Please select \"often\" for this answer to confirm that you are paying attention."
  data$attention_check1_ok <- ifelse(data$autogestion_7 == 3, 1, 0)
  
  ## ATTENTION CHECK 2: multiple answers possible. Respondents need to have check issue_ai_data_2_5
  data$attention_check2_ok <- ifelse(is.na(data$issue_ai_data_2_5), 0, ifelse(data$issue_ai_data_2_5 == 1, 1, 0))
  
  ## ATTENTION CHECK 3: Please select \"Like me\" for this answer to confirm that you are paying attention."
  data$attention_check3_ok <- ifelse(data$values_inventory_4 == 2, 1, 0)
  
  ## At least attention_check_nb of attention checks should be validated
  data$attention_filter <- ifelse(rowSums(data[, c("attention_check1_ok", "attention_check2_ok", "attention_check3_ok")]) >= attention_check_nb, 1, 0)
  
  ## Remove participants not validating attention checks
  data <- data %>% 
    filter(attention_filter == 1)
  
  ## Remove attention check columns
  data <- data %>% select(-attention_check1_ok, -attention_check2_ok, -attention_check3_ok, -attention_filter)
  
  return(data)
}
