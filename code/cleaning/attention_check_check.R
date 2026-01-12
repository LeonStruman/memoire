data <- readRDS("_SharedFolder_datagotchi-santé/data/clean/datagotchi-sante_clean.rds")

data_full_check <- data %>% 
    filter(attention_check1_ok == 1 & attention_check2_ok == 1 & attention_check2b_ok == 1 & attention_check3_ok == 1)

data_check1 <- data %>% 
    filter(attention_check1_ok == 1)

data_check2 <- data %>%
    filter(attention_check2_ok == 1)

data_check2b <- data %>%
    filter(attention_check2b_ok == 1)

data_check3 <- data %>%
    filter(attention_check3_ok == 1)
