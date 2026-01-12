library(dplyr)

data <- haven::read_sav("/Users/camillepelletier/Dropbox/git/datagotchi_sante/_SharedFolder_datagotchi-santé/data/raw/Datagotchi-Santé_Pilote_May 24, 2024_10.51.sav") %>% 
  filter(code == "complete")
    


# filter data for Quebec

attributes(data$province_)

data_quebec <- data %>%
  filter(province_ == 9)
table(data$province_, useNA = "ifany")

# filter data for prairie provinces

data_prairie <- data %>%
  filter(province_ == 1 | province_ == 4 | province_ == 10)

# filter data for BC

data_bc <- data %>%
  filter(province_ == 2)

# filter data for ontario     

data_ontario <- data %>%
  filter(province_ == 8)

# filter data for atlantic provinces

data_atlantic <- data %>%
  filter(province_ == 3 | province_ == 5 | province_ == 6 | province_ == 11)

# filter data for territories

data_territories <- data %>%
  filter(province_ == 7 | province_ == 12 | province_ == 13)

# filter data for  heterosexual women

attributes(data$genre)

data_woman <- data %>%
  filter(genre != 1)

data_men <- data %>%
  filter(genre == 1)
# filter for people between 18 and 29 years old

attributes(data$age)

data_18_29 <- data %>%
  filter(age >= 18 & age <= 29)

# filter for people between 30 and 39 years old

data_30_39 <- data %>%
  filter(age >= 30 & age <= 39)

# filter for people between 40 and 49 years old

data_40_49 <- data %>%
  filter(age >= 40 & age <= 49)

# filter for people between 50 and 59 years old

data_50_59 <- data %>%
  filter(age >= 50 & age <= 59)

# filter for people between 60 and 69 years old

data_60_69 <- data %>%
  filter(age >= 60 & age <= 69)

# filter for people over 70 years old inclusively

data_70 <- data %>%
  filter(age >= 70)

# filter for people who are immigrants (pays_origine != 44)

attributes(data$pays_origine)

data_immigrant <- data %>%
  filter(pays_origine != 44)

data_non_immigrant <- data  %>% 
  filter(pays_origine == 44)

# filter for people who are not white 

attributes(data$origines_ethniques)

data_not_white <- data %>%
  filter(origines_ethniques != 1)

data_white <- data %>%
  filter(origines_ethniques == 1)

