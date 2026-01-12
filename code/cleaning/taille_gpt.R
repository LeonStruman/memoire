library(dplyr)

## Taille ----------------------------------------------------------------------

data_raw <- haven::read_sav("_SharedFolder_datagotchi-santé/data/raw/data_raw.sav") |> 
  filter(code == "complete")

data_taille <- data.frame(id = 1:nrow(data_raw))

attributes(data_raw$taille)
table(data_raw$taille)
attributes(data_raw$taille_1_TEXT)
table(data_raw$taille_1_TEXT)
attributes(data_raw$taille_2_TEXT)
table(data_raw$taille_2_TEXT)

data_taille$ses_taille <- NA
valid_indices_taille_1 <- which(data_raw$taille_1_TEXT != "")
valid_indices_taille_2 <- which(data_raw$taille_2_TEXT != "")
data_taille$ses_taille[valid_indices_taille_1] <- data_raw$taille_1_TEXT[valid_indices_taille_1]
data_taille$ses_taille[valid_indices_taille_2] <- data_raw$taille_2_TEXT[valid_indices_taille_2]
table(data_taille$ses_taille, useNA = "ifany")

data_taille$ses_taille_gpt <- NA

for (i in 1880:nrow(data_taille)) {
  
    if (is.na(data_taille$ses_taille[i]) || data_taille$ses_taille[i] == "NA") {
      next
    } else {

    system <- "your role is to output the height of respondents in CM based on given height information" # Donne un role au modèle. Doit être clair et concis

    prompt <- paste0("Please read a survey answer representing the height of a survey respondent and convert it to centimeters. Output only a single number representing the height of the respondent in CM. The given value might be numeric or textual, might be in meters, centimeters, or in feet and inches. Your role is to convert the value to centimeters. Use your own judgement to determine if the given value is in centimeters, meters or feet. It is the height of respondents so there shouldn't be extreme values. Here is the respondant's height: ", data_taille$ses_taille[i], ". If there are multiple scenarios, give me the result for the one you judge the most probable. It is VERY important that you output only a single number. NO TEXT.") # Ton prompt
  
    chat_prompt <- openai::create_chat_completion(
        model = "gpt-4-turbo",
        messages = list(
            list("role" = "system",
                 "content" = system
            ),
            list(
                "role" = "user",
                "content" = prompt)
            )
    )
  
  print(paste0("i = ",i ,". Taille: ", data_taille$ses_taille[i]))
  print(paste0("i = ",i ,". Taille convertie: ", chat_prompt$choices$message.content))

  data_taille$ses_taille_gpt[i] <- chat_prompt$choices$message.content

  }
}

for (i in 1:nrow(data_taille)) {
  
    if (is.na(data_taille$ses_taille[i]) || data_taille$ses_taille[i] == "NA") {
      print(paste0("i = ",i ,". Skip NA"))
      next
    }

    if (as.numeric(data_taille$ses_taille_gpt[i]) < 250 & as.numeric(data_taille$ses_taille_gpt[i]) > 50) {
      print(paste0("i = ",i ,". Skip cause OK"))
      next
    }

    system <- "your role is to output the height of respondents in CM based on given height information" # Donne un role au modèle. Doit être clair et concis

    prompt <- paste0("Please read a survey answer representing the height of a survey respondent and convert it to centimeters. Output only a single number representing the height of the respondent in CM. The given value might be numeric or textual, might be in meters, centimeters, or in feet and inches. Your role is to convert the value to centimeters. Use your own judgement to determine if the given value is in centimeters, meters or feet. It is the height of respondents so there shouldn't be extreme values. Here is the respondant's height: ", data_taille$ses_taille[i], ". If there are multiple scenarios, give me the result for the one you judge the most probable. It is VERY important that you output only a single number. NO TEXT.") # Ton prompt
  
    chat_prompt <- openai::create_chat_completion(
        model = "gpt-4-turbo",
        messages = list(
            list("role" = "system",
                 "content" = system
            ),
            list(
                "role" = "user",
                "content" = prompt)
            )
    )
  
  print(paste0("i = ",i ,". Taille: ", data_taille$ses_taille[i]))
  print(paste0("i = ",i ,". Taille convertie: ", chat_prompt$choices$message.content))

  data_taille$ses_taille_gpt[i] <- chat_prompt$choices$message.content

}


data_taille <- data_taille |> 
  select(id, ses_taille_gpt) |> 
  mutate(ses_taille = as.numeric(ses_taille_gpt)) |> 
  select(-ses_taille_gpt)

saveRDS(data_taille, "_SharedFolder_datagotchi-santé/data/clean/data_taille.rds")

data_clean <- readRDS("_SharedFolder_datagotchi-santé/data/clean/datagotchi-sante_clean.rds")

data_clean <- merge(data_clean, data_taille, by = "id")

saveRDS(data_clean, "_SharedFolder_datagotchi-santé/data/clean/datagotchi-sante_clean.rds")
