# best_cad_1

attributes(data_raw$best_cad_1_TEXT)
table(data_raw$best_cad_1_TEXT)
data_clean$best_canadian_book <- NA
data_clean$best_canadian_book <- data_raw$best_cad_1_TEXT
data_clean$best_canadian_book[data_clean$best_canadian_book == ""] <- "dont_know"
table(data_clean$best_canadian_book)

# best_cad_2

attributes(data_raw$best_cad_2_TEXT)
table(data_raw$best_cad_2_TEXT)
data_clean$best_canadian_album <- NA
data_clean$best_canadian_album <- data_raw$best_cad_2_TEXT
data_clean$best_canadian_album[data_clean$best_canadian_album == ""] <- "dont_know"
table(data_clean$best_canadian_album)

# best_cad_3

attributes(data_raw$best_cad_3_TEXT)
table(data_raw$best_cad_3_TEXT)
data_clean$best_canadian_movie <- NA
data_clean$best_canadian_movie <- data_raw$best_cad_3_TEXT
data_clean$best_canadian_movie[data_clean$best_canadian_movie == ""] <- "dont_know"
table(data_clean$best_canadian_movie)

# best_cad_4

attributes(data_raw$best_cad_4_TEXT)
table(data_raw$best_cad_4_TEXT)
data_clean$best_canadian_newspaper <- NA
data_clean$best_canadian_newspaper <- data_raw$best_cad_4_TEXT
data_clean$best_canadian_newspaper[data_clean$best_canadian_newspaper == ""] <- "dont_know"
table(data_clean$best_canadian_newspaper)

# know_fre_1

attributes(data_raw$know_fre_1_TEXT)
table(data_raw$know_fre_1_TEXT)
data_clean$know_french_media_personality <- NA
data_clean$know_french_media_personality <- data_raw$know_fre_1_TEXT
data_clean$know_french_media_personality[data_clean$know_french_media_personality == ""] <- "dont_know"
table(data_clean$know_french_media_personality)

# know_fre_2

attributes(data_raw$know_fre_2_TEXT)
table(data_raw$know_fre_2_TEXT)
data_clean$know_french_singer <- NA
data_clean$know_french_singer <- data_raw$know_fre_2_TEXT
data_clean$know_french_singer[data_clean$know_french_singer == ""] <- "dont_know"
table(data_clean$know_french_singer)

# know_fre_3

attributes(data_raw$know_fre_3_TEXT)
table(data_raw$know_fre_3_TEXT)
data_clean$know_french_actor <- NA
data_clean$know_french_actor <- data_raw$know_fre_3_TEXT
data_clean$know_french_actor[data_clean$know_french_actor == ""] <- "dont_know"
table(data_clean$know_french_actor)

# know_fre_4

attributes(data_raw$know_fre_4_TEXT)
table(data_raw$know_fre_4_TEXT)
data_clean$know_french_writer <- NA
data_clean$know_french_writer <- data_raw$know_fre_4_TEXT
data_clean$know_french_writer[data_clean$know_french_writer == ""] <- "dont_know"
table(data_clean$know_french_writer)

# know_eng_1

attributes(data_raw$know_eng_1_TEXT)
table(data_raw$know_eng_1_TEXT)
data_clean$know_english_media_personality <- NA
data_clean$know_english_media_personality <- data_raw$know_eng_1_TEXT
data_clean$know_english_media_personality[data_clean$know_english_media_personality == ""] <- "dont_know"
table(data_clean$know_english_media_personality)

# know_eng_2

attributes(data_raw$know_eng_2_TEXT)
table(data_raw$know_eng_2_TEXT)
data_clean$know_english_singer <- NA
data_clean$know_english_singer <- data_raw$know_eng_2_TEXT
data_clean$know_english_singer[data_clean$know_english_singer == ""] <- "don't know"
table(data_clean$know_english_singer)

# know_eng_3

attributes(data_raw$know_eng_3_TEXT)
table(data_raw$know_eng_3_TEXT)
data_clean$know_english_actor <- NA
data_clean$know_english_actor <- data_raw$know_eng_3_TEXT
data_clean$know_english_actor[data_clean$know_english_actor == ""] <- "don't know"
table(data_clean$know_english_actor)

# know_eng_4

attributes(data_raw$know_eng_4_TEXT)
table(data_raw$know_eng_4_TEXT)
data_clean$know_english_writer <- NA
data_clean$know_english_writer <- data_raw$know_eng_4_TEXT
data_clean$know_english_writer[data_clean$know_english_writer == ""] <- "don't know"
table(data_clean$know_english_writer)

# friends_1

attributes(data_raw$friends_1_1)

data_clean$french_speaking_friends <- NA
data_clean$french_speaking_friends[data_raw$friends_1_1 == 1] <- "1_2"
data_clean$french_speaking_friends[data_raw$friends_1_2 == 1] <- "3_4"
data_clean$french_speaking_friends[data_raw$friends_1_3 == 1] <- "5_6"
data_clean$french_speaking_friends[data_raw$friends_1_4 == 1] <- "7_plus"
data_clean$french_speaking_friends <- factor(data_clean$french_speaking_friends, levels = c("1_2", "3_4", "5_6", "7_plus"))

table(data_clean$french_speaking_friends)

# friends_2

data_clean$english_speaking_friends <- NA
data_clean$english_speaking_friends[data_raw$friends_2_1 == 1] <- "1_2"
data_clean$english_speaking_friends[data_raw$friends_2_2 == 1] <- "3_4"
data_clean$english_speaking_friends[data_raw$friends_2_3 == 1] <- "5_6"
data_clean$english_speaking_friends[data_raw$friends_2_4 == 1] <- "7+"
data_clean$english_speaking_friends <- factor(data_clean$english_speaking_friends, levels = c("1_2", "3_4", "5_6", "7+"))
table(data_clean$english_speaking_friends)

# friends_3

data_clean$other_language_speaking_friends <- NA
data_clean$other_language_speaking_friends[data_raw$friends_3_1 == 1] <- "1_2"
data_clean$other_language_speaking_friends[data_raw$friends_3_2 == 1] <- "3_4"
data_clean$other_language_speaking_friends[data_raw$friends_3_3 == 1] <- "5_6"
data_clean$other_language_speaking_friends[data_raw$friends_3_4 == 1] <- "7+"
data_clean$other_language_speaking_friends <- factor(data_clean$other_language_speaking_friends, levels = c("1_2", "3_4", "5_6", "7+"))
table(data_clean$other_language_speaking_friends)

# ling_1

attributes(data_raw$ling_1)
table(data_raw$ling_1)
data_clean$french_speaking_skills <- NA
data_clean$french_speaking_skills <- data_raw$ling_1 / 10
table(data_clean$french_speaking_skills)

# ling_2

attributes(data_raw$ling_2)
table(data_raw$ling_2)
data_clean$english_speaking_skills <- NA
data_clean$english_speaking_skills <- data_raw$ling_2 / 10
table(data_clean$english_speaking_skills)




