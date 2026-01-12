# I trust the people and organizations that develop computational artificial  intelligence to disclose sufficient information to the public.
attributes(data_raw$issue_ai_data_1)
table(data_raw$issue_ai_data_1)
data_clean$ai_trust_enough_disclosure <- sondr::clean_likert_numeric_vector(data_raw$issue_ai_data_1)
table(data_clean$ai_trust_enough_disclosure)


# I take great care to protect my personal digital data.

attributes(data_raw$issue_ai_data_2_1)
table(data_raw$issue_ai_data_2_1)
attributes(data_raw$issue_ai_data_2_2)
attributes(data_raw$issue_ai_data_2_3)
attributes(data_raw$issue_ai_data_2_4)
attributes(data_raw$issue_ai_data_2_5)

data_raw_ai_data_2 <- data_raw %>% 
  select(issue_ai_data_2_1, issue_ai_data_2_2, issue_ai_data_2_3, issue_ai_data_2_4, issue_ai_data_2_5) %>% 
  mutate(## change all columns to numeric
         across(everything(), as.numeric),
         id = 1:nrow(.)) %>% 
  tidyr::pivot_longer(
    cols = starts_with("issue_ai_data_2"),
    names_to = "ai_take_care_protect_personal_data",
    names_prefix = "issue_ai_data_2_",
    values_to = "value"
  ) %>% 
  tidyr::replace_na(list(value = 0)) %>% 
  filter(ai_take_care_protect_personal_data != 5) %>% 
  mutate(ai_take_care_protect_personal_data = case_when(
    ai_take_care_protect_personal_data == 1 ~ 0,
    ai_take_care_protect_personal_data == 2 ~ 0.33,
    ai_take_care_protect_personal_data == 3 ~ 0.67,
    ai_take_care_protect_personal_data == 4 ~ 1
  )) %>%
  group_by(id) %>% 
  mutate(second_attention_check = ifelse(sum(value) != 1, 0, 1)) %>% 
  filter(value == 1 | second_attention_check == 0) %>% 
  distinct(id, .keep_all = TRUE) %>%
  mutate(ai_take_care_protect_personal_data = ifelse(second_attention_check == 1, ai_take_care_protect_personal_data, NA)) %>% 
  select(id, ai_take_care_protect_personal_data) %>% 
  arrange(id)
  
data_clean$ai_take_care_protect_personal_data <- data_raw_ai_data_2$ai_take_care_protect_personal_data

rm(list = c("data_raw_ai_data_2"))

table(data_clean$ai_take_care_protect_personal_data)

# I agree that the government uses my numeric personal data, if it is for the public good
attributes(data_raw$issue_ai_data_3)
table(data_raw$issue_ai_data_3)
data_clean$ai_government_use_numeric_data_public_good <- sondr::clean_likert_numeric_vector(data_raw$issue_ai_data_3)
table(data_clean$ai_government_use_numeric_data_public_good)
