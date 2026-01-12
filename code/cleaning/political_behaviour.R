# On scale from 0 to 10, how confident are you about voting? (0 being not at all likely, 10 being extremely likely)

attributes(data_raw$turnout_2023_1)
table(data_raw$turnout_2023_1)
data_clean$political_behaviour_turnout_prob <- data_raw$turnout_2023_1 / 10
table(data_clean$political_behaviour_turnout_prob)


# Did you vote in the 2021 Canadian federal election?" -------------------------

attributes(data_raw$turnout_2021)
table(data_raw$turnout_2021)
data_clean$political_behaviour_voted_2021 <- ifelse(data_raw$turnout_2021 == 1, 1, 0)
table(data_clean$political_behaviour_voted_2021)

# Which party did you vote for in the 2021 Canadian federal election? ----------

attributes(data_raw$vote_choice_2021)
table(data_raw$vote_choice_2021)
data_clean$political_behaviour_vote_choice_2021 <- NA
data_clean$political_behaviour_vote_choice_2021[data_raw$vote_choice_2021 == 1] <- "PLC"
data_clean$political_behaviour_vote_choice_2021[data_raw$vote_choice_2021 == 2] <- "PCC"
data_clean$political_behaviour_vote_choice_2021[data_raw$vote_choice_2021 == 3] <- "NPD"
data_clean$political_behaviour_vote_choice_2021[data_raw$vote_choice_2021 == 4] <- "BQ"
data_clean$political_behaviour_vote_choice_2021[data_raw$vote_choice_2021 == 5] <- "PVC"
data_clean$political_behaviour_vote_choice_2021[data_raw$vote_choice_2021 == 6] <- "PPC"
data_clean$political_behaviour_vote_choice_2021[data_raw$vote_choice_2021 == 7] <- "other"
data_clean$political_behaviour_vote_choice_2021[data_raw$vote_choice_2021 %in% c(8, 9)] <- "did_not_vote_spoiled_ballot"
data_clean$political_behaviour_vote_choice_2021 <- factor(data_clean$political_behaviour_vote_choice_2021)
table(data_clean$political_behaviour_vote_choice_2021)

# If a Canadian federal election was held today, which party would you vote for?

attributes(data_raw$vote_choice_today)
table(data_raw$vote_choice_today)
data_clean$political_behaviour_vote_intent <- NA
data_clean$political_behaviour_vote_intent[data_raw$vote_choice_today == 1] <- "PLC"
data_clean$political_behaviour_vote_intent[data_raw$vote_choice_today == 2] <- "PCC"
data_clean$political_behaviour_vote_intent[data_raw$vote_choice_today == 3] <- "NPD"
data_clean$political_behaviour_vote_intent[data_raw$vote_choice_today == 4] <- "BQ"
data_clean$political_behaviour_vote_intent[data_raw$vote_choice_today == 5] <- "PVC"
data_clean$political_behaviour_vote_intent[data_raw$vote_choice_today == 6] <- "PPC"
data_clean$political_behaviour_vote_intent[data_raw$vote_choice_today == 7] <- "other"
data_clean$political_behaviour_vote_intent <- factor(data_clean$political_behaviour_vote_intent)
table(data_clean$political_behaviour_vote_intent)

# To what degree are you certain? ----------------------------------------------

attributes(data_raw$certainty_choice)
table(data_raw$certainty_choice)
data_clean$political_behaviour_vote_intent_certainty <- sondr::clean_likert_numeric_vector(data_raw$certainty_choice)
data_clean$political_behaviour_vote_intent_certainty <- sondr::finverser(data_clean$political_behaviour_vote_intent_certainty)
table(data_clean$political_behaviour_vote_intent_certainty)


# Regardless of the party you intend to vote for in the next Canadian federal election, in general how likely are you to support... 

# The Liberal Party of Canada (LPC)" -------------------------------------------

attributes(data_raw$prob_parties_1)
table(data_raw$prob_parties_1)
data_clean$political_behaviour_prob_support_PLC <- data_raw$prob_parties_1 / 10
table(data_clean$political_behaviour_prob_support_PLC)

# The Conservative Party of Canada (CPC) ---------------------------------------

attributes(data_raw$prob_parties_2)
table(data_raw$prob_parties_2)
data_clean$political_behaviour_prob_support_PCC <- data_raw$prob_parties_2 / 10
table(data_clean$political_behaviour_prob_support_PCC)

# The Bloc Québécois (BQ) ------------------------------------------------------

attributes(data_raw$prob_parties_3)
table(data_raw$prob_parties_3)
data_clean$political_behaviour_prob_support_BQ <- data_raw$prob_parties_3 / 10
table(data_clean$political_behaviour_prob_support_BQ)

# The People's Party of Canada (PPC) -------------------------------------------

attributes(data_raw$prob_parties_4)
table(data_raw$prob_parties_4)
data_clean$political_behaviour_prob_support_PPC <- data_raw$prob_parties_4 / 10
table(data_clean$political_behaviour_prob_support_PPC)

# The New Democratic Party (NDP) -----------------------------------------------

attributes(data_raw$prob_parties_5)
table(data_raw$prob_parties_5)
data_clean$political_behaviour_prob_support_NPD <- data_raw$prob_parties_5 / 10
table(data_clean$political_behaviour_prob_support_NPD)

# The Green Party of Canada (PVC) ----------------------------------------------

attributes(data_raw$prob_parties_6)
table(data_raw$prob_parties_6)
data_clean$political_behaviour_prob_support_PVC <- data_raw$prob_parties_6 / 10
table(data_clean$political_behaviour_prob_support_PVC)

#### Calculate irc -------------------------------------------------------------

long_data <- data_clean %>%
  select(
    id,
    starts_with("political_behaviour_prob_support")
  ) %>% 
  tidyr::pivot_longer(
    cols = -id,
    names_to = "party",
    values_to = "prob_support",
    names_prefix = "political_behaviour_prob_support_"
  )

first_df <- long_data %>% 
  group_by(id) %>% 
  summarise(first = max(prob_support))

second_df <- long_data %>% 
  group_by(id) %>% 
  filter(prob_support != max(prob_support) | sum(prob_support) == 0) %>% 
  summarise(second = max(prob_support))

irc_data <- left_join(long_data, first_df, by = "id") %>% 
  left_join(., second_df, by = "id") %>% 
  group_by(id) %>% 
  mutate(rank = rank(-prob_support, ties.method = "min"),
         irc = ifelse(rank == 1, prob_support - second, prob_support - first)) %>% 
  tidyr::pivot_wider(
    id_cols = id,
    names_from = party,
    values_from = irc,
    names_prefix = "political_behaviour_irc_"
  )

data_clean <- left_join(data_clean, irc_data, by = "id")

rm(list = c("irc_data", "long_data", "first_df", "second_df"))

# In politics, people sometimes talk of left and right. Where would you place yourself on the scale below, where 0 is left and 10 is right?

attributes(data_raw$left_right_1)
table(data_raw$left_right_1)
data_clean$political_behaviour_left_right <- data_raw$left_right_1 / 10
table(data_clean$political_behaviour_left_right)
