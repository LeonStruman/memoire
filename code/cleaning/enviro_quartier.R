# Bloc enviro_quartier ---------------------------------------------------------------

## quartier_domicile_1 ------------------------------------------------------------------------

attributes(data_raw$quartier_domicile_1)
table(data_raw$quartier_domicile_1)
data_clean$enviro_quartier_safe_bike_paths <- NA
data_clean$enviro_quartier_safe_bike_paths[data_raw$quartier_domicile_1 == 1] <- "strongly_disagree"
data_clean$enviro_quartier_safe_bike_paths[data_raw$quartier_domicile_1 == 2] <- "somewhat_disagree"
data_clean$enviro_quartier_safe_bike_paths[data_raw$quartier_domicile_1 == 3] <- "somewhat_agree"
data_clean$enviro_quartier_safe_bike_paths[data_raw$quartier_domicile_1 == 4] <- "strongly_agree"

data_clean$enviro_quartier_safe_bike_paths <- factor(data_clean$enviro_quartier_safe_bike_paths, levels = c("strongly_disagree", "somewhat_disagree", "somewhat_agree", "strongly_agree"))  
table(data_clean$enviro_quartier_safe_bike_paths)


## quartier_domicile_2 ------------------------------------------------------------------------

attributes(data_raw$quartier_domicile_2)
table(data_raw$quartier_domicile_2)
data_clean$enviro_quartier_fresh_fruits_and_vegetables <- NA
data_clean$enviro_quartier_fresh_fruits_and_vegetables[data_raw$quartier_domicile_2 == 1] <- "strongly_disagree"
data_clean$enviro_quartier_fresh_fruits_and_vegetables[data_raw$quartier_domicile_2 == 2] <- "somewhat_disagree"
data_clean$enviro_quartier_fresh_fruits_and_vegetables[data_raw$quartier_domicile_2 == 3] <- "somewhat_agree"
data_clean$enviro_quartier_fresh_fruits_and_vegetables[data_raw$quartier_domicile_2 == 4] <- "strongly_agree"

data_clean$enviro_quartier_fresh_fruits_and_vegetables <- factor(data_clean$enviro_quartier_fresh_fruits_and_vegetables, levels = c("strongly_disagree", "somewhat_disagree", "somewhat_agree", "strongly_agree"))  
table(data_clean$enviro_quartier_fresh_fruits_and_vegetables)


## quartier_domicile_3 ------------------------------------------------------------------------

attributes(data_raw$quartier_domicile_3)
table(data_raw$quartier_domicile_3)
data_clean$enviro_quartier_friendly_neighborhood <- NA
data_clean$enviro_quartier_friendly_neighborhood[data_raw$quartier_domicile_3 == 1] <- "strongly_disagree"
data_clean$enviro_quartier_friendly_neighborhood[data_raw$quartier_domicile_3 == 2] <- "somewhat_disagree"
data_clean$enviro_quartier_friendly_neighborhood[data_raw$quartier_domicile_3 == 3] <- "somewhat_agree"
data_clean$enviro_quartier_friendly_neighborhood[data_raw$quartier_domicile_3 == 4] <- "strongly_agree"

data_clean$enviro_quartier_friendly_neighborhood <- factor(data_clean$enviro_quartier_friendly_neighborhood, levels = c("strongly_disagree", "somewhat_disagree", "somewhat_agree", "strongly_agree"))  
table(data_clean$enviro_quartier_friendly_neighborhood)


## quartier_domicile_4 ------------------------------------------------------------------------

attributes(data_raw$quartier_domicile_4)
table(data_raw$quartier_domicile_4)
data_clean$enviro_quartier_nice_looking_neighborhood <- NA
data_clean$enviro_quartier_nice_looking_neighborhood[data_raw$quartier_domicile_4 == 1] <- "strongly_disagree"
data_clean$enviro_quartier_nice_looking_neighborhood[data_raw$quartier_domicile_4 == 2] <- "somewhat_disagree"
data_clean$enviro_quartier_nice_looking_neighborhood[data_raw$quartier_domicile_4 == 3] <- "somewhat_agree"
data_clean$enviro_quartier_nice_looking_neighborhood[data_raw$quartier_domicile_4 == 4] <- "strongly_agree"

data_clean$enviro_quartier_nice_looking_neighborhood <- factor(data_clean$enviro_quartier_nice_looking_neighborhood, levels = c("strongly_disagree", "somewhat_disagree", "somewhat_agree", "strongly_agree"))  
table(data_clean$enviro_quartier_nice_looking_neighborhood)


## quartier_domicile_5 ------------------------------------------------------------------------

attributes(data_raw$quartier_domicile_5)
table(data_raw$quartier_domicile_5)
data_clean$enviro_quartier_accessible_sports_facilities <- NA
data_clean$enviro_quartier_accessible_sports_facilities[data_raw$quartier_domicile_5 == 1] <- "strongly_disagree"
data_clean$enviro_quartier_accessible_sports_facilities[data_raw$quartier_domicile_5 == 2] <- "somewhat_disagree"
data_clean$enviro_quartier_accessible_sports_facilities[data_raw$quartier_domicile_5 == 3] <- "somewhat_agree"
data_clean$enviro_quartier_accessible_sports_facilities[data_raw$quartier_domicile_5 == 4] <- "strongly_agree"

data_clean$enviro_quartier_accessible_sports_facilities <- factor(data_clean$enviro_quartier_accessible_sports_facilities, levels = c("strongly_disagree", "somewhat_disagree", "somewhat_agree", "strongly_agree"))  
table(data_clean$enviro_quartier_accessible_sports_facilities)

## quartier_domicile_6 ------------------------------------------------------------------------

attributes(data_raw$quartier_domicile_6)
table(data_raw$quartier_domicile_6)
data_clean$enviro_quartier_fast_food <- NA
data_clean$enviro_quartier_fast_food[data_raw$quartier_domicile_6 == 1] <- "strongly_disagree"
data_clean$enviro_quartier_fast_food[data_raw$quartier_domicile_6 == 2] <- "somewhat_disagree"
data_clean$enviro_quartier_fast_food[data_raw$quartier_domicile_6 == 3] <- "somewhat_agree"
data_clean$enviro_quartier_fast_food[data_raw$quartier_domicile_6 == 4] <- "strongly_agree"

data_clean$enviro_quartier_fast_food <- factor(data_clean$enviro_quartier_fast_food, levels = c("strongly_disagree", "somewhat_disagree", "somewhat_agree", "strongly_agree"))  
table(data_clean$enviro_quartier_fast_food)


## quartier_domicile_7 ------------------------------------------------------------------------

attributes(data_raw$quartier_domicile_7)
table(data_raw$quartier_domicile_7)
data_clean$enviro_quartier_green_spaces <- NA
data_clean$enviro_quartier_green_spaces[data_raw$quartier_domicile_7 == 1] <- "strongly_disagree"
data_clean$enviro_quartier_green_spaces[data_raw$quartier_domicile_7 == 2] <- "somewhat_disagree"
data_clean$enviro_quartier_green_spaces[data_raw$quartier_domicile_7 == 3] <- "somewhat_agree"
data_clean$enviro_quartier_green_spaces[data_raw$quartier_domicile_7 == 4] <- "strongly_agree"

data_clean$enviro_quartier_green_spaces <- factor(data_clean$enviro_quartier_green_spaces, levels = c("strongly_disagree", "somewhat_disagree", "somewhat_agree", "strongly_agree"))  
table(data_clean$enviro_quartier_green_spaces)


## quartier_domicile_8 ------------------------------------------------------------------------

attributes(data_raw$quartier_domicile_8)
table(data_raw$quartier_domicile_8)
data_clean$enviro_quartier_safe_neighborhood <- NA
data_clean$enviro_quartier_safe_neighborhood[data_raw$quartier_domicile_8 == 1] <- "strongly_disagree"
data_clean$enviro_quartier_safe_neighborhood[data_raw$quartier_domicile_8 == 2] <- "somewhat_disagree"
data_clean$enviro_quartier_safe_neighborhood[data_raw$quartier_domicile_8 == 3] <- "somewhat_agree"
data_clean$enviro_quartier_safe_neighborhood[data_raw$quartier_domicile_8 == 4] <- "strongly_agree"

data_clean$enviro_quartier_safe_neighborhood <- factor(data_clean$enviro_quartier_safe_neighborhood, levels = c("strongly_disagree", "somewhat_disagree", "somewhat_agree", "strongly_agree"))  
table(data_clean$enviro_quartier_safe_neighborhood)


## quartier_domicile_9 ------------------------------------------------------------------------

attributes(data_raw$quartier_domicile_9)
table(data_raw$quartier_domicile_9)
data_clean$enviro_quartier_places_for_socializing <- NA
data_clean$enviro_quartier_places_for_socializing[data_raw$quartier_domicile_9 == 1] <- "strongly_disagree"
data_clean$enviro_quartier_places_for_socializing[data_raw$quartier_domicile_9 == 2] <- "somewhat_disagree"
data_clean$enviro_quartier_places_for_socializing[data_raw$quartier_domicile_9 == 3] <- "somewhat_agree"
data_clean$enviro_quartier_places_for_socializing[data_raw$quartier_domicile_9 == 4] <- "strongly_agree"

data_clean$enviro_quartier_places_for_socializing <- factor(data_clean$enviro_quartier_places_for_socializing, levels = c("strongly_disagree", "somewhat_disagree", "somewhat_agree", "strongly_agree"))  
table(data_clean$enviro_quartier_places_for_socializing)

## quartier_domicile_10------------------------------------------------------------------------

attributes(data_raw$quartier_domicile_10)
table(data_raw$quartier_domicile_10)
data_clean$enviro_quartier_noisy_polluted_neighborhood <- NA
data_clean$enviro_quartier_noisy_polluted_neighborhood[data_raw$quartier_domicile_10 == 1] <- "strongly_disagree"
data_clean$enviro_quartier_noisy_polluted_neighborhood[data_raw$quartier_domicile_10 == 2] <- "somewhat_disagree"
data_clean$enviro_quartier_noisy_polluted_neighborhood[data_raw$quartier_domicile_10 == 3] <- "somewhat_agree"
data_clean$enviro_quartier_noisy_polluted_neighborhood[data_raw$quartier_domicile_10 == 4] <- "strongly_agree"

data_clean$enviro_quartier_noisy_polluted_neighborhood <- factor(data_clean$enviro_quartier_noisy_polluted_neighborhood, levels = c("strongly_disagree", "somewhat_disagree", "somewhat_agree", "strongly_agree"))  
table(data_clean$enviro_quartier_noisy_polluted_neighborhood)


## quartier_opportunite------------------------------------------------------------------------

attributes(data_raw$quartier_opportunite)
table(data_raw$quartier_opportunite)
data_clean$quartier_opportunities_health <- NA
data_clean$quartier_opportunities_health[data_raw$quartier_opportunite == 1] <- "strongly_disagree"
data_clean$quartier_opportunities_health[data_raw$quartier_opportunite == 2] <- "disagree"
data_clean$quartier_opportunities_health[data_raw$quartier_opportunite == 3] <- "agree"
data_clean$quartier_opportunities_health[data_raw$quartier_opportunite == 4] <- "strongly_agree"

data_clean$quartier_opportunities_health <- factor(data_clean$quartier_opportunities_health, levels = c("strongly_disagree", "disagree", "agree", "strongly_agree"))  
table(data_clean$quartier_opportunities_health)

## quartier satisfaction -------------------------------------------------------

attributes(data_raw$quartier_satisf_1)
table(data_raw$quartier_satisf_1)
data_clean$quartier_satisfaction <- NA
data_clean$quartier_satisfaction <- data_raw$quartier_satisf_1 / 10
table(data_clean$quartier_satisfaction)
