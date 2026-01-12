# values_inventory--------------------------------------------------------------

attributes(data_raw$values_inventory_1)
table(data_raw$values_inventory_1)
data_clean$valeurs_respect_elders <- sondr::finverser(data_raw$values_inventory_1 -1) / 5
table(data_clean$valeurs_respect_elders)

attributes(data_raw$values_inventory_2)
table(data_raw$values_inventory_2)
data_clean$valeurs_religiosity <- sondr::finverser(data_raw$values_inventory_2 -1) / 5
table(data_clean$valeurs_religiosity)

attributes(data_raw$values_inventory_3)
table(data_raw$values_inventory_3)
data_clean$valeurs_care_for_others <- sondr::finverser(data_raw$values_inventory_3 -1) / 5
table(data_clean$valeurs_care_for_others)

attributes(data_raw$values_inventory_5)
table(data_raw$values_inventory_5)
data_clean$valeurs_equal_opportunity_for_everyone <- sondr::finverser(data_raw$values_inventory_5 -1) / 5
table(data_clean$valeurs_equal_opportunity_for_everyone)

attributes(data_raw$values_inventory_6)
table(data_raw$values_inventory_6)
data_clean$valeurs_curious <- sondr::finverser(data_raw$values_inventory_6 -1) / 5
table(data_clean$valeurs_curious)

attributes(data_raw$values_inventory_7)
table(data_raw$values_inventory_7)
data_clean$valeurs_takes_risks <- sondr::finverser(data_raw$values_inventory_7 -1) / 5
table(data_clean$valeurs_takes_risks)

attributes(data_raw$values_inventory_8)
table(data_raw$values_inventory_8)
data_clean$valeurs_having_fun <- sondr::finverser(data_raw$values_inventory_8 -1) / 5
table(data_clean$valeurs_having_fun)

attributes(data_raw$values_inventory_9)
table(data_raw$values_inventory_9)
data_clean$valeurs_important_being_successful <- sondr::finverser(data_raw$values_inventory_9 -1) / 5
table(data_clean$valeurs_important_being_successful)

attributes(data_raw$values_inventory_10)
table(data_raw$values_inventory_10)
data_clean$valeurs_important_being_in_charge <- sondr::finverser(data_raw$values_inventory_10 -1) / 5
table(data_clean$valeurs_important_being_in_charge)

attributes(data_raw$values_inventory_11)
table(data_raw$values_inventory_11)
data_clean$valeurs_organized <- sondr::finverser(data_raw$values_inventory_11 -1) / 5
table(data_clean$valeurs_organized)

## perso_1 --------------------------------------------------------
attributes(data_raw$perso_1)
table(data_raw$perso_1)
data_clean$valeurs_extraverted_enthusiastic <- NA
data_clean$valeurs_extraverted_enthusiastic[data_raw$perso_1 == 1] <- 1
data_clean$valeurs_extraverted_enthusiastic[data_raw$perso_1 == 2] <- 0.83
data_clean$valeurs_extraverted_enthusiastic[data_raw$perso_1 == 3] <- 0.67
data_clean$valeurs_extraverted_enthusiastic[data_raw$perso_1 == 4] <- 0.5
data_clean$valeurs_extraverted_enthusiastic[data_raw$perso_1 == 5] <- 0.33
data_clean$valeurs_extraverted_enthusiastic[data_raw$perso_1 == 6] <- 0.17
data_clean$valeurs_extraverted_enthusiastic[data_raw$perso_1 == 7] <- 0
table(data_clean$valeurs_extraverted_enthusiastic)

## perso_2 --------------------------------------------------------
attributes(data_raw$perso_2)
table(data_raw$perso_2)

data_clean$valeurs_critical_quarrelsome <- NA
data_clean$valeurs_critical_quarrelsome <- sondr::clean_likert_numeric_vector(data_raw$perso_2)
table(data_clean$valeurs_critical_quarrelsome)

## perso_3 --------------------------------------------------------

attributes(data_raw$perso_3)
table(data_raw$perso_3)
data_clean$valeurs_dependable_self_disciplined <- NA
data_clean$valeurs_dependable_self_disciplined[data_raw$perso_3 == 1] <- 1
data_clean$valeurs_dependable_self_disciplined[data_raw$perso_3 == 2] <- 0.83
data_clean$valeurs_dependable_self_disciplined[data_raw$perso_3 == 3] <- 0.67
data_clean$valeurs_dependable_self_disciplined[data_raw$perso_3 == 4] <- 0.5
data_clean$valeurs_dependable_self_disciplined[data_raw$perso_3 == 5] <- 0.33
data_clean$valeurs_dependable_self_disciplined[data_raw$perso_3 == 6] <- 0.17
data_clean$valeurs_dependable_self_disciplined[data_raw$perso_3 == 7] <- 0
table(data_clean$valeurs_dependable_self_disciplined)

## perso_4 --------------------------------------------------------
attributes(data_raw$perso_4)
table(data_raw$perso_4)

data_clean$valeurs_anxious_easily_upset <- NA
data_clean$valeurs_anxious_easily_upset[data_raw$perso_4 == 1] <- 1
data_clean$valeurs_anxious_easily_upset[data_raw$perso_4 == 2] <- 0.83
data_clean$valeurs_anxious_easily_upset[data_raw$perso_4 == 3] <- 0.67
data_clean$valeurs_anxious_easily_upset[data_raw$perso_4 == 4] <- 0.5
data_clean$valeurs_anxious_easily_upset[data_raw$perso_4 == 5] <- 0.33
data_clean$valeurs_anxious_easily_upset[data_raw$perso_4 == 6] <- 0.17
data_clean$valeurs_anxious_easily_upset[data_raw$perso_4 == 7] <- 0

table(data_clean$valeurs_anxious_easily_upset)

## perso_5 --------------------------------------------------------
attributes(data_raw$perso_5)
table(data_raw$perso_5)

data_clean$valeurs_open_experiences_complex <- NA
data_clean$valeurs_open_experiences_complex[data_raw$perso_5 == 1] <- 1
data_clean$valeurs_open_experiences_complex[data_raw$perso_5 == 2] <- 0.83
data_clean$valeurs_open_experiences_complex[data_raw$perso_5 == 3] <- 0.67
data_clean$valeurs_open_experiences_complex[data_raw$perso_5 == 4] <- 0.5
data_clean$valeurs_open_experiences_complex[data_raw$perso_5 == 5] <- 0.33
data_clean$valeurs_open_experiences_complex[data_raw$perso_5 == 6] <- 0.17
data_clean$valeurs_open_experiences_complex[data_raw$perso_5 == 7] <- 0

table(data_clean$valeurs_open_experiences_complex)

## perso_6 --------------------------------------------------------
attributes(data_raw$perso_6)
table(data_raw$perso_6)

data_clean$valeurs_reserved_quiet <- NA

data_clean$valeurs_reserved_quiet[data_raw$perso_6 == 1] <- 1
data_clean$valeurs_reserved_quiet[data_raw$perso_6 == 2] <- 0.83
data_clean$valeurs_reserved_quiet[data_raw$perso_6 == 3] <- 0.67
data_clean$valeurs_reserved_quiet[data_raw$perso_6 == 4] <- 0.5
data_clean$valeurs_reserved_quiet[data_raw$perso_6 == 5] <- 0.33
data_clean$valeurs_reserved_quiet[data_raw$perso_6 == 6] <- 0.17
data_clean$valeurs_reserved_quiet[data_raw$perso_6 == 7] <- 0

table(data_clean$valeurs_reserved_quiet)

## perso_7 --------------------------------------------------------
attributes(data_raw$perso_7)
table(data_raw$perso_7)

data_clean$valeurs_sympathetic_warm <- NA
data_clean$valeurs_sympathetic_warm[data_raw$perso_7 == 1] <- 1
data_clean$valeurs_sympathetic_warm[data_raw$perso_7 == 2] <- 0.83
data_clean$valeurs_sympathetic_warm[data_raw$perso_7 == 3] <- 0.67
data_clean$valeurs_sympathetic_warm[data_raw$perso_7 == 4] <- 0.5
data_clean$valeurs_sympathetic_warm[data_raw$perso_7 == 5] <- 0.33
data_clean$valeurs_sympathetic_warm[data_raw$perso_7 == 6] <- 0.17
data_clean$valeurs_sympathetic_warm[data_raw$perso_7 == 7] <- 0

table(data_clean$valeurs_sympathetic_warm)

## perso_8 --------------------------------------------------------
attributes(data_raw$perso_8)
table(data_raw$perso_8)

data_clean$valeurs_disorganized_careless <- NA
data_clean$valeurs_disorganized_careless[data_raw$perso_8 == 1] <- 1
data_clean$valeurs_disorganized_careless[data_raw$perso_8 == 2] <- 0.83
data_clean$valeurs_disorganized_careless[data_raw$perso_8 == 3] <- 0.67
data_clean$valeurs_disorganized_careless[data_raw$perso_8 == 4] <- 0.5
data_clean$valeurs_disorganized_careless[data_raw$perso_8 == 5] <- 0.33
data_clean$valeurs_disorganized_careless[data_raw$perso_8 == 6] <- 0.17
data_clean$valeurs_disorganized_careless[data_raw$perso_8 == 7] <- 0

table(data_clean$valeurs_disorganized_careless)

## perso_9 --------------------------------------------------------
attributes(data_raw$perso_9)
table(data_raw$perso_9)  

data_clean$valeurs_calm_emotionally_stable <- NA
data_clean$valeurs_calm_emotionally_stable[data_raw$perso_9 == 1] <- 1
data_clean$valeurs_calm_emotionally_stable[data_raw$perso_9 == 2] <- 0.83
data_clean$valeurs_calm_emotionally_stable[data_raw$perso_9 == 3] <- 0.67
data_clean$valeurs_calm_emotionally_stable[data_raw$perso_9 == 4] <- 0.5
data_clean$valeurs_calm_emotionally_stable[data_raw$perso_9 == 5] <- 0.33
data_clean$valeurs_calm_emotionally_stable[data_raw$perso_9 == 6] <- 0.17
data_clean$valeurs_calm_emotionally_stable[data_raw$perso_9 == 7] <- 0

table(data_clean$valeurs_calm_emotionally_stable)

## perso_10 --------------------------------------------------------
attributes(data_raw$perso_10)
table(data_raw$perso_10)

data_clean$valeurs_conventional_uncreative <- NA
data_clean$valeurs_conventional_uncreative[data_raw$perso_10 == 1] <- 1
data_clean$valeurs_conventional_uncreative[data_raw$perso_10 == 2] <- 0.83
data_clean$valeurs_conventional_uncreative[data_raw$perso_10 == 3] <- 0.67
data_clean$valeurs_conventional_uncreative[data_raw$perso_10 == 4] <- 0.5
data_clean$valeurs_conventional_uncreative[data_raw$perso_10 == 5] <- 0.33
data_clean$valeurs_conventional_uncreative[data_raw$perso_10 == 6] <- 0.17
data_clean$valeurs_conventional_uncreative[data_raw$perso_10 == 7] <- 0

table(data_clean$valeurs_conventional_uncreative)

## climate_worry__1 --------------------------------------------------------

attributes(data_raw$climate_worry__1)
table(data_raw$climate_worry__1)
data_clean$valeurs_worry_climate_change <- NA
data_clean$valeurs_worry_climate_change <- (data_raw$climate_worry__1 - 1) / 4
table(data_clean$valeurs_worry_climate_change)


## climate_worry__2 --------------------------------------------------------

attributes(data_raw$climate_worry__2)
table(data_raw$climate_worry__2)
data_clean$valeurs_worry_future <- NA
data_clean$valeurs_worry_future <- (data_raw$climate_worry__2 - 1) / 4
table(data_clean$valeurs_worry_future)


## climate_worry__3 --------------------------------------------------------

attributes(data_raw$climate_worry__3)
table(data_raw$climate_worry__3)
data_clean$valeurs_worry_climate_people <- NA
data_clean$valeurs_worry_climate_people <- (data_raw$climate_worry__3 - 1) / 4
table(data_clean$valeurs_worry_climate_people)
