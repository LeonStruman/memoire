# comp_sante

# I focus my attention on the present moment
table(data_raw$autogestion_1)
attributes(data_raw$autogestion_1)
data_clean$comp_sante_focus_attention_present <- sondr::clean_likert_numeric_vector(data_raw$autogestion_1)
table(data_clean$comp_sante_focus_attention_present)

# I am learning to live with my strengths and weaknesses.
table(data_raw$autogestion_2)
attributes(data_raw$autogestion_2)
data_clean$comp_sante_learn_live_strengths_weaknesses <- sondr::clean_likert_numeric_vector(data_raw$autogestion_2)
table(data_clean$comp_sante_learn_live_strengths_weaknesses)

# I congratulate myself on my successes, big and small
table(data_raw$autogestion_3)
attributes(data_raw$autogestion_3)
data_clean$comp_sante_congratulate_myself_successes <- sondr::clean_likert_numeric_vector(data_raw$autogestion_3)
table(data_clean$comp_sante_congratulate_myself_successes)

# I try to love myself as I am.
table(data_raw$autogestion_4)
attributes(data_raw$autogestion_4)
data_clean$comp_sante_try_love_myself_as_i_am <- sondr::clean_likert_numeric_vector(data_raw$autogestion_4)
table(data_clean$comp_sante_try_love_myself_as_i_am)

# I consider my abilities when I make my schedule.
table(data_raw$autogestion_5)
attributes(data_raw$autogestion_5)
data_clean$comp_sante_consider_abilities_make_schedule <- sondr::clean_likert_numeric_vector(data_raw$autogestion_5)
table(data_clean$comp_sante_consider_abilities_make_schedule)

# I find comfort, people around me listen to me.
table(data_raw$autogestion_6)
attributes(data_raw$autogestion_6)
data_clean$comp_sante_find_comfort_people_around_listen <- sondr::clean_likert_numeric_vector(data_raw$autogestion_6)
table(data_clean$comp_sante_find_comfort_people_around_listen)

# I do activities that I enjoy to keep my life active
table(data_raw$autogestion_8)
attributes(data_raw$autogestion_8)
data_clean$comp_sante_do_activities_enjoy_keep_life_active <- sondr::clean_likert_numeric_vector(data_raw$autogestion_8)
table(data_clean$comp_sante_do_activities_enjoy_keep_life_active)

# I have a healthy diet
table(data_raw$autogestion_9)
attributes(data_raw$autogestion_9)
data_clean$comp_sante_have_healthy_diet <- sondr::clean_likert_numeric_vector(data_raw$autogestion_9)
table(data_clean$comp_sante_have_healthy_diet)

# I do exercises to relax (yoga, tai chi, breathing techniques, etc.)
table(data_raw$autogestion_10)
attributes(data_raw$autogestion_10)
data_clean$comp_sante_do_exercises_relax <- sondr::clean_likert_numeric_vector(data_raw$autogestion_10)
table(data_clean$comp_sante_do_exercises_relax)


# I kindly acknowledged my own challenges and difficulties.
attributes(data_raw$MSCS_1)
table(data_raw$MSCS_1)
data_clean$comp_sante_acknowledge_challenges_difficulties <- sondr::clean_likert_numeric_vector(data_raw$MSCS_1)
table(data_clean$comp_sante_acknowledge_challenges_difficulties)

# I engaged in supportive and comforting self-talk (e.g., my effort is valuable and meaningful).
attributes(data_raw$MSCS_2)
table(data_raw$MSCS_2)
data_clean$comp_sante_engage_supportive_comforting_self_talk <- sondr::clean_likert_numeric_vector(data_raw$MSCS_2)
table(data_clean$comp_sante_engage_supportive_comforting_self_talk)

# I reminded myself that failure and challenge are part of the human experience.
attributes(data_raw$MSCS_3)
table(data_raw$MSCS_3)
data_clean$comp_sante_remind_failure_challenge_part_human_experience <- sondr::clean_likert_numeric_vector(data_raw$MSCS_3)
table(data_clean$comp_sante_remind_failure_challenge_part_human_experience)

# During the past seven days, how would you rate your sleep quality overall? - Sleep quality
attributes(data_raw$sommeil_1)
table(data_raw$sommeil_1)
data_clean$comp_sante_sleep_quality <- data_raw$sommeil_1 / 10
table(data_clean$comp_sante_sleep_quality)

# Self-assess your own chronotype by choosing a graph representing the evolution of your level of alertness over the course of the day.
attributes(data_raw$chronotype)
table(data_raw$chronotype)
data_clean$comp_sante_chronotype <- NA
data_clean$comp_sante_chronotype[data_raw$chronotype == 1] <- "morning"
data_clean$comp_sante_chronotype[data_raw$chronotype == 2] <- "evening"
data_clean$comp_sante_chronotype[data_raw$chronotype == 3] <- "highly_active"
data_clean$comp_sante_chronotype[data_raw$chronotype == 4] <- "daytime_sleep"
data_clean$comp_sante_chronotype[data_raw$chronotype == 5] <- "diurnal"
data_clean$comp_sante_chronotype[data_raw$chronotype == 6] <- "moderately_active"
data_clean$comp_sante_chronotype <- factor(data_clean$comp_sante_chronotype)
table(data_clean$comp_sante_chronotype)


# How satisfied/dissatisfied are you with your current sleep pattern?
attributes(data_raw$som_satisf__isi)
table(data_raw$som_satisf__isi)
data_clean$comp_sante_sleep_pattern_satisfaction <- sondr::clean_likert_numeric_vector(data_raw$som_satisf__isi)
data_clean$comp_sante_sleep_pattern_satisfaction <- sondr::finverser(data_clean$comp_sante_sleep_pattern_satisfaction)
table(data_clean$comp_sante_sleep_pattern_satisfaction)


# To what extent do you consider your sleep problem to interfere with your daily functioning (e.g. fatigue, concentration, memory, mood)
attributes(data_raw$som_perturb)
table(data_raw$som_perturb)
data_clean$comp_sante_sleep_problem_interfere_daily_functioning <- sondr::clean_likert_numeric_vector(data_raw$som_perturb)
table(data_clean$comp_sante_sleep_problem_interfere_daily_functioning)

# Generally speaking, how many servings of fruit and vegetables do you eat on average each day
attributes(data_raw$alim_fruits__leg)
table(data_raw$alim_fruits__leg)
data_clean$comp_sante_fruit_vegetables_servings <- NA
data_clean$comp_sante_fruit_vegetables_servings[data_raw$alim_fruits__leg == 9] <- "dont_consume"
data_clean$comp_sante_fruit_vegetables_servings[data_raw$alim_fruits__leg == 1] <- "less_than_1_portion_per_day"
data_clean$comp_sante_fruit_vegetables_servings[data_raw$alim_fruits__leg == 2] <- "1_portions_per_day"
data_clean$comp_sante_fruit_vegetables_servings[data_raw$alim_fruits__leg == 3] <- "2_portions_per_day"
data_clean$comp_sante_fruit_vegetables_servings[data_raw$alim_fruits__leg == 4] <- "3_portions_per_day"
data_clean$comp_sante_fruit_vegetables_servings[data_raw$alim_fruits__leg == 5] <- "4_portions_per_day"
data_clean$comp_sante_fruit_vegetables_servings[data_raw$alim_fruits__leg == 6] <- "5_portions_per_day"
data_clean$comp_sante_fruit_vegetables_servings[data_raw$alim_fruits__leg == 7] <- "6_portions_per_day"
data_clean$comp_sante_fruit_vegetables_servings[data_raw$alim_fruits__leg == 8] <- "7_plus_portions_per_day"
data_clean$comp_sante_fruit_vegetables_servings <- factor(data_clean$comp_sante_fruit_vegetables_servings)
table(data_clean$comp_sante_fruit_vegetables_servings)

# Over the past 12 months, how often have you consumed alcohol?
attributes(data_raw$conso_freq)
table(data_raw$conso_freq)
data_clean$comp_sante_alcohol_freq <- sondr::clean_likert_numeric_vector(data_raw$conso_freq)
table(data_clean$comp_sante_alcohol_freq)

# In the last 12 months, how often have you : - used cannabis (marijuana, hashish, oil, wax, cannabinoids synth.)
attributes(data_raw$conso_drogue__1)
table(data_raw$conso_drogue__1)
data_clean$comp_sante_cannabis_freq <- sondr::clean_likert_numeric_vector(data_raw$conso_drogue__1)
table(data_clean$comp_sante_cannabis_freq)

data_clean$comp_sante_cannabis_freq_bin <- NA
data_clean$comp_sante_cannabis_freq_bin[data_raw$conso_drogue__1 == 1] <- 0
data_clean$comp_sante_cannabis_freq_bin[data_raw$conso_drogue__1 %in% 2:7] <- 1
table(data_clean$comp_sante_cannabis_freq_bin)

# In the last 12 months, how often have you : - used amphetamines (methamphetamines, speeds, Vyvanse®, Concerta®, Ritalin®)?
attributes(data_raw$conso_drogue__2)
table(data_raw$conso_drogue__2)
data_clean$comp_sante_amphetamines_freq <- sondr::clean_likert_numeric_vector(data_raw$conso_drogue__2)
table(data_clean$comp_sante_amphetamines_freq)

data_clean$comp_sante_amphetamines_freq_bin <- NA
data_clean$comp_sante_amphetamines_freq_bin[data_raw$conso_drogue__2 == 1] <- 0
data_clean$comp_sante_amphetamines_freq_bin[data_raw$conso_drogue__2 %in% 2:7] <- 1
table(data_clean$comp_sante_amphetamines_freq_bin)


# In the last 12 months, how often have you : - used tobacco (e.g. cigarettes, cigars, pipe, vaporizer)?
attributes(data_raw$conso_drogue__3)
table(data_raw$conso_drogue__3)
data_clean$comp_sante_tobacco_freq <- sondr::clean_likert_numeric_vector(data_raw$conso_drogue__3)
table(data_clean$comp_sante_tobacco_freq)

data_clean$comp_sante_tobacco_freq_bin <- NA
data_clean$comp_sante_tobacco_freq_bin[data_raw$conso_drogue__3 == 1] <- 0
data_clean$comp_sante_tobacco_freq_bin[data_raw$conso_drogue__3 %in% 2:7] <- 1
table(data_clean$comp_sante_tobacco_freq_bin)

# Activité physique High intensity

attributes(data_raw$act_physique_godin_1)
table(data_raw$act_physique_godin_1)
data_clean$comp_sante_physical_activity_high_intensity <- NA
valid_indices <- which(data_raw$act_physique_godin_1 %in% 0:14)
data_clean$comp_sante_physical_activity_high_intensity[valid_indices] <- as.numeric(data_raw$act_physique_godin_1[valid_indices])
table(data_clean$comp_sante_physical_activity_high_intensity)

# Activité physique moderate activity

attributes(data_raw$act_physique_godin_2)
table(data_raw$act_physique_godin_2)
data_clean$comp_sante_physical_activity_moderate_intensity <- NA
valid_indices <- which(data_raw$act_physique_godin_2 %in% 0:28)
data_clean$comp_sante_physical_activity_moderate_intensity[valid_indices] <- as.numeric(data_raw$act_physique_godin_2[valid_indices])
table(data_clean$comp_sante_physical_activity_moderate_intensity)

# Activité physique low intensity

attributes(data_raw$act_physique_godin_3)
table(data_raw$act_physique_godin_3)
data_clean$comp_sante_physical_activity_low_intensity <- NA
valid_indices <- which(data_raw$act_physique_godin_3 %in% 0:28)
data_clean$comp_sante_physical_activity_low_intensity[valid_indices] <- as.numeric(data_raw$act_physique_godin_3[valid_indices])
table(data_clean$comp_sante_physical_activity_low_intensity)


