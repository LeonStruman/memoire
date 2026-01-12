library(sondr)
# Bloc bien-Être

## CSM_Comparison --------------------------------------------------------------

attributes(data_raw$CSM_Comparison)
table(data_raw$CSM_Comparison)

data_clean$bien_etre_general_health <- (data_raw$CSM_Comparison - 1) / 4
table(data_clean$bien_etre_general_health)

## CSM_QA1 ---------------------------------------------------------------------
### CSM_QA1_1 ------------------------------------------------------------------

attributes(data_raw$CSM_QA1_1)
table(data_raw$CSM_QA1_1)

data_clean$bien_etre_felt_happy <- (data_raw$CSM_QA1_1 - 1) / 5
table(data_clean$bien_etre_felt_happy)

### CSM_QA1_2 ------------------------------------------------------------------

attributes(data_raw$CSM_QA1_2)
table(data_raw$CSM_QA1_2)

data_clean$bien_etre_interested_in_life <- (data_raw$CSM_QA1_2 - 1) / 5
table(data_clean$bien_etre_interested_in_life)

### CSM_QA1_3 ------------------------------------------------------------------

attributes(data_raw$CSM_QA1_3)
table(data_raw$CSM_QA1_3)

data_clean$bien_etre_satisfied_with_life <- (data_raw$CSM_QA1_3 - 1) / 5
table(data_clean$bien_etre_satisfied_with_life)

## CSM_QA2 -------------------------------------------------------------------
### CSM_QA2_1 ----------------------------------------------------------------

attributes(data_raw$CSM_QA2_1)
table(data_raw$CSM_QA2_1)

data_clean$bien_etre_important_contribution_society <- NA
data_clean$bien_etre_important_contribution_society[data_raw$CSM_QA2_1 == 1] <- 1
data_clean$bien_etre_important_contribution_society[data_raw$CSM_QA2_1 == 2] <- 0.8
data_clean$bien_etre_important_contribution_society[data_raw$CSM_QA2_1 == 3] <- 0.6
data_clean$bien_etre_important_contribution_society[data_raw$CSM_QA2_1 == 4] <- 0.4
data_clean$bien_etre_important_contribution_society[data_raw$CSM_QA2_1 == 5] <- 0.2
data_clean$bien_etre_important_contribution_society[data_raw$CSM_QA2_1 == 6] <- 0

table(data_clean$bien_etre_important_contribution_society)

### CSM_QA2_2 ----------------------------------------------------------------

attributes(data_raw$CSM_QA2_2)
table(data_raw$CSM_QA2_2)

data_clean$bien_etre_sense_belonging_community <- NA
data_clean$bien_etre_sense_belonging_community[data_raw$CSM_QA2_2 == 1] <- 1
data_clean$bien_etre_sense_belonging_community[data_raw$CSM_QA2_2 == 2] <- 0.8
data_clean$bien_etre_sense_belonging_community[data_raw$CSM_QA2_2 == 3] <- 0.6
data_clean$bien_etre_sense_belonging_community[data_raw$CSM_QA2_2 == 4] <- 0.4
data_clean$bien_etre_sense_belonging_community[data_raw$CSM_QA2_2 == 5] <- 0.2
data_clean$bien_etre_sense_belonging_community[data_raw$CSM_QA2_2 == 6] <- 0

table(data_clean$bien_etre_sense_belonging_community)

### CSM_QA2_3 ----------------------------------------------------------------

attributes(data_raw$CSM_QA2_3)
table(data_raw$CSM_QA2_3)

data_clean$bien_etre_society_becomes_better_place <- NA
data_clean$bien_etre_society_becomes_better_place[data_raw$CSM_QA2_3 == 1] <- 1
data_clean$bien_etre_society_becomes_better_place[data_raw$CSM_QA2_3 == 2] <- 0.8
data_clean$bien_etre_society_becomes_better_place[data_raw$CSM_QA2_3 == 3] <- 0.6
data_clean$bien_etre_society_becomes_better_place[data_raw$CSM_QA2_3 == 4] <- 0.4
data_clean$bien_etre_society_becomes_better_place[data_raw$CSM_QA2_3 == 5] <- 0.2
data_clean$bien_etre_society_becomes_better_place[data_raw$CSM_QA2_3 == 6] <- 0

table(data_clean$bien_etre_society_becomes_better_place)

### CSM_QA2_4 ----------------------------------------------------------------

attributes(data_raw$CSM_QA2_4)
table(data_raw$CSM_QA2_4)

data_clean$bien_etre_people_fundamentally_good <- NA
data_clean$bien_etre_people_fundamentally_good[data_raw$CSM_QA2_4 == 1] <- 1
data_clean$bien_etre_people_fundamentally_good[data_raw$CSM_QA2_4 == 2] <- 0.8
data_clean$bien_etre_people_fundamentally_good[data_raw$CSM_QA2_4 == 3] <- 0.6
data_clean$bien_etre_people_fundamentally_good[data_raw$CSM_QA2_4 == 4] <- 0.4
data_clean$bien_etre_people_fundamentally_good[data_raw$CSM_QA2_4 == 5] <- 0.2
data_clean$bien_etre_people_fundamentally_good[data_raw$CSM_QA2_4 == 6] <- 0

table(data_clean$bien_etre_people_fundamentally_good)

### CSM_QA2_5 ----------------------------------------------------------------

attributes(data_raw$CSM_QA2_5)
table(data_raw$CSM_QA2_5)

data_clean$bien_etre_society_makes_sense <- NA
data_clean$bien_etre_society_makes_sense[data_raw$CSM_QA2_5 == 1] <- 1
data_clean$bien_etre_society_makes_sense[data_raw$CSM_QA2_5 == 2] <- 0.8
data_clean$bien_etre_society_makes_sense[data_raw$CSM_QA2_5 == 3] <- 0.6
data_clean$bien_etre_society_makes_sense[data_raw$CSM_QA2_5 == 4] <- 0.4
data_clean$bien_etre_society_makes_sense[data_raw$CSM_QA2_5 == 5] <- 0.2
data_clean$bien_etre_society_makes_sense[data_raw$CSM_QA2_5 == 6] <- 0

table(data_clean$bien_etre_society_makes_sense)

### CSM_QA2_6 ----------------------------------------------------------------

attributes(data_raw$CSM_QA2_6)
table(data_raw$CSM_QA2_6)

data_clean$bien_etre_like_personality <- NA
data_clean$bien_etre_like_personality[data_raw$CSM_QA2_6 == 1] <- 1
data_clean$bien_etre_like_personality[data_raw$CSM_QA2_6 == 2] <- 0.8
data_clean$bien_etre_like_personality[data_raw$CSM_QA2_6 == 3] <- 0.6
data_clean$bien_etre_like_personality[data_raw$CSM_QA2_6 == 4] <- 0.4
data_clean$bien_etre_like_personality[data_raw$CSM_QA2_6 == 5] <- 0.2
data_clean$bien_etre_like_personality[data_raw$CSM_QA2_6 == 6] <- 0

table(data_clean$bien_etre_like_personality)

### CSM_QA2_7 ----------------------------------------------------------------

attributes(data_raw$CSM_QA2_7)
table(data_raw$CSM_QA2_7, useNA = "always")

data_clean$bien_etre_handling_responsibilities <- NA
data_clean$bien_etre_handling_responsibilities[data_raw$CSM_QA2_7 == 1] <- 1
data_clean$bien_etre_handling_responsibilities[data_raw$CSM_QA2_7 == 2] <- 0.8
data_clean$bien_etre_handling_responsibilities[data_raw$CSM_QA2_7 == 3] <- 0.6
data_clean$bien_etre_handling_responsibilities[data_raw$CSM_QA2_7 == 4] <- 0.4
data_clean$bien_etre_handling_responsibilities[data_raw$CSM_QA2_7 == 5] <- 0.2
data_clean$bien_etre_handling_responsibilities[data_raw$CSM_QA2_7 == 6] <- 0

table(data_clean$bien_etre_handling_responsibilities)

### CSM_QA2_8 ----------------------------------------------------------------

attributes(data_raw$CSM_QA2_8)
table(data_raw$CSM_QA2_8)

data_clean$bien_etre_warm_relationships_with_people <- NA
data_clean$bien_etre_warm_relationships_with_people[data_raw$CSM_QA2_8 == 1] <- 1
data_clean$bien_etre_warm_relationships_with_people[data_raw$CSM_QA2_8 == 2] <- 0.8
data_clean$bien_etre_warm_relationships_with_people[data_raw$CSM_QA2_8 == 3] <- 0.6
data_clean$bien_etre_warm_relationships_with_people[data_raw$CSM_QA2_8 == 4] <- 0.4
data_clean$bien_etre_warm_relationships_with_people[data_raw$CSM_QA2_8 == 5] <- 0.2
data_clean$bien_etre_warm_relationships_with_people[data_raw$CSM_QA2_8 == 6] <- 0

table(data_clean$bien_etre_warm_relationships_with_people)

### CSM_QA2_9 ----------------------------------------------------------------

attributes(data_raw$CSM_QA2_9)
table(data_raw$CSM_QA2_9)

data_clean$bien_etre_better_person <- NA
data_clean$bien_etre_better_person[data_raw$CSM_QA2_9 == 1] <- 1
data_clean$bien_etre_better_person[data_raw$CSM_QA2_9 == 2] <- 0.8
data_clean$bien_etre_better_person[data_raw$CSM_QA2_9 == 3] <- 0.6
data_clean$bien_etre_better_person[data_raw$CSM_QA2_9 == 4] <- 0.4
data_clean$bien_etre_better_person[data_raw$CSM_QA2_9 == 5] <- 0.2
data_clean$bien_etre_better_person[data_raw$CSM_QA2_9 == 6] <- 0

table(data_clean$bien_etre_better_person)

### CSM_QA2_10 ---------------------------------------------------------------

attributes(data_raw$CSM_QA2_10)
table(data_raw$CSM_QA2_10)

data_clean$bien_etre_able_express_opinions <- NA
data_clean$bien_etre_able_express_opinions[data_raw$CSM_QA2_10 == 1] <- 1
data_clean$bien_etre_able_express_opinions[data_raw$CSM_QA2_10 == 2] <- 0.8
data_clean$bien_etre_able_express_opinions[data_raw$CSM_QA2_10 == 3] <- 0.6
data_clean$bien_etre_able_express_opinions[data_raw$CSM_QA2_10 == 4] <- 0.4
data_clean$bien_etre_able_express_opinions[data_raw$CSM_QA2_10 == 5] <- 0.2
data_clean$bien_etre_able_express_opinions[data_raw$CSM_QA2_10 == 6] <- 0

table(data_clean$bien_etre_able_express_opinions)

### CSM_QA2_11 ---------------------------------------------------------------

attributes(data_raw$CSM_QA2_11)
table(data_raw$CSM_QA2_11)

data_clean$bien_etre_life_has_meaning <- NA
data_clean$bien_etre_life_has_meaning[data_raw$CSM_QA2_11 == 1] <- 1
data_clean$bien_etre_life_has_meaning[data_raw$CSM_QA2_11 == 2] <- 0.8
data_clean$bien_etre_life_has_meaning[data_raw$CSM_QA2_11 == 3] <- 0.6
data_clean$bien_etre_life_has_meaning[data_raw$CSM_QA2_11 == 4] <- 0.4
data_clean$bien_etre_life_has_meaning[data_raw$CSM_QA2_11 == 5] <- 0.2
data_clean$bien_etre_life_has_meaning[data_raw$CSM_QA2_11 == 6] <- 0

table(data_clean$bien_etre_life_has_meaning)

## PHQ4_1 ----------------------------------------------------------------------

attributes(data_raw$PHQ4_1)
table(data_raw$PHQ4_1)
data_clean$bien_etre_pleasure_doing_things <- NA
data_clean$bien_etre_pleasure_doing_things <- (data_raw$PHQ4_1 - 1) / 3
table(data_clean$bien_etre_pleasure_doing_things)


## PHQ4_2 ------------------------------------------------------------------

attributes(data_raw$PHQ4_2)
table(data_raw$PHQ4_2)
data_clean$bien_etre_sad <- NA
data_clean$bien_etre_sad <- (data_raw$PHQ4_2 - 1) / 3
table(data_clean$bien_etre_sad)


## PHQ4_3 ------------------------------------------------------------------

attributes(data_raw$PHQ4_3)
table(data_raw$PHQ4_3)
data_clean$bien_etre_nervous <- NA
data_clean$bien_etre_nervous <- (data_raw$PHQ4_3 - 1) / 3
table(data_clean$bien_etre_nervous)


## PHQ4_4 ------------------------------------------------------------------

attributes(data_raw$PHQ4_4)
table(data_raw$PHQ4_4)
data_clean$bien_etre_worries <- NA
data_clean$bien_etre_worries <- (data_raw$PHQ4_4 - 1) / 3
table(data_clean$bien_etre_worries)


## hsa_stress__vie -------------------------------------------------------------

attributes(data_raw$hsa_stress__vie)
table(data_raw$hsa_stress__vie)
data_clean$bien_etre_stress_vie <- NA
data_clean$bien_etre_stress_vie <- sondr::clean_likert_numeric_vector(data_raw$hsa_stress__vie)
table(data_clean$bien_etre_stress_vie)


## maladies_1 -------------------------------------------------------------------- 

attributes(data_raw$maladies_1)
table(data_raw$maladies_1)
data_clean$bien_etre_chronic_problems <- data_raw$maladies_1
data_clean$bien_etre_chronic_problems[is.na(data_raw$maladies_1)] <- 0
table(data_clean$bien_etre_chronic_problems)


## maladies_2 --------------------------------------------------------------

attributes(data_raw$maladies_2)
table(data_raw$maladies_2)
data_clean$bien_etre_asthma <- data_raw$maladies_2
data_clean$bien_etre_asthma[is.na(data_raw$maladies_2)] <- 0
table(data_clean$bien_etre_asthma)


## maladies_3 --------------------------------------------------------------

attributes(data_raw$maladies_3)
table(data_raw$maladies_3)
data_clean$bien_etre_copd <- data_raw$maladies_3
data_clean$bien_etre_copd[is.na(data_raw$maladies_3)] <- 0
table(data_clean$bien_etre_copd)


## maladies_4 --------------------------------------------------------------

attributes(data_raw$maladies_4)      
table(data_raw$maladies_4)
data_clean$bien_etre_sleep_apnea <- data_raw$maladies_4
data_clean$bien_etre_sleep_apnea[is.na(data_raw$maladies_4)] <- 0
table(data_clean$bien_etre_sleep_apnea)


## maladies_5 --------------------------------------------------------------

attributes(data_raw$maladies_5)
table(data_raw$maladies_5)
data_clean$bien_etre_fibromyalgia <- data_raw$maladies_5
data_clean$bien_etre_fibromyalgia[is.na(data_raw$maladies_5)] <- 0
table(data_clean$bien_etre_fibromyalgia)


## maladies_6 --------------------------------------------------------------

attributes(data_raw$maladies_6)
table(data_raw$maladies_6)
data_clean$bien_etre_joint_problems <- data_raw$maladies_6
data_clean$bien_etre_joint_problems[is.na(data_raw$maladies_6)] <- 0
table(data_clean$bien_etre_joint_problems)


## maladies_7 --------------------------------------------------------------

attributes(data_raw$maladies_7)
table(data_raw$maladies_7)
data_clean$bien_etre_back_pain <- data_raw$maladies_7
data_clean$bien_etre_back_pain[is.na(data_raw$maladies_7)] <- 0
table(data_clean$bien_etre_back_pain)


## maladies_8 --------------------------------------------------------------

attributes(data_raw$maladies_8)
table(data_raw$maladies_8)
data_clean$bien_etre_osteoporosis <- data_raw$maladies_8
data_clean$bien_etre_osteoporosis[is.na(data_raw$maladies_8)] <- 0
table(data_clean$bien_etre_osteoporosis)


## maladies_9 --------------------------------------------------------------

attributes(data_raw$maladies_9)
table(data_raw$maladies_9)
data_clean$bien_etre_hypertension <- data_raw$maladies_9
data_clean$bien_etre_hypertension[is.na(data_raw$maladies_9)] <- 0
table(data_clean$bien_etre_hypertension)


## maladies_10 -------------------------------------------------------------

attributes(data_raw$maladies_10)
table(data_raw$maladies_10)
data_clean$bien_etre_cholesterol_lipids <- data_raw$maladies_10
data_clean$bien_etre_cholesterol_lipids[is.na(data_raw$maladies_10)] <- 0
table(data_clean$bien_etre_cholesterol_lipids)


## maladies_11 -------------------------------------------------------------

attributes(data_raw$maladies_11)
table(data_raw$maladies_11)
data_clean$bien_etre_celiac_disease <- data_raw$maladies_11
data_clean$bien_etre_celiac_disease[is.na(data_raw$maladies_11)] <- 0
table(data_clean$bien_etre_celiac_disease)


## maladies_12 -------------------------------------------------------------

attributes(data_raw$maladies_12)
table(data_raw$maladies_12)
data_clean$bien_etre_instestinal_disease <- data_raw$maladies_12
data_clean$bien_etre_instestinal_disease[is.na(data_raw$maladies_12)] <- 0
table(data_clean$bien_etre_instestinal_disease)


## maladies_13 ------------------------------------------------------------

attributes(data_raw$maladies_13)
table(data_raw$maladies_13)
data_clean$bien_etre_heart_disease <- data_raw$maladies_13
data_clean$bien_etre_heart_disease[is.na(data_raw$maladies_13)] <- 0
table(data_clean$bien_etre_heart_disease)


## maladies_14 -------------------------------------------------------------

attributes(data_raw$maladies_14)
table(data_raw$maladies_14, useNA = "always")
data_clean$bien_etre_stroke_disorders <- data_raw$maladies_14
data_clean$bien_etre_stroke_disorders[is.na(data_raw$maladies_14)] <- 0
table(data_clean$bien_etre_stroke_disorders)


## maladies_15 -------------------------------------------------------------

attributes(data_raw$maladies_15)
table(data_raw$maladies_15)
data_clean$bien_etre_diabetes <- data_raw$maladies_15
data_clean$bien_etre_diabetes[is.na(data_raw$maladies_15)] <- 0
table(data_clean$bien_etre_diabetes)


## maladies_16 -------------------------------------------------------------

attributes(data_raw$maladies_16)
table(data_raw$maladies_16)
data_clean$bien_etre_cancer <- data_raw$maladies_16
data_clean$bien_etre_cancer[is.na(data_raw$maladies_16)] <- 0
table(data_clean$bien_etre_cancer)


## maladies_17 -------------------------------------------------------------

attributes(data_raw$maladies_17)
table(data_raw$maladies_17)
data_clean$bien_etre_thyroid_problems <- data_raw$maladies_17
data_clean$bien_etre_thyroid_problems[is.na(data_raw$maladies_17)] <- 0
table(data_clean$bien_etre_thyroid_problems)


## maladies_18 -------------------------------------------------------------

attributes(data_raw$maladies_18)
table(data_raw$maladies_18, useNA = "always")
data_clean$bien_etre_alzheimers_dementia <- data_raw$maladies_18
data_clean$bien_etre_alzheimers_dementia[is.na(data_raw$maladies_18)] <- 0
table(data_clean$bien_etre_alzheimers_dementia)


## maladies_19 -------------------------------------------------------------

attributes(data_raw$maladies_19)
table(data_raw$maladies_19)
data_clean$bien_etre_chronic_fatigue <- data_raw$maladies_19
data_clean$bien_etre_chronic_fatigue[is.na(data_raw$maladies_19)] <- 0
table(data_clean$bien_etre_chronic_fatigue)


## maladies_20 -------------------------------------------------------------

attributes(data_raw$maladies_20)
table(data_raw$maladies_20)
data_clean$bien_etre_renal_failure <- data_raw$maladies_20
data_clean$bien_etre_renal_failure[is.na(data_raw$maladies_20)] <- 0
table(data_clean$bien_etre_renal_failure)


## maladies_21 -------------------------------------------------------------

attributes(data_raw$maladies_21)
table(data_raw$maladies_21)
data_clean$bien_etre_liver_disease <- data_raw$maladies_21
data_clean$bien_etre_liver_disease[is.na(data_raw$maladies_21)] <- 0
table(data_clean$bien_etre_liver_disease)


## maladies_22 -------------------------------------------------------------

attributes(data_raw$maladies_22)
table(data_raw$maladies_22)
data_clean$bien_etre_abdomen_ulcers <- data_raw$maladies_22
data_clean$bien_etre_abdomen_ulcers[is.na(data_raw$maladies_22)] <- 0
table(data_clean$bien_etre_abdomen_ulcers)


## maladies_23 -------------------------------------------------------------

attributes(data_raw$maladies_23)
table(data_raw$maladies_23)
data_clean$bien_etre_allergies <- data_raw$maladies_23
data_clean$bien_etre_allergies[is.na(data_raw$maladies_23)] <- 0
table(data_clean$bien_etre_allergies)





