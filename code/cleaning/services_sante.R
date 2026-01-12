# Bloc services santé ---------------------------------------------------------------

## Consultation médecin de famille ------------------------------------------------------------------------

attributes(data_raw$fournisseurs_sante_1)
table(data_raw$fournisseurs_sante_1)
table(data_raw$fournisseurs_sante_1, useNA = "always")

data_clean$service_sante_famille_general <- NA
data_clean$service_sante_famille_general[data_raw$fournisseurs_sante_1 == 1] <- 1
data_clean$service_sante_famille_general[is.na(data_raw$fournisseurs_sante_1)] <- 0
table(data_clean$service_sante_famille_general)

## Consultation spécialiste ------------------------------------------------------------------------

attributes(data_raw$fournisseurs_sante_2)
table(data_raw$fournisseurs_sante_2)
table(data_raw$fournisseurs_sante_2, useNA = "always")
data_clean$service_sante_specialist <- NA
data_clean$service_sante_specialist[data_raw$fournisseurs_sante_2 == 1] <- 1
data_clean$service_sante_specialist[is.na(data_raw$fournisseurs_sante_2)] <- 0
table(data_clean$service_sante_specialist)

## Consultation infirmier.ère practicien.ne ------------------------------------------------------------------------

attributes(data_raw$fournisseurs_sante_3)
table(data_raw$fournisseurs_sante_3)
table(data_raw$fournisseurs_sante_3, useNA = "always")
data_clean$service_sante_nurse_practitioner <- NA
data_clean$service_sante_nurse_practitioner[data_raw$fournisseurs_sante_3 == 1] <- 1
data_clean$service_sante_nurse_practitioner[is.na(data_raw$fournisseurs_sante_3)] <- 0
table(data_clean$service_sante_nurse_practitioner)

## Consultation autre (psychologue, chiro) ------------------------------------------------------------------------

attributes(data_raw$fournisseurs_sante_4)
table(data_raw$fournisseurs_sante_4)
table(data_raw$fournisseurs_sante_4, useNA = "always")
data_clean$service_sante_other <- NA
data_clean$service_sante_other[data_raw$fournisseurs_sante_4 == 1] <- 1
data_clean$service_sante_other[is.na(data_raw$fournisseurs_sante_4)] <- 0
table(data_clean$service_sante_other)

## Consultation aucun ------------------------------------------------------------------------

attributes(data_raw$fournisseurs_sante_5)
table(data_raw$fournisseurs_sante_5)
table(data_raw$fournisseurs_sante_5, useNA = "always")
data_clean$service_sante_none <- NA
data_clean$service_sante_none[data_raw$fournisseurs_sante_5 == 1] <- 1
data_clean$service_sante_none[is.na(data_raw$fournisseurs_sante_5)] <- 0
table(data_clean$service_sante_none)

## Satisfaction sexuelle ------------------------------------------------------------------------

attributes(data_raw$sexual_concerns_1)
table(data_raw$sexual_concerns_1)
data_clean$service_sante_sexual_satisfaction <- NA
data_clean$service_sante_sexual_satisfaction <- (data_raw$sexual_concerns_1) / 10
table(data_clean$service_sante_sexual_satisfaction)

## Consultation santé mentale ------------------------------------------------------------------------

attributes(data_raw$consult_sante)
table(data_raw$consult_sante)
data_clean$service_sante_mental_consult <- NA
data_clean$service_sante_mental_consult[data_raw$consult_sante == 1] <- 1
data_clean$service_sante_mental_consult[data_raw$consult_sante == 2] <- 0
table(data_clean$service_sante_mental_consult)

## Consultation santé mentale - Médecin de famille ------------------------------------------------------------------------

attributes(data_raw$consult_who_1)
table(data_raw$consult_who_1)
table(data_raw$consult_who_1, useNA = "always")
data_clean$service_sante_mental_famille_general <- NA
data_clean$service_sante_mental_famille_general[data_raw$consult_who_1 == 1] <- 1
data_clean$service_sante_mental_famille_general[is.na(data_raw$consult_who_1)] <- 0
table(data_clean$service_sante_mental_famille_general)

## Consultation santé mentale - Psychiatre ------------------------------------------------------------------------

attributes(data_raw$consult_who_2)
table(data_raw$consult_who_2)
table(data_raw$consult_who_2, useNA = "always")
data_clean$service_sante_mental_psychiatrist <- NA
data_clean$service_sante_mental_psychiatrist[data_raw$consult_who_2 == 1] <- 1
data_clean$service_sante_mental_psychiatrist[is.na(data_raw$consult_who_2)] <- 0
table(data_clean$service_sante_mental_psychiatrist)

## Consultation santé mentale - Psychologue ------------------------------------------------------------------------

attributes(data_raw$consult_who_3)
table(data_raw$consult_who_3)
table(data_raw$consult_who_3, useNA = "always")
data_clean$service_sante_mental_psychologue <- NA
data_clean$service_sante_mental_psychologue[data_raw$consult_who_3 == 1] <- 1
data_clean$service_sante_mental_psychologue[is.na(data_raw$consult_who_3)] <- 0
table(data_clean$service_sante_mental_psychologue)

## Consultation santé mentale - Infirmier.ère ------------------------------------------------------------------------

attributes(data_raw$consult_who_4)
table(data_raw$consult_who_4)
table(data_raw$consult_who_4, useNA = "always")
data_clean$service_sante_mental_nurse <- NA
data_clean$service_sante_mental_nurse[data_raw$consult_who_4 == 1] <- 1
data_clean$service_sante_mental_nurse[is.na(data_raw$consult_who_4)] <- 0
table(data_clean$service_sante_mental_nurse)

## Consultation santé mentale - Travailleur.euse social.e ------------------------------------------------------------------------

attributes(data_raw$consult_who_5)
table(data_raw$consult_who_5)
table(data_raw$consult_who_5, useNA = "always")
data_clean$service_sante_mental_social_worker <- NA
data_clean$service_sante_mental_social_worker[data_raw$consult_who_5 == 1] <- 1
data_clean$service_sante_mental_social_worker[is.na(data_raw$consult_who_5)] <- 0
table(data_clean$service_sante_mental_social_worker)

## Consultation santé mentale - Autre ------------------------------------------------------------------------

attributes(data_raw$consult_who_6)
table(data_raw$consult_who_6)
table(data_raw$consult_who_6, useNA = "always")
data_clean$service_sante_mental_other <- NA
data_clean$service_sante_mental_other[data_raw$consult_who_6 == 1] <- 1
data_clean$service_sante_mental_other[is.na(data_raw$consult_who_6)] <- 0
table(data_clean$service_sante_mental_other)

## Consultation santé mentale - Aucun ------------------------------------------------------------------------

attributes(data_raw$consult_who_7)
table(data_raw$consult_who_7)
table(data_raw$consult_who_7, useNA = "always")
data_clean$service_sante_mental_none <- NA
data_clean$service_sante_mental_none[data_raw$consult_who_7 == 1] <- 1
data_clean$service_sante_mental_none[is.na(data_raw$consult_who_7)] <- 0
table(data_clean$service_sante_mental_none)

## Consultation santé mentale - Autre ------------------------------------------------------------------------

table(data_raw$consult_who_6_TEXT)
data_clean$service_sante_mental_other_text <- (data_raw$consult_who_6_TEXT)
table(data_clean$service_sante_mental_other_text)

