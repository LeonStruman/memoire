# Bloc Comport_Socio

## nb_friends_home_/ -------------------------------------------------------

attributes(data_raw$nb_friends_home_1)
table(data_raw$nb_friends_home_1)
data_clean$comport_socio_people_count_on <- NA
data_clean$comport_socio_people_count_on <- (data_raw$nb_friends_home_1 - 1) / 3
table(data_clean$comport_socio_people_count_on)



## nb_friends_home_2 -------------------------------------------------------

attributes(data_raw$nb_friends_home_2)
table(data_raw$nb_friends_home_2)
data_clean$comport_socio_emotional_security <- NA
data_clean$comport_socio_emotional_security <- (data_raw$nb_friends_home_2 - 1) / 3
table(data_clean$comport_socio_emotional_security)


## nb_friends_home_3 -------------------------------------------------------

attributes(data_raw$nb_friends_home_3)
table(data_raw$nb_friends_home_3)
data_clean$comport_socio_life_decision <- NA
data_clean$comport_socio_life_decision <- (data_raw$nb_friends_home_3 - 1) / 3
table(data_clean$comport_socio_life_decision)


## nb_friends_home_4 -------------------------------------------------------

attributes(data_raw$nb_friends_home_4)
table(data_raw$nb_friends_home_4)
data_clean$comport_socio_skills_relationship <- NA
data_clean$comport_socio_skills_relationship <- (data_raw$nb_friends_home_4 - 1) / 3
table(data_clean$comport_socio_skills_relationship)


## nb_friends_home_5 -------------------------------------------------------

attributes(data_raw$nb_friends_home_5)
table(data_raw$nb_friends_home_5)
data_clean$comport_socio_group_beliefs <- NA
data_clean$comport_socio_group_beliefs <- (data_raw$nb_friends_home_5 - 1) / 3 
table(data_clean$comport_socio_group_beliefs)


## nb_friends_dispo --------------------------------------------------------

attributes(data_raw$nb_friends_dispo)
table(data_raw$nb_friends_dispo)
data_clean$comport_socio_immediate_friends <- NA
data_clean$comport_socio_immediate_friends <- data_raw$nb_friends_dispo
data_clean$comport_socio_immediate_friends[data_raw$nb_friends_dispo > 100] <- NA
table(data_clean$comport_socio_immediate_friends)


## nb_friends_dispo_2 ------------------------------------------------------

attributes(data_raw$nb_friends_dispo_2)
table(data_raw$nb_friends_dispo_2)
data_clean$comport_socio_friends_dispo <- NA
data_clean$comport_socio_friends_dispo <- (data_raw$nb_friends_dispo_2 - 1) / 2
table(data_clean$comport_socio_friends_dispo)


## poly_likert_1 -----------------------------------------------------------

attributes(data_raw$poly_likert_1)
table(data_raw$poly_likert_1)
data_clean$comport_socio_success_poly <- NA
data_clean$comport_socio_success_poly <- (data_raw$poly_likert_1 - 1) / 6
table(data_clean$comport_socio_success_poly)


## poly_likert_2 -----------------------------------------------------------

attributes(data_raw$poly_likert_2)
table(data_raw$poly_likert_2)
data_clean$comport_socio_child_poly_parent <- NA
data_clean$comport_socio_child_poly_parent <- (data_raw$poly_likert_2 - 1) / 6
table(data_clean$comport_socio_child_poly_parent)


## poly_likert_3 -----------------------------------------------------------

attributes(data_raw$poly_likert_3)
table(data_raw$poly_likert_3)
data_clean$comport_socio_poly_cheat <- NA
data_clean$comport_socio_poly_cheat <- (data_raw$poly_likert_3 - 1) / 6
table(data_clean$comport_socio_poly_cheat)


## poly_rel ----------------------------------------------------------------

attributes(data_raw$poly_rel)
table(data_raw$poly_rel)
data_clean$comport_socio_poly_relation <- NA
data_clean$comport_socio_poly_relation <- (data_raw$poly_rel - 1) / 3
table(data_clean$comport_socio_poly_relation)

