# Scores of mental health based on ESSAIM scales


## Scores ----------------------------------------------------------------------

### Emotional mental health (hedonic), varies between 0 and 3 (based on 3 variables)
data_clean$emotional_health <- data_clean$bien_etre_felt_happy + data_clean$bien_etre_interested_in_life + data_clean$bien_etre_satisfied_with_life

### Positive functioning, varies between 0 and 11 (based on 11 variables)
data_clean$positive_functioning <- data_clean$bien_etre_important_contribution_society + data_clean$bien_etre_sense_belonging_community + data_clean$bien_etre_society_becomes_better_place + data_clean$bien_etre_people_fundamentally_good + data_clean$bien_etre_society_makes_sense + data_clean$bien_etre_like_personality + data_clean$bien_etre_handling_responsibilities + data_clean$bien_etre_warm_relationships_with_people + data_clean$bien_etre_better_person + data_clean$bien_etre_able_express_opinions + data_clean$bien_etre_life_has_meaning

### Total score, varies between 0 and 14
data_clean$score_tot <- data_clean$emotional_health + data_clean$positive_functioning

### Normalized score (set between 0 and 1)
data_clean$score_norm <- data_clean$score_tot/14


## Binary indicators -----------------------------------------------------------
### Santé mentale florissante, binaire. Une santé mentale florissante exige la réponse « presque tous les jours » ou « tous les jours » à au moins l'une des 3 questions sur le bien-être émotionnel/santé mentale émotionnelle, et à au moins 6 des 11 questions sur le fonctionnement positif.
data_clean$flourishing <- if_else(
  (data_clean$bien_etre_felt_happy>=0.8 | data_clean$bien_etre_interested_in_life>=0.8 |data_clean$bien_etre_satisfied_with_life>=0.8)
  & (rowSums(data_clean[, c('bien_etre_important_contribution_society', 'bien_etre_sense_belonging_community', 'bien_etre_society_becomes_better_place', 'bien_etre_people_fundamentally_good', 'bien_etre_society_makes_sense', 'bien_etre_like_personality', 'bien_etre_handling_responsibilities', 'bien_etre_warm_relationships_with_people', 'bien_etre_better_person', 'bien_etre_able_express_opinions', 'bien_etre_life_has_meaning')] >= 0.8) >= 6),
  1,0)

### Santé mentale languissante, binaire. Une santé mentale languissante exige la réponse « une fois ou deux » ou « jamais » à au moins l'une des 3 questions sur le bien-être émotionnel/santé mentale émotionnelle, et à au moins 6 des 11 questions sur le fonctionnement positif. 
data_clean$languishing  <- if_else(
  (data_clean$bien_etre_felt_happy<=0.2 | data_clean$bien_etre_interested_in_life<=0.2 |data_clean$bien_etre_satisfied_with_life<=0.2)
  & (rowSums(data_clean[, c('bien_etre_important_contribution_society', 'bien_etre_sense_belonging_community', 'bien_etre_society_becomes_better_place', 'bien_etre_people_fundamentally_good', 'bien_etre_society_makes_sense', 'bien_etre_like_personality', 'bien_etre_handling_responsibilities', 'bien_etre_warm_relationships_with_people', 'bien_etre_better_person', 'bien_etre_able_express_opinions', 'bien_etre_life_has_meaning')] <= 0.2) >= 6),
  1,0)

### Santé mentale mentale modérément bonne, binaire. Ce sont ceux qui ne sont ni dans florissante ni dans languissante.
data_clean$moderate <- if_else(
  (data_clean$flourishing==0)
  & (data_clean$languishing==0),
  1,0)