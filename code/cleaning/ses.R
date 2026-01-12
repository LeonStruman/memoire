# Bloc SES

## Genre -----------------------------------------------------------------------

## Male

attributes(data_raw$genre)
table(data_raw$genre)
data_clean$ses_male_bin <- NA
data_clean$ses_male_bin[data_raw$genre == 1] <- 1
data_clean$ses_male_bin[data_raw$genre != 1] <- 0
table(data_clean$ses_male_bin)

## Female

data_clean$ses_female_bin <- NA
data_clean$ses_female_bin[data_raw$genre == 2] <- 1
data_clean$ses_female_bin[data_raw$genre != 2] <- 0
table(data_clean$ses_female_bin)

## Factor 

data_clean$ses_genre <- NA
data_clean$ses_genre[data_raw$genre == 1] <- "male"
data_clean$ses_genre[data_raw$genre == 2] <- "female"
data_clean$ses_genre[data_raw$genre == 3] <- "transgender_male"
data_clean$ses_genre[data_raw$genre == 4] <- "transgender_woman"
data_clean$ses_genre[data_raw$genre == 5] <- "non_binary"
data_clean$ses_genre[data_raw$genre == 6] <- "queer"
data_clean$ses_genre[data_raw$genre == 7] <- "agender"
data_clean$ses_genre[data_raw$genre == 8] <- "other"
table(data_clean$ses_genre)


## Orientation sexuelle --------------------------------------------------------

attributes(data_raw$orientation)
data_clean$ses_sexual_orientation <- NA
data_clean$ses_sexual_orientation[data_raw$orientation == 1] <- "heterosexual"
data_clean$ses_sexual_orientation[data_raw$orientation == 2] <- "bisexual"
data_clean$ses_sexual_orientation[data_raw$orientation == 3] <- "gay_lesbian"
data_clean$ses_sexual_orientation[data_raw$orientation == 4] <- "other"
data_clean$ses_sexual_orientation <- factor(data_clean$ses_sexual_orientation)
table(data_clean$ses_sexual_orientation)

## Âge -------------------------------------------------------------------------

attributes(data_raw$age)
table(data_raw$age)

data_clean$ses_age <- NA
data_clean$ses_age <- data_raw$age
table(data_clean$ses_age)

## Age group

data_clean$ses_age_group <- NA
data_clean$ses_age_group[data_raw$age < 18] <- "under_18"
data_clean$ses_age_group[data_raw$age >= 18 & data_raw$age <= 19] <- "18_19"
data_clean$ses_age_group[data_raw$age >= 20 & data_raw$age <= 24] <- "20_24"
data_clean$ses_age_group[data_raw$age >= 25 & data_raw$age <= 29] <- "25_29"
data_clean$ses_age_group[data_raw$age >= 30 & data_raw$age <= 34] <- "30_34"
data_clean$ses_age_group[data_raw$age >= 35 & data_raw$age <= 39] <- "35_39"
data_clean$ses_age_group[data_raw$age >= 40 & data_raw$age <= 44] <- "40_44"
data_clean$ses_age_group[data_raw$age >= 45 & data_raw$age <= 49] <- "45_49"
data_clean$ses_age_group[data_raw$age >= 50 & data_raw$age <= 54] <- "50_54"
data_clean$ses_age_group[data_raw$age >= 55 & data_raw$age <= 59] <- "55_59"
data_clean$ses_age_group[data_raw$age >= 60 & data_raw$age <= 64] <- "60_64"
data_clean$ses_age_group[data_raw$age >= 65 & data_raw$age <= 69] <- "65_69"
data_clean$ses_age_group[data_raw$age >= 70 & data_raw$age <= 74] <- "70_74"
data_clean$ses_age_group[data_raw$age >= 75 & data_raw$age <= 79] <- "75_79"
data_clean$ses_age_group[data_raw$age >= 80 & data_raw$age <= 84] <- "80_84"
data_clean$ses_age_group[data_raw$age > 84] <- "over_85"
data_clean$ses_age_group <- factor(data_clean$ses_age_group, levels = c("under_18",
                                                                        "18_19",
                                                                        "20_24",
                                                                        "25_29",
                                                                        "30_34",
                                                                        "35_39",
                                                                        "40_44",
                                                                        "45_49",
                                                                        "50_54",
                                                                        "55_59",
                                                                        "60_64",
                                                                        "65_69",
                                                                        "70_74",
                                                                        "75_79",
                                                                        "80_84",
                                                                        "over_85"))

table(data_clean$ses_age_group)

## Langue maternelle -----------------------------------------------------------

attributes(data_raw$langue_maternelle)
table(data_raw$langue_maternelle)
data_clean$ses_langue_maternelle <- NA
data_clean$ses_langue_maternelle[data_raw$langue_maternelle == 1] <- "francais"
data_clean$ses_langue_maternelle[data_raw$langue_maternelle == 2] <- "anglais"
data_clean$ses_langue_maternelle[data_raw$langue_maternelle == 3] <- "autre"
table(data_clean$ses_langue_maternelle)
data_clean$ses_langue_maternelle <- factor(data_clean$ses_langue_maternelle)
table(data_clean$ses_langue_maternelle)

## Occupation ------------------------------------------------------------------

# Occupation 1

attributes(data_raw$occupation_1)
table(data_raw$occupation_1)
data_clean$ses_occupation_self_employed <- 0
data_clean$ses_occupation_self_employed[data_raw$occupation_1 == 1] <- 1
table(data_clean$ses_occupation_self_employed)

# Occupation 2

attributes(data_raw$occupation_2)
table(data_raw$occupation_2)
data_clean$ses_occupation_salaried_work <- 0
data_clean$ses_occupation_salaried_work[data_raw$occupation_2 == 1] <- 1
table(data_clean$ses_occupation_salaried_work)

# Occupation 3

attributes(data_raw$occupation_3)
table(data_raw$occupation_3)
data_clean$ses_occupation_retired <- 0
data_clean$ses_occupation_retired[data_raw$occupation_3 == 1] <- 1
table(data_clean$ses_occupation_retired)

# Occupation 4

attributes(data_raw$occupation_4)
table(data_raw$occupation_4)
data_clean$ses_occupation_unemployed <- 0
data_clean$ses_occupation_unemployed[data_raw$occupation_4 == 1] <- 1
table(data_clean$ses_occupation_unemployed)

# Occupation 5

attributes(data_raw$occupation_5)
table(data_raw$occupation_5)
data_clean$ses_occupation_student <- 0
data_clean$ses_occupation_student[data_raw$occupation_5 == 1] <- 1
table(data_clean$ses_occupation_student)

# Occupation 6 

attributes(data_raw$occupation_6)
table(data_raw$occupation_6)
data_clean$ses_occupation_caregiver <- 0
data_clean$ses_occupation_caregiver[data_raw$occupation_6 == 1] <- 1
table(data_clean$ses_occupation_caregiver)

# Occupation 7

attributes(data_raw$occupation_7)
table(data_raw$occupation_7)
data_clean$ses_occupation_parent <- 0
data_clean$ses_occupation_parent[data_raw$occupation_7 == 1] <- 1
table(data_clean$ses_occupation_parent)

# Occupation 8 

attributes(data_raw$occupation_8)
table(data_raw$occupation_8)
data_clean$ses_occupation_handicap <- 0
data_clean$ses_occupation_handicap[data_raw$occupation_8 == 1] <- 1
table(data_clean$ses_occupation_handicap)

# Occupation 9 

attributes(data_raw$occupation_9)
table(data_raw$occupation_9)
data_clean$ses_occupation_multiple_job_self_employment <- 0
data_clean$ses_occupation_multiple_job_self_employment[data_raw$occupation_9 == 1] <- 1
table(data_clean$ses_occupation_multiple_job_self_employment)

# Occupation 12 

attributes(data_raw$occupation_12)
table(data_raw$occupation_12)
data_clean$ses_occupation_multiple_job_salaried <- 0
data_clean$ses_occupation_multiple_job_salaried[data_raw$occupation_12 == 1] <- 1
table(data_clean$ses_occupation_multiple_job_salaried)

# Occupation 13

attributes(data_raw$occupation_13)
table(data_raw$occupation_13)
data_clean$ses_occupation_unemployed_social_assistance <- 0
data_clean$ses_occupation_unemployed_social_assistance[data_raw$occupation_13 == 1] <- 1
table(data_clean$ses_occupation_unemployed_social_assistance)


## Travail emploi --------------------------------------------------------------

attributes(data_raw$travail_emploi)
table(data_raw$travail_emploi)
data_clean$ses_emploi <- data_raw


## Travail domaine -------------------------------------------------------------

### travail_domaine_1

attributes(data_raw$travail_domaine_1)
table(data_raw$travail_domaine_1)
data_clean$ses_travail_management <- 0
data_clean$ses_travail_management[data_raw$travail_domaine_1 == 1] <- 1
table(data_clean$ses_travail_management)

### travail_domaine_2

attributes(data_raw$travail_domaine_2)
table(data_raw$travail_domaine_2)
data_clean$ses_travail_finance <- 0
data_clean$ses_travail_finance[data_raw$travail_domaine_2 == 1] <- 1
table(data_clean$ses_travail_finance)

### travail_domaine_3

attributes(data_raw$travail_domaine_3)
table(data_raw$travail_domaine_3)
data_clean$ses_travail_natural_sciences <- 0
data_clean$ses_travail_natural_sciences[data_raw$travail_domaine_3 == 1] <- 1
table(data_clean$ses_travail_natural_sciences)

### travail_domaine_4

attributes(data_raw$travail_domaine_4)
table(data_raw$travail_domaine_4)
data_clean$ses_travail_health <- 0
data_clean$ses_travail_health[data_raw$travail_domaine_4 == 1] <- 1
table(data_clean$ses_travail_health)

### travail_domaine_5

attributes(data_raw$travail_domaine_5)
table(data_raw$travail_domaine_5)
data_clean$ses_travail_education_law_government <- 0
data_clean$ses_travail_education_law_government[data_raw$travail_domaine_5 == 1] <- 1
table(data_clean$ses_travail_education_law_government)

### travail_domaine_6

attributes(data_raw$travail_domaine_6)
table(data_raw$travail_domaine_6)
data_clean$ses_travail_art_sport <- 0
data_clean$ses_travail_art_sport[data_raw$travail_domaine_6 == 1] <- 1
table(data_clean$ses_travail_art_sport)

### travail_domaine_7

attributes(data_raw$travail_domaine_7)
table(data_raw$travail_domaine_7)
data_clean$ses_travail_sales_services <- 0
data_clean$ses_travail_sales_services[data_raw$travail_domaine_7 == 1] <- 1
table(data_clean$ses_travail_sales_services)

### travail_domaine_8

attributes(data_raw$travail_domaine_8)
table(data_raw$travail_domaine_8)
data_clean$ses_travail_trades_transport <- 0
data_clean$ses_travail_trades_transport[data_raw$travail_domaine_8 == 1] <- 1
table(data_clean$ses_travail_trades_transport)

### travail_domaine_9

attributes(data_raw$travail_domaine_9)
table(data_raw$travail_domaine_9)
data_clean$ses_travail_transport <- 0
data_clean$ses_travail_transport[data_raw$travail_domaine_9 == 1] <- 1
table(data_clean$ses_travail_transport)

### travail_domaine_10

attributes(data_raw$travail_domaine_10)
table(data_raw$travail_domaine_10)
data_clean$ses_travail_manufacturing <- 0
data_clean$ses_travail_manufacturing[data_raw$travail_domaine_10 == 1] <- 1
table(data_clean$ses_travail_manufacturing)

### travail_domaine_11

attributes(data_raw$travail_domaine_11_TEXT)
table(data_raw$travail_domaine_11_TEXT)
data_clean$ses_travail_other <- NA
data_clean$ses_travail_other <- data_raw$travail_domaine_11_TEXT 
table(data_clean$ses_travail_other)


## Télétravail heure -----------------------------------------------------------
# Occupation 12 

attributes(data_raw$occupation_12)
table(data_raw$occupation_12)
data_clean$ses_occupation_multiple_job_salaried <- 0
data_clean$ses_occupation_multiple_job_salaried[data_raw$occupation_12 == 1] <- 1
table(data_clean$ses_occupation_multiple_job_salaried)

# Occupation 13

attributes(data_raw$occupation_13)
table(data_raw$occupation_13)
data_clean$ses_occupation_unemployed_social_assistance <- 0
data_clean$ses_occupation_unemployed_social_assistance[data_raw$occupation_13 == 1] <- 1
table(data_clean$ses_occupation_unemployed_social_assistance)


## Travail emploi --------------------------------------------------------------

attributes(data_raw$travail_emploi)
table(data_raw$travail_emploi)
data_clean$ses_emploi <- data_raw

## Travail heure ---------------------------------------------------------------

attributes(data_raw$travail_heures)
table(data_raw$travail_heures)
data_clean$ses_heures_travail <- NA
data_clean$ses_heures_travail <- data_raw$travail_heures

## Télétravail heure -----------------------------------------------------------

attributes(data_raw$teletravail_heures)
table(data_raw$teletravail_heures)
data_clean$ses_heures_teletravail <- NA
data_clean$ses_heures_teletravail <- data_raw$teletravail_heures

## Revenu ----------------------------------------------------------------------

attributes(data_raw$revenu)
table(data_raw$revenu)
data_clean$ses_revenu <- NA
data_clean$ses_revenu[data_raw$revenu == 1] <- "no_income"
data_clean$ses_revenu[data_raw$revenu == 2] <- "1_30000"
data_clean$ses_revenu[data_raw$revenu == 3] <- "30001_60000"
data_clean$ses_revenu[data_raw$revenu == 4] <- "60001_90000"
data_clean$ses_revenu[data_raw$revenu == 5] <- "90001_110000"
data_clean$ses_revenu[data_raw$revenu == 6] <- "110001_150000"
data_clean$ses_revenu[data_raw$revenu == 7] <- "150001_200000"
data_clean$ses_revenu[data_raw$revenu == 8] <- "200000_more"
table(data_clean$ses_revenu)
data_clean$ses_revenu <- factor(data_clean$ses_revenu, levels = c("no_income",
                                                                  "1_30000",
                                                                  "30001_60000",
                                                                  "60001_90000",
                                                                  "90001_110000",
                                                                  "110001_150000",
                                                                  "150001_200000",
                                                                  "200000_more"
))
table(data_clean$ses_revenu)

## Éducation -------------------------------------------------------------------

attributes(data_raw$education)
table(data_raw$education)
data_clean$ses_education <- NA
data_clean$ses_education[data_raw$education == 1] <- "no_schooling"
data_clean$ses_education[data_raw$education == 2] <- "elementary_school"
data_clean$ses_education[data_raw$education == 3] <- "high_school"
data_clean$ses_education[data_raw$education == 4] <- "cegep"
data_clean$ses_education[data_raw$education == 5] <- "bachelor"
data_clean$ses_education[data_raw$education == 6] <- "master"
data_clean$ses_education[data_raw$education == 7] <- "phd"
table(data_clean$ses_education)
data_clean$ses_education <- factor(data_clean$ses_education, levels = c("no_schooling",
                                                                        "elementary_school",
                                                                        "high_school",
                                                                        "cegep",
                                                                        "bachelor",
                                                                        "master",
                                                                        "phd"
))
table(data_clean$ses_education)


## Pays origine ----------------------------------------------------------------

attributes(data_raw$pays_origine)
table(data_raw$pays_origine)
data_clean$ses_pays_origine <- data_raw$pays_origine

## Origines ethniques ----------------------------------------------------------

attributes(data_raw$origines_ethniques)
table(data_raw$origines_ethniques)
data_clean$ses_origines_ethniques <- NA
data_clean$ses_origines_ethniques[data_raw$origines_ethniques == 1] <- "white"
data_clean$ses_origines_ethniques[data_raw$origines_ethniques == 2] <- "black"
data_clean$ses_origines_ethniques[data_raw$origines_ethniques == 3] <- "indigenous"
data_clean$ses_origines_ethniques[data_raw$origines_ethniques == 4] <- "asian"
data_clean$ses_origines_ethniques[data_raw$origines_ethniques == 5] <- "hispanic"
data_clean$ses_origines_ethniques[data_raw$origines_ethniques == 6] <- "arab"
data_clean$ses_origines_ethniques[data_raw$origines_ethniques == 7] <- "other"
data_clean$ses_origines_ethniques <- factor(data_clean$ses_origines_ethniques)
table(data_clean$ses_origines_ethniques)

## visible minority

data_clean$ses_visible_minority <- NA
data_clean$ses_visible_minority[data_raw$origines_ethniques == 1] <- 0
data_clean$ses_visible_minority[data_raw$origines_ethniques != 1] <- 1
table(data_clean$ses_visible_minority)

## Religion --------------------------------------------------------------------

attributes(data_raw$religion)
table(data_raw$religion)
data_clean$ses_religion <- NA
data_clean$ses_religion[data_raw$religion == 1] <- "atheist"
data_clean$ses_religion[data_raw$religion == 2] <- "agnostic"
data_clean$ses_religion[data_raw$religion == 3] <- "buddhist"
data_clean$ses_religion[data_raw$religion == 4] <- "hindu"
data_clean$ses_religion[data_raw$religion == 5] <- "judaism"
data_clean$ses_religion[data_raw$religion == 6] <- "muslim"
data_clean$ses_religion[data_raw$religion == 7] <- "sikhism"
data_clean$ses_religion[data_raw$religion == 8] <- "catholic"
data_clean$ses_religion[data_raw$religion == 9] <- "protestant"
data_clean$ses_religion[data_raw$religion == 10] <- "orthodox"
data_clean$ses_religion[data_raw$religion == 11] <- "other"
data_clean$ses_religion <- factor(data_clean$ses_religion)
table(data_clean$ses_religion)

## Religiosité -----------------------------------------------------------------

attributes(data_raw$religiosite_1)
table(data_raw$religiosite_1)
data_clean$ses_religiosite <- NA
data_clean$ses_religiosite <- data_raw$religiosite_1 / 100
table(data_clean$ses_religiosite)

## Enfants ---------------------------------------------------------------------

attributes(data_raw$enfants)
table(data_raw$enfants)
data_clean$ses_enfants <- NA
data_clean$ses_enfants[data_raw$enfants == 1] <- "0"
data_clean$ses_enfants[data_raw$enfants == 2] <- "1"
data_clean$ses_enfants[data_raw$enfants == 3] <- "2"
data_clean$ses_enfants[data_raw$enfants == 4] <- "3"
data_clean$ses_enfants[data_raw$enfants == 5] <- "4"
data_clean$ses_enfants[data_raw$enfants == 6] <- "5_or_more"
data_clean$ses_enfants <- factor(data_clean$ses_enfants, levels = c("0", "1", "2", "3", "4", "5_or_more"))  
table(data_clean$ses_enfants)

## Enfants bin

data_clean$ses_enfants_bin <- NA
data_clean$ses_enfants_bin[data_raw$enfants == 1] <- 0
data_clean$ses_enfants_bin[data_raw$enfants != 1] <- 1
table(data_clean$ses_enfants_bin)

## Statut marital --------------------------------------------------------------

attributes(data_raw$married)
table(data_raw$married)
data_clean$ses_marital_status <- NA
data_clean$ses_marital_status[data_raw$married == 1] <- "single"
data_clean$ses_marital_status[data_raw$married == 2] <- "married"
data_clean$ses_marital_status[data_raw$married == 3] <- "common_law"
data_clean$ses_marital_status[data_raw$married == 4] <- "widow"
data_clean$ses_marital_status[data_raw$married == 5] <- "divorced"
data_clean$ses_marital_status <- factor(data_clean$ses_marital_status)
table(data_clean$ses_marital_status)

## Status marital married

data_clean$ses_married_bin <- NA
data_clean$ses_married_bin[data_raw$married == 2] <- 1
data_clean$ses_married_bin[data_raw$married != 2] <- 0
table(data_clean$ses_married_bin)

## status marital single

data_clean$ses_single_bin <- NA
data_clean$ses_single_bin[data_raw$married == 1] <- 1
data_clean$ses_single_bin[data_raw$married != 1] <- 0
table(data_clean$ses_single_bin)

## Poids -----------------------------------------------------------------------
## This takes the value from the data_raw$poids_1_TEXT variable and converts it to a numeric value then converts it to pounds (it's in kilo)
## then it takes the value from the data_raw$poids_2_TEXT variable and converts it to a numeric value.

# Inspect the attributes and distribution of data
attributes(data_raw$poids)
table(data_raw$poids, useNA = "ifany")
table(data_raw$poids_1_TEXT, useNA = "ifany")
table(data_raw$poids_2_TEXT, useNA = "ifany")

# Convert 'poids_1_TEXT' and 'poids_2_TEXT' to numeric safely
data_raw <- data_raw %>%
  mutate(
    poids_1_TEXT = suppressWarnings(as.numeric(poids_1_TEXT)),
    poids_2_TEXT = suppressWarnings(as.numeric(poids_2_TEXT))
  )

# Initialize 'ses_poids' in 'data_clean'
data_clean$ses_poids <- NA

# Update 'ses_poids' based on conditions
data_clean <- data_clean %>%
  mutate(
    ses_poids = ifelse(data_raw$poids == 1 & !is.na(data_raw$poids_1_TEXT), data_raw$poids_1_TEXT * 2.20462, ses_poids),
    ses_poids = ifelse(data_raw$poids == 2 & !is.na(data_raw$poids_2_TEXT), data_raw$poids_2_TEXT, ses_poids)
  )

# Inspect the cleaned data
table(data_clean$ses_poids, useNA = "ifany")

## Taille ----------------------------------------------------------------------

# voir code/cleaning/taille_gpt.R

## Code postal -----------------------------------------------------------------

attributes(data_raw$code_postal)
table(data_raw$code_postal)
data_clean$ses_code_postal <- NA
data_clean$ses_code_postal <- data_raw$code_postal
table(data_clean$ses_code_postal)


## Travail deplacement ---------------------------------------------------------

attributes(data_raw$travail_deplacement)
table(data_raw$travail_deplacement)
data_clean$ses_travail_deplacement <- NA
data_clean$ses_travail_deplacement[data_raw$travail_deplacement == 1] <- "less_than_15_min"
data_clean$ses_travail_deplacement[data_raw$travail_deplacement == 2] <- "15_30_min"
data_clean$ses_travail_deplacement[data_raw$travail_deplacement == 3] <- "30_60_min"
data_clean$ses_travail_deplacement[data_raw$travail_deplacement == 4] <- "1_2_hours"
data_clean$ses_travail_deplacement[data_raw$travail_deplacement == 5] <- "2+_hours"
data_clean$ses_travail_deplacement[data_raw$travail_deplacement == 6] <- "wfh"
data_clean$ses_travail_deplacement <- factor(data_clean$ses_travail_deplacement, levels = c("less_than_15_min", "15_30_min", "30_60_min", "1_2_hours", "2+_hours", "wfh"))
table(data_clean$ses_travail_deplacement)


## Milieu vie ------------------------------------------------------------------

attributes(data_raw$milieu_vie)
table(data_raw$milieu_vie)
data_clean$ses_urban_rural <- NA
data_clean$ses_urban_rural[data_raw$milieu_vie == 1] <- "city"
data_clean$ses_urban_rural[data_raw$milieu_vie == 2] <- "suburb"
data_clean$ses_urban_rural[data_raw$milieu_vie == 3] <- "small_town"
data_clean$ses_urban_rural[data_raw$milieu_vie == 4] <- "rural"
data_clean$ses_urban_rural <- factor(data_clean$ses_urban_rural)
table(data_clean$ses_urban_rural)

## Habitation ------------------------------------------------------------------
attributes(data_raw$habitation)
table(data_raw$habitation)
data_clean$ses_habitation <- NA
data_clean$ses_habitation[data_raw$habitation == 1] <- "less_than_5_stories"
data_clean$ses_habitation[data_raw$habitation == 2] <- "loft"
data_clean$ses_habitation[data_raw$habitation == 3] <- "condo"
data_clean$ses_habitation[data_raw$habitation == 4] <- "high_rise"
data_clean$ses_habitation[data_raw$habitation == 5] <- "house"
data_clean$ses_habitation[data_raw$habitation == 6] <- "townhouse"
data_clean$ses_habitation[data_raw$habitation == 7] <- "duplex"
data_clean$ses_habitation[data_raw$habitation == 8] <- "cooperative"
data_clean$ses_habitation[data_raw$habitation == 9] <- "social_housing"
data_clean$ses_habitation[data_raw$habitation == 10] <- "mobile_home"
data_clean$ses_habitation[data_raw$habitation == 11] <- "other"
data_clean$ses_habitation <- factor(data_clean$ses_habitation)  
table(data_clean$ses_habitation)

## Province --------------------------------------------------------------------
attributes(data_raw$province_)
table(data_raw$province_)
data_clean$ses_province_ <- NA
data_clean$ses_province_[data_raw$province_ == 1] <- "Alberta"
data_clean$ses_province_[data_raw$province_ == 2] <- "British Columbia"
data_clean$ses_province_[data_raw$province_ == 3] <- "Prince Edward Island"
data_clean$ses_province_[data_raw$province_ == 4] <- "Manitoba"
data_clean$ses_province_[data_raw$province_ == 5] <- "New Brunswick"
data_clean$ses_province_[data_raw$province_ == 6] <- "Nova Scotia"
data_clean$ses_province_[data_raw$province_ == 7] <- "Nunavut"
data_clean$ses_province_[data_raw$province_ == 8] <- "Ontario"
data_clean$ses_province_[data_raw$province_ == 9] <- "Quebec"
data_clean$ses_province_[data_raw$province_ == 10] <- "Saskatchewan"
data_clean$ses_province_[data_raw$province_ == 11] <- "Newfoudland and Labrado"
data_clean$ses_province_[data_raw$province_ == 12] <- "Northwest Territories"
data_clean$ses_province_[data_raw$province_ == 13] <- "Yukon"
data_clean$ses_habitation <- factor(data_clean$ses_habitation)  
table(data_clean$ses_habitation)

## 



