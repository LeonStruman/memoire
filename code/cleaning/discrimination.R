# Bloc Discrimination -----------------------------------------------------


## discrimination_1 --------------------------------------------------------

attributes(data_raw$discrimination_1)
table(data_raw$discrimination_1)
data_clean$discrimination_gender <- data_raw$discrimination_1
data_clean$discrimination_gender[is.na(data_raw$discrimination_1)] <- 0
table(data_clean$discrimination_gender)


## discrimination_2 --------------------------------------------------------

attributes(data_raw$discrimination_2)
table(data_raw$discrimination_2)
data_clean$discrimination_gender_expression <- data_raw$discrimination_2
data_clean$discrimination_gender_expression[is.na(data_raw$discrimination_2)] <- 0
table(data_clean$discrimination_gender_expression)


## discrimination_3 --------------------------------------------------------

attributes(data_raw$discrimination_3)
table(data_raw$discrimination_3)
data_clean$discrimination_ethnicity_culture <- data_raw$discrimination_3
data_clean$discrimination_ethnicity_culture[is.na(data_raw$discrimination_3)] <- 0
table(data_clean$discrimination_ethnicity_culture)


## discrimination_4 --------------------------------------------------------

attributes(data_raw$discrimination_4)
table(data_raw$discrimination_4)
data_clean$discrimination_race <- data_raw$discrimination_4
data_clean$discrimination_race[is.na(data_raw$discrimination_4)] <- 0
table(data_clean$discrimination_race)


## discrimination_5 --------------------------------------------------------

attributes(data_raw$discrimination_5)
table(data_raw$discrimination_5)
data_clean$discrimination_physical_appearance <- data_raw$discrimination_5
data_clean$discrimination_physical_appearance[is.na(data_raw$discrimination_5)] <- 0
table(data_clean$discrimination_physical_appearance)


## discrimination_6 --------------------------------------------------------

attributes(data_raw$discrimination_6)
table(data_raw$discrimination_6)
data_clean$discrimination_religion <- data_raw$discrimination_6
data_clean$discrimination_religion[is.na(data_raw$discrimination_6)] <- 0
table(data_clean$discrimination_religion)


# discrimination_7 --------------------------------------------------------

attributes(data_raw$discrimination_7)
table(data_raw$discrimination_7)
data_clean$discrimination_sexual_orientation <- data_raw$discrimination_7
data_clean$discrimination_sexual_orientation[is.na(data_raw$discrimination_7)] <- 0
table(data_clean$discrimination_sexual_orientation)


## discrimination_8 --------------------------------------------------------

attributes(data_raw$discrimination_8)
table(data_raw$discrimination_8)
data_clean$discrimination_age <- data_raw$discrimination_8
data_clean$discrimination_age[is.na(data_raw$discrimination_8)] <- 0
table(data_clean$discrimination_age)


## discrimination_9 --------------------------------------------------------

attributes(data_raw$discrimination_9)
table(data_raw$discrimination_9)
data_clean$discrimination_disability <- data_raw$discrimination_9
data_clean$discrimination_disability[is.na(data_raw$discrimination_9)] <- 0
table(data_clean$discrimination_disability)


## discrimination_10 -------------------------------------------------------

attributes(data_raw$discrimination_10)
table(data_raw$discrimination_10)
data_clean$discrimination_language <- data_raw$discrimination_10
data_clean$discrimination_language[is.na(data_raw$discrimination_10)] <- 0
table(data_clean$discrimination_language)


## discrimination_11 --------------------------------------------------------

attributes(data_raw$discrimination_11)
table(data_raw$discrimination_11)
data_clean$discrimination_other <- data_raw$discrimination_11
data_clean$discrimination_other[is.na(data_raw$discrimination_11)] <- 0
table(data_clean$discrimination_other)


## discrimination_12 -------------------------------------------------------

attributes(data_raw$discrimination_12)
table(data_raw$discrimination_12)
data_clean$discrimination_never <- data_raw$discrimination_12
data_clean$discrimination_never[is.na(data_raw$discrimination_12)] <- 0
table(data_clean$discrimination_never)
