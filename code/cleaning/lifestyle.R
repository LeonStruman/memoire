# Bloc lifestyle ---------------------------------------------------------------

## Pets ------------------------------------------------------------------------

attributes(data_raw$pets)
table(data_raw$pets)
data_clean$lifestyle_pets <- NA
data_clean$lifestyle_pets[data_raw$pets == 1] <- "Cat(s)"
data_clean$lifestyle_pets[data_raw$pets == 2] <- "Dog(s)"
data_clean$lifestyle_pets[data_raw$pets == 3] <- "Cat(s) and dog(s)"
data_clean$lifestyle_pets[data_raw$pets == 4] <- "Other Pet(s)"
data_clean$lifestyle_pets[data_raw$pets == 5] <- "Farm animals"
data_clean$lifestyle_pets[data_raw$pets == 6] <- "I don't have pets"


data_clean$lifestyle_pets <- factor(data_clean$lifestyle_pets)  
table(data_clean$lifestyle_pets)


## Activité physique -----------------------------------------------------------

attributes(data_raw$act_physique)
table(data_raw$act_physique)
data_clean$lifestyle_act_physique <- NA
data_clean$lifestyle_act_physique[data_raw$act_physique == 1] <- "Run"
data_clean$lifestyle_act_physique[data_raw$act_physique == 2] <- "Gym"
data_clean$lifestyle_act_physique[data_raw$act_physique == 3] <- "Walk"
data_clean$lifestyle_act_physique[data_raw$act_physique == 4] <- "Swim"
data_clean$lifestyle_act_physique[data_raw$act_physique == 5] <- "Team sport"
data_clean$lifestyle_act_physique[data_raw$act_physique == 6] <- "Yoga"
data_clean$lifestyle_act_physique[data_raw$act_physique == 7] <- "Other"
data_clean$lifestyle_act_physique[data_raw$act_physique == 8] <- "I don't do physical activities"


data_clean$lifestyle_act_physique <- factor(data_clean$lifestyle_act_physique)  
table(data_clean$lifestyle_act_physique)


## Freq_physique ---------------------------------------------------------------

attributes(data_raw$freq_physique_)
table(data_raw$freq_physique_)
data_clean$lifestyle_freq_physique_ <- NA
data_clean$lifestyle_freq_physique_[data_raw$freq_physique_ == 1] <- "Never"
data_clean$lifestyle_freq_physique_[data_raw$freq_physique_ == 2] <- "A few times a year"
data_clean$lifestyle_freq_physique_[data_raw$freq_physique_ == 3] <- "Once a month"
data_clean$lifestyle_freq_physique_[data_raw$freq_physique_ == 4] <- "Once a week"
data_clean$lifestyle_freq_physique_[data_raw$freq_physique_ == 5] <- "A few times a week"
data_clean$lifestyle_freq_physique_[data_raw$freq_physique_ == 6] <- "Once a day"
data_clean$lifestyle_freq_physique_[data_raw$freq_physique_ == 7] <- "More than once a day"

data_clean$lifestyle_freq_physique_ <- factor(data_clean$lifestyle_freq_physique_, levels = c("Never", "A few times a year", "Once a month", "Once a week", "A few times a week", "Once a day", "More than once a day"))  
table(data_clean$lifestyle_freq_physique_)



## Transport -------------------------------------------------------------------

attributes(data_raw$transport)
table(data_raw$transport)
data_clean$lifestyle_transport <- NA
data_clean$lifestyle_transport[data_raw$transport == 1] <- "Public transit"
data_clean$lifestyle_transport[data_raw$transport == 2] <- "Walking"
data_clean$lifestyle_transport[data_raw$transport == 3] <- "Motorbike"
data_clean$lifestyle_transport[data_raw$transport == 4] <- "Motorcycle"
data_clean$lifestyle_transport[data_raw$transport == 5] <- "Car"

data_clean$lifestyle_transport <- factor(data_clean$lifestyle_transport)  
table(data_clean$lifestyle_transport)



## Car model -------------------------------------------------------------------

attributes(data_raw$car_model)
table(data_raw$car_model)
data_clean$lifestyle_car_model <- NA
data_clean$lifestyle_car_model[data_raw$car_model == 1] <- "4x4"
data_clean$lifestyle_car_model[data_raw$car_model == 2] <- "Regular sedan or station wagon"
data_clean$lifestyle_car_model[data_raw$car_model == 3] <- "Convertible or roadster"
data_clean$lifestyle_car_model[data_raw$car_model == 4] <- "Pickup"
data_clean$lifestyle_car_model[data_raw$car_model == 5] <- "Van or minivan"
data_clean$lifestyle_car_model[data_raw$car_model == 6] <- "Luxury car (Mercedes, Porsche, etc.)"
data_clean$lifestyle_car_model[data_raw$car_model == 7] <- "Sports car"
data_clean$lifestyle_car_model[data_raw$car_model == 8] <- "Hybrid or electric car"
data_clean$lifestyle_car_model[data_raw$car_model == 9] <- "SUV"
data_clean$lifestyle_car_model[data_raw$car_model == 10] <- "Other"
data_clean$lifestyle_car_model[data_raw$car_model == 11] <- "I do not have a car or I never use a car"

data_clean$lifestyle_car_model <- factor(data_clean$lifestyle_car_model)  
table(data_clean$lifestyle_car_model)


## Consumption -----------------------------------------------------------------

attributes(data_raw$consumption)
table(data_raw$consumption)
data_clean$lifestyle_consumption <- NA
data_clean$lifestyle_consumption[data_raw$consumption == 1] <- "Independent stores"
data_clean$lifestyle_consumption[data_raw$consumption == 2] <- "Chain stores (Gap, Zara, etc.)"
data_clean$lifestyle_consumption[data_raw$consumption == 3] <- "Thrift stores"
data_clean$lifestyle_consumption[data_raw$consumption == 4] <- "Superstores (Walmart, Costco, etc.)"
data_clean$lifestyle_consumption[data_raw$consumption == 5] <- "Department stores (The Bay, Simons, etc.)"
data_clean$lifestyle_consumption[data_raw$consumption == 6] <- "Online only stores"
data_clean$lifestyle_consumption[data_raw$consumption == 7] <- "Other"


data_clean$lifestyle_consumption <- factor(data_clean$lifestyle_consumption)  
table(data_clean$lifestyle_consumption)


## Coffee ----------------------------------------------------------------------

attributes(data_raw$coffee)
table(data_raw$coffee)
data_clean$lifestyle_coffee <- NA
data_clean$lifestyle_coffee[data_raw$coffee == 1] <- "Independent coffee shops"
data_clean$lifestyle_coffee[data_raw$coffee == 2] <- "McDonald's"
data_clean$lifestyle_coffee[data_raw$coffee == 3] <- "Second Cup"
data_clean$lifestyle_coffee[data_raw$coffee == 4] <- "Starbucks"
data_clean$lifestyle_coffee[data_raw$coffee == 5] <- "Tim Hortons"
data_clean$lifestyle_coffee[data_raw$coffee == 6] <- "Other coffee shop chains"
data_clean$lifestyle_coffee[data_raw$coffee == 7] <- "I don't go to coffee shops"

data_clean$lifestyle_coffee <- factor(data_clean$lifestyle_coffee)  
table(data_clean$lifestyle_coffee)

## Meat ------------------------------------------------------------------------

attributes(data_raw$meat)
table(data_raw$meat)
data_clean$lifestyle_meat <- NA
data_clean$lifestyle_meat <- (data_raw$meat - 1) / 6
table(data_clean$lifestyle_meat)

## Alcool pref -----------------------------------------------------------------

attributes(data_raw$alcool_pref)
table(data_raw$alcool_pref)
data_clean$lifestyle_alcool_pref <- NA
data_clean$lifestyle_alcool_pref[data_raw$alcool_pref == 1] <- "Craft or microbrewery beer"
data_clean$lifestyle_alcool_pref[data_raw$alcool_pref == 2] <- "Regular beer"
data_clean$lifestyle_alcool_pref[data_raw$alcool_pref == 3] <- "Spirit drink"
data_clean$lifestyle_alcool_pref[data_raw$alcool_pref == 4] <- "Cocktail"
data_clean$lifestyle_alcool_pref[data_raw$alcool_pref == 5] <- "White wine"
data_clean$lifestyle_alcool_pref[data_raw$alcool_pref == 6] <- "Sparkling wine or champagne"
data_clean$lifestyle_alcool_pref[data_raw$alcool_pref == 7] <- "Rosé wine"
data_clean$lifestyle_alcool_pref[data_raw$alcool_pref == 8] <- "Red wine"
data_clean$lifestyle_alcool_pref[data_raw$alcool_pref == 9] <- "I do not drink alcohol"
data_clean$lifestyle_alcool_pref <- factor(data_clean$lifestyle_alcool_pref)
table(data_clean$lifestyle_alcool_pref)

## Smoking ---------------------------------------------------------------------

attributes(data_raw$smoking)
table(data_raw$smoking)
data_clean$lifestyle_smoking_freq <- NA
data_clean$lifestyle_smoking_freq <- (data_raw$smoking - 1) / 6
table(data_clean$lifestyle_smoking_freq)

## Cannabis --------------------------------------------------------------------

attributes(data_raw$cannabis)
table(data_raw$cannabis)
data_clean$lifestyle_cannabis <- NA
data_clean$lifestyle_cannabis <- (data_raw$cannabis - 1) / 6
table(data_clean$lifestyle_cannabis)


## Hunting ---------------------------------------------------------------------

attributes(data_raw$act_hunting)
table(data_raw$act_hunting)
data_clean$lifestyle_hunting <- NA
data_clean$lifestyle_hunting <- (data_raw$act_hunting - 1) / 4
table(data_clean$lifestyle_hunting)

## Fishing ---------------------------------------------------------------------

attributes(data_raw$act_fishing)
table(data_raw$act_fishing)
data_clean$lifestyle_fishing <- NA
data_clean$lifestyle_fishing <- (data_raw$act_fishing - 1) / 4
table(data_clean$lifestyle_fishing)


## Act_motorized ---------------------------------------------------------------

attributes(data_raw$act_motorized)
table(data_raw$act_motorized)
data_clean$lifestyle_motorized <- NA
data_clean$lifestyle_motorized <- (data_raw$act_motorized - 1) / 4
table(data_clean$lifestyle_motorized)

## Friends ---------------------------------------------------------------------

attributes(data_raw$act_friends)
table(data_raw$act_friends)
data_clean$lifestyle_friends_activity <- NA
data_clean$lifestyle_friends_activity <- (data_raw$act_friends - 1) / 4
table(data_clean$lifestyle_friends_activity)



## Volunteer -------------------------------------------------------------------

attributes(data_raw$act_volunteer)
table(data_raw$act_volunteer)
data_clean$lifestyle_volunteer <- NA
data_clean$lifestyle_volunteer <- (data_raw$act_volunteer - 1) / 4
table(data_clean$lifestyle_volunteer)



## Museum ----------------------------------------------------------------------

attributes(data_raw$act_museum)
table(data_raw$act_museum)
data_clean$lifestyle_museum <- NA
data_clean$lifestyle_museum <- (data_raw$act_museum - 1) / 4
table(data_clean$lifestyle_museum)


## Nature ----------------------------------------------------------------------

attributes(data_raw$act_nature_1)
table(data_raw$act_nature_1)
data_clean$lifestyle_nature_may_september <- NA
data_clean$lifestyle_nature_may_september <- (data_raw$act_nature_1 - 1) / 4
table(data_clean$lifestyle_nature_may_september)

attributes(data_raw$act_nature_2)
table(data_raw$act_nature_2)
data_clean$lifestyle_nature_october_april <- NA
data_clean$lifestyle_nature_october_april <- (data_raw$act_nature_2 - 1) / 4
table(data_clean$lifestyle_nature_october_april)


## Movie pref ------------------------------------------------------------------

table(data_raw$movie_pref)
data_clean$lifestyle_movie_pref <- (data_raw$movie_pref)
table(data_clean$lifestyle_movie_pref)


## Style -----------------------------------------------------------------------

attributes(data_raw$style)
table(data_raw$style)
data_clean$lifestyle_style <- NA
data_clean$lifestyle_style[data_raw$style == 1] <- "Hippie"
data_clean$lifestyle_style[data_raw$style == 2] <- "Elegant"
data_clean$lifestyle_style[data_raw$style == 3] <- "Classical"
data_clean$lifestyle_style[data_raw$style == 4] <- "Casual"
data_clean$lifestyle_style[data_raw$style == 5] <- "Formal"
data_clean$lifestyle_style[data_raw$style == 6] <- "Punk"
data_clean$lifestyle_style[data_raw$style == 7] <- "Rock"
data_clean$lifestyle_style[data_raw$style == 8] <- "Sporty"
data_clean$lifestyle_style[data_raw$style == 9] <- "Other"
data_clean$lifestyle_style <- factor(data_clean$lifestyle_style)
table(data_clean$lifestyle_style)

## Style autre-----------------------------------------------------------------------

attributes(data_raw$style_9_TEXT)
table(data_raw$style_9_TEXT)
data_clean$lifestyle_style_other <- NA
data_clean$lifestyle_style_other <- (data_raw$style_9_TEXT)
table(data_clean$lifestyle_style_other)


## Music pref ------------------------------------------------------------------

attributes(data_raw$music_pref)
table(data_raw$music_pref)
data_clean$lifestyle_music_pref <- NA
data_clean$lifestyle_music_pref <- (data_raw$music_pref)
table(data_clean$lifestyle_music_pref)

## Temps écran -----------------------------------------------------------------

# temps__cran__1 -----------------------------------------

attributes(data_raw$temps__cran__1)
table(data_raw$temps__cran__1)
data_clean$lifestyle_temps_ecran_semaine_travail <- NA


for (i in 1:nrow(data_raw)) {
  screen_time_str <- data_raw$temps__cran__1[i]  # Store the current value in a variable for easier handling
  
  # Clean up the string by removing all non-numeric and non-delimiter characters except delimiters
  clean_str <- gsub("[^0-9/_ -]", "", screen_time_str)
  
  if (grepl("[/_ -]", clean_str)) {
    # If there's any of the specified delimiters, split the string
    num_parts <- strsplit(clean_str, "[/_ -]")[[1]]
    num_values <- as.numeric(num_parts)  # Convert to numeric
    if (any(is.na(num_values))) {
      # Handle any NA values that may result from conversion (e.g., empty strings)
      num_values <- num_values[!is.na(num_values)]
    }
    if (length(num_values) > 0) {
      data_clean$lifestyle_temps_ecran_semaine_travail[i] <- max(num_values, na.rm = TRUE)  # Assign the maximum value if non-empty
    } else {
      # Assign a default value or handle empty num_values vector appropriately
      data_clean$lifestyle_temps_ecran_semaine_travail[i] <- NA  # Or some other default or error handling
    }
  } else if (grepl("min", screen_time_str)) {
    # If "min" is found, convert from minutes to hours
    data_clean$lifestyle_temps_ecran_semaine_travail[i] <- as.numeric(gsub("[^0-9]", "", screen_time_str)) / 60
  } else {
    # Assume the input is in hours and just remove non-numeric characters
    data_clean$lifestyle_temps_ecran_semaine_travail[i] <- as.numeric(gsub("[^0-9]", "", screen_time_str))
  }
}

table(data_clean$lifestyle_temps_ecran_semaine_travail) 


# temps__cran__2 ---------------------------------------

attributes(data_raw$temps__cran__2)
table(data_raw$temps__cran__2)
data_clean$lifestyle_temps_ecran_finsemaine_travail <- NA


for (i in 1:nrow(data_raw)) {
  screen_time_str <- data_raw$temps__cran__2[i]  # Store the current value in a variable for easier handling
  
  # Clean up the string by removing all non-numeric and non-delimiter characters except delimiters
  clean_str <- gsub("[^0-9/_ -]", "", screen_time_str)
  
  if (grepl("[/_ -]", clean_str)) {
    # If there's any of the specified delimiters, split the string
    num_parts <- strsplit(clean_str, "[/_ -]")[[1]]
    num_values <- as.numeric(num_parts)  # Convert to numeric
    if (any(is.na(num_values))) {
      # Handle any NA values that may result from conversion (e.g., empty strings)
      num_values <- num_values[!is.na(num_values)]
    }
    if (length(num_values) > 0) {
      data_clean$lifestyle_temps_ecran_finsemaine_travail[i] <- max(num_values, na.rm = TRUE)  # Assign the maximum value if non-empty
    } else {
      # Assign a default value or handle empty num_values vector appropriately
      data_clean$lifestyle_temps_ecran_finsemaine_travail[i] <- NA  # Or some other default or error handling
    }
  } else if (grepl("min", screen_time_str)) {
    # If "min" is found, convert from minutes to hours
    data_clean$lifestyle_temps_ecran_finsemaine_travail[i] <- as.numeric(gsub("[^0-9]", "", screen_time_str)) / 60
  } else {
    # Assume the input is in hours and just remove non-numeric characters
    data_clean$lifestyle_temps_ecran_finsemaine_travail[i] <- as.numeric(gsub("[^0-9]", "", screen_time_str))
  }
}

table(data_clean$lifestyle_temps_ecran_finsemaine_travail)

# temps__cran__3 ----------------------------------------
# Display attributes and table for the column
attributes(data_raw$temps__cran__3)
table(data_raw$temps__cran__3)

# Initialize the column with NA values
data_clean$lifestyle_temps_ecran_semaine_loisir <- NA

# Loop through each entry to process the screen time strings
for (i in 1:nrow(data_raw)) {
  screen_time_str <- as.character(data_raw$temps__cran__3[i])  # Ensure the value is treated as a string
  
  if (is.na(screen_time_str)) {
    next  # Skip if the screen time string is NA
  }
  
  # Clean up the string by removing all non-numeric and non-delimiter characters except delimiters
  clean_str <- gsub("[^0-9/_ -]", "", screen_time_str)
  
  if (grepl("[/_ -]", clean_str)) {
    # If there's any of the specified delimiters, split the string
    num_parts <- strsplit(clean_str, "[/_ -]")[[1]]
    num_values <- as.numeric(num_parts)  # Convert to numeric
    
    if (any(is.na(num_values))) {
      # Handle any NA values that may result from conversion (e.g., empty strings)
      num_values <- num_values[!is.na(num_values)]
    }
    
    if (length(num_values) > 0) {
      data_clean$lifestyle_temps_ecran_semaine_loisir[i] <- max(num_values, na.rm = TRUE)  # Assign the maximum value
    } else {
      data_clean$lifestyle_temps_ecran_semaine_loisir[i] <- NA
    }
  } else if (grepl("min", screen_time_str)) {
    # If "min" is found, convert from minutes to hours
    minutes <- as.numeric(gsub("[^0-9]", "", screen_time_str))
    if (!is.na(minutes)) {
      data_clean$lifestyle_temps_ecran_semaine_loisir[i] <- minutes / 60
    } else {
      data_clean$lifestyle_temps_ecran_semaine_loisir[i] <- NA
    }
  } else {
    # Assume the input is in hours and just remove non-numeric characters
    hours <- as.numeric(gsub("[^0-9]", "", screen_time_str))
    data_clean$lifestyle_temps_ecran_semaine_loisir[i] <- ifelse(!is.na(hours), hours, NA)
  }
}

# Display the table of cleaned data
table(data_clean$lifestyle_temps_ecran_semaine_loisir, useNA = "ifany")


# temps__cran__4 ---------------------------------------

attributes(data_raw$temps__cran__4)
table(data_raw$temps__cran__4)
data_clean$lifestyle_temps_ecran_finsemaine_loisir <- NA

for (i in 1:nrow(data_raw)) {
  screen_time_str <- data_raw$temps__cran__4[i]  # Store the current value in a variable for easier handling
  
  # Clean up the string by removing all non-numeric and non-delimiter characters except delimiters
  clean_str <- gsub("[^0-9/_ -]", "", screen_time_str)
  
  if (grepl("[/_ -]", clean_str)) {
    # If there's any of the specified delimiters, split the string
    num_parts <- strsplit(clean_str, "[/_ -]")[[1]]
    num_values <- as.numeric(num_parts)  # Convert to numeric
    if (any(is.na(num_values))) {
      # Handle any NA values that may result from conversion (e.g., empty strings)
      num_values <- num_values[!is.na(num_values)]
    }
    if (length(num_values) > 0) {
      data_clean$lifestyle_temps_ecran_finsemaine_loisir[i] <- max(num_values, na.rm = TRUE)  # Assign the maximum value if non-empty
    } else {
      # Assign a default value or handle empty num_values vector appropriately
      data_clean$lifestyle_temps_ecran_finsemaine_loisir[i] <- NA  # Or some other default or error handling
    }
  } else if (grepl("min", screen_time_str)) {
    # If "min" is found, convert from minutes to hours
    data_clean$lifestyle_temps_ecran_finsemaine_loisir[i] <- as.numeric(gsub("[^0-9]", "", screen_time_str)) / 60
  } else {
    # Assume the input is in hours and just remove non-numeric characters
    data_clean$lifestyle_temps_ecran_finsemaine_loisir[i] <- as.numeric(gsub("[^0-9]", "", screen_time_str))
  }
}

table(data_clean$lifestyle_temps_ecran_finsemaine_loisir)
