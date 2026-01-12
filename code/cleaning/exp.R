# Experience 1

attributes(data_raw$exp_1)
table(data_raw$exp_1)
data_clean$exp_1 <- NA
data_clean$exp_1[data_raw$exp_1 == 1] <- 0
data_clean$exp_1[data_raw$exp_1 == 2] <- 1
data_clean$exp_1[data_raw$exp_1 == 3] <- 2
data_clean$exp_1[data_raw$exp_1 == 4] <- 3
data_clean$exp_1[data_raw$exp_1 == 5] <- 4
table(data_clean$exp_1)

# Experience 2

attributes(data_raw$exp_2)
table(data_raw$exp_2)
data_clean$exp_2 <- NA
data_clean$exp_2[data_raw$exp_2 == 1] <- 0
data_clean$exp_2[data_raw$exp_2 == 2] <- 1
data_clean$exp_2[data_raw$exp_2 == 3] <- 2
data_clean$exp_2[data_raw$exp_2 == 4] <- 3
data_clean$exp_2[data_raw$exp_2 == 5] <- 4
data_clean$exp_2[data_raw$exp_2 == 6] <- 5
table(data_clean$exp_2)
