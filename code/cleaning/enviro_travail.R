# DemPsy_1 ---------------------------------------------------------------------

attributes(data_raw$DemPsy_1)
table(data_raw$DemPsy_1)
data_clean$enviro_work_lack_time_for_tasks <- NA
data_clean$enviro_work_lack_time_for_tasks <- (data_raw$DemPsy_1 -1) / 4
table(data_clean$enviro_work_lack_time_for_tasks)

# DemPsy_2 ---------------------------------------------------------------------

attributes(data_raw$DemPsy_2)
table(data_raw$DemPsy_2)
data_clean$enviro_work_one_task_at_a_time <- NA
data_clean$enviro_work_one_task_at_a_time <- (data_raw$DemPsy_2 -1) / 4
table(data_clean$enviro_work_one_task_at_a_time)

# DemPsy_3 ---------------------------------------------------------------------

attributes(data_raw$DemPsy_3)
table(data_raw$DemPsy_3)
data_clean$enviro_work_contradictory <- NA
data_clean$enviro_work_contradictory <- (data_raw$DemPsy_3 -1) / 4
table(data_clean$enviro_work_contradictory)

# DemPsy_4 ---------------------------------------------------------------------

attributes(data_raw$DemPsy_4)
table(data_raw$DemPsy_4)
data_clean$enviro_work_heavy_workload <- NA
data_clean$enviro_work_heavy_workload <- (data_raw$DemPsy_4 -1) / 4
table(data_clean$enviro_work_heavy_workload)

# DemPsy_5 ---------------------------------------------------------------------

attributes(data_raw$DemPsy_5)
table(data_raw$DemPsy_5)
data_clean$enviro_work_frequently_interrupted <- NA
data_clean$enviro_work_frequently_interrupted <- (data_raw$DemPsy_5 -1) / 4
table(data_clean$enviro_work_frequently_interrupted)

# DemPsy_6 ---------------------------------------------------------------------

attributes(data_raw$DemPsy_6)
table(data_raw$DemPsy_6)
data_clean$enviro_work_lot_of_responsibility <- NA
data_clean$enviro_work_lot_of_responsibility <- (data_raw$DemPsy_6 -1) / 4
table(data_clean$enviro_work_lot_of_responsibility)

# DemPsy_7 ---------------------------------------------------------------------

attributes(data_raw$DemPsy_7)
table(data_raw$DemPsy_7)
data_clean$enviro_work_forced_overtime <- NA
data_clean$enviro_work_forced_overtime <- (data_raw$DemPsy_7 -1) / 4
table(data_clean$enviro_work_forced_overtime)

# DemPsy_8 ---------------------------------------------------------------------

attributes(data_raw$DemPsy_8)
table(data_raw$DemPsy_8)
data_clean$enviro_work_objectives_are_clear <- NA
data_clean$enviro_work_objectives_are_clear <- (data_raw$DemPsy_8 -1) / 4
table(data_clean$enviro_work_objectives_are_clear)

# DemPsy_9 ---------------------------------------------------------------------

attributes(data_raw$DemPsy_9)
table(data_raw$DemPsy_9)
data_clean$enviro_work_physical_effort <- NA
data_clean$enviro_work_physical_effort <- (data_raw$DemPsy_9 -1) / 4
table(data_clean$enviro_work_physical_effort)

# DemPsy_10 ---------------------------------------------------------------------

attributes(data_raw$DemPsy_10)
table(data_raw$DemPsy_10)
data_clean$enviro_work_became_demanding <- NA
data_clean$enviro_work_became_demanding <- (data_raw$DemPsy_10 -1) / 4
table(data_clean$enviro_work_became_demanding)

# DemPsy_11 ---------------------------------------------------------------------

attributes(data_raw$DemPsy_11)
table(data_raw$DemPsy_11)
data_clean$enviro_work_require_concentration <- NA
data_clean$enviro_work_require_concentration <- (data_raw$DemPsy_11 -1) / 4
table(data_clean$enviro_work_require_concentration)

# DemPsy_12 ---------------------------------------------------------------------

attributes(data_raw$DemPsy_12)
table(data_raw$DemPsy_12)
data_clean$enviro_work_workload_too_heavy <- NA
data_clean$enviro_work_workload_too_heavy <- (data_raw$DemPsy_12 -1) / 4
table(data_clean$enviro_work_workload_too_heavy)

# LatDec -----------------------------------------------------------------------
attributes(data_raw$LatDec_1)
table(data_raw$LatDec_1)
data_clean$enviro_travail_freedom_work <- sondr::finverser(data_raw$LatDec_1 -1) / 4
table(data_clean$enviro_travail_freedom_work)

attributes(data_raw$LatDec_2)
table(data_raw$LatDec_2)
data_clean$enviro_travail_influence_amount_work <- sondr::finverser(data_raw$LatDec_2 -1) / 4
table(data_clean$enviro_travail_influence_amount_work)

attributes(data_raw$LatDec_3)
table(data_raw$LatDec_3)
data_clean$enviro_travail_decide_break <- sondr::finverser(data_raw$LatDec_3 -1) / 4
table(data_clean$enviro_travail_decide_break)

attributes(data_raw$LatDec_4)
table(data_raw$LatDec_4)
data_clean$enviro_travail_influence_decisions_work <- sondr::finverser(data_raw$LatDec_4 -1) / 4
table(data_clean$enviro_travail_influence_decisions_work)

attributes(data_raw$LatDec_5)
table(data_raw$LatDec_5)
data_clean$enviro_travail_influence_do_job <- sondr::finverser(data_raw$LatDec_5 -1) / 4
table(data_clean$enviro_travail_influence_do_job)

# SoutColl ---------------------------------------------------------------------
attributes(data_raw$SoutColl_1)
table(data_raw$SoutColl_1)
data_clean$enviro_travail_colleagues_sensitive <- sondr::finverser(data_raw$SoutColl_1 -1) / 4
table(data_clean$enviro_travail_colleagues_sensitive)

attributes(data_raw$SoutColl_2)
table(data_raw$SoutColl_2)
data_clean$enviro_travail_support_from_colleagues <- sondr::finverser(data_raw$SoutColl_2 -1) / 4
table(data_clean$enviro_travail_support_from_colleagues)

attributes(data_raw$SoutColl_3)
table(data_raw$SoutColl_3)
data_clean$enviro_travail_ask_colleagues_for_help <- sondr::finverser(data_raw$SoutColl_3 -1) / 4
table(data_clean$enviro_travail_ask_colleagues_for_help)

# SoutSup --------------------------------------------------------------------
attributes(data_raw$SoutSup_1)
table(data_raw$SoutSup_1)
data_clean$enviro_travail_task_distributed_fairly <- sondr::finverser(data_raw$SoutSup_1 -1) / 4
table(data_clean$enviro_travail_task_distributed_fairly)

attributes(data_raw$SoutSup_2)
table(data_raw$SoutSup_2)
data_clean$enviro_travail_superior_conflict_resolution <- sondr::finverser(data_raw$SoutSup_2 -1) / 4
table(data_clean$enviro_travail_superior_conflict_resolution)

attributes(data_raw$SoutSup_3)
table(data_raw$SoutSup_3)
data_clean$enviro_travail_support_from_superior <- sondr::finverser(data_raw$SoutSup_3 -1) / 4
table(data_clean$enviro_travail_support_from_superior)

attributes(data_raw$SoutSup_4)
table(data_raw$SoutSup_4)
data_clean$enviro_travail_superior_competent_planning <- sondr::finverser(data_raw$SoutSup_4 -1) / 4
table(data_clean$enviro_travail_superior_competent_planning)

attributes(data_raw$SoutSup_5)
table(data_raw$SoutSup_5)
data_clean$enviro_travail_superior_priority_job_satisfaction <- sondr::finverser(data_raw$SoutSup_5 -1) / 4
table(data_clean$enviro_travail_superior_priority_job_satisfaction)

attributes(data_raw$SoutSup_6)
table(data_raw$SoutSup_6)
data_clean$enviro_travail_times_discuss_difficulties <- sondr::finverser(data_raw$SoutSup_6 -1) / 4
table(data_clean$enviro_travail_times_discuss_difficulties)

attributes(data_raw$SoutSup_7)
table(data_raw$SoutSup_7)
data_clean$enviro_travail_superior_listens <- sondr::finverser(data_raw$SoutSup_7 -1) / 4
table(data_clean$enviro_travail_superior_listens)

attributes(data_raw$SoutSup_8)
table(data_raw$SoutSup_8)
data_clean$enviro_travail_superior_work_together <- sondr::finverser(data_raw$SoutSup_8 -1) / 4
table(data_clean$enviro_travail_superior_work_together)