<<<<<<< HEAD
data <- haven::read_sav("~/Dropbox/git/datagotchi_sante/_SharedFolder_datagotchi-santé/data/raw/data_raw.sav")

=======
data <- haven::read_sav("_SharedFolder_datagotchi-santé/data/raw/datagotchi_sante/data_raw.sav")  %>% 
  filter(code == "complete")
>>>>>>> c6d69af7f7284fa2aaebd9be2553710f1b0d0425
codebook <- sondr::sav_to_codebook(data)

write.csv(codebook, "~/Dropbox/git/datagotchi_sante/_SharedFolder_datagotchi-santé/data/raw/raw_codebook.csv")
