git add .
git commit -m "nom du push"
git push

git reset --hard

git checkout main
git reset --hard origin/nom_de_la_branch

# créer une branche en local
git checkout -b nom_de_la_branch

# l'envoyer sur github
git push -u origin nom_de_la_branch


