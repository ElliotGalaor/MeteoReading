# Présenter le projet en entretien

## Le pitch (30 secondes)

« J'ai construit un pipeline qui récupère automatiquement les 25 millions de relevés quotidiens publiés par
Météo-France pour 20 grandes villes depuis 1950. Je les nettoie et les stocke dans un entrepôt DuckDB, j'en tire
des indicateurs testés statistiquement, et je les restitue dans un dashboard. Résultat principal : +0,33 °C par
décennie depuis 1951, un rythme qui accélère, des journées à 35 °C cinq fois plus fréquentes, mais aucune tendance
nette sur la pluie. Le plus intéressant techniquement, c'est la qualité des données : les stations déménagent,
il a fallu raccorder des séries. »

## La démo (5 minutes)

1. **Vue d'ensemble** : les 4 chiffres clés, puis les bandes de réchauffement. Survoler 2022.
2. **Explorer une ville** : Paris contre Marseille, puis passer en « Écart à la normale » : les deux courbes se
   superposent, le réchauffement est le même malgré des climats différents. Montrer la p-value sous le graphique.
3. **Une année en détail** : Paris 2022 contre la normale ; 2003 pour la canicule d'août.
4. **Données et qualité** : le schéma du pipeline, le tableau des raccordements (Reims, Grenoble, Limoges),
   la validation ERA5.
5. **Requêtes SQL** : lancer « Rang de l'année 2022 » (fonction de fenêtre), puis la requête sur les 25 millions de
   lignes : réponse en ~0,1 s.

## Questions probables

**Pourquoi DuckDB plutôt que pandas ou PostgreSQL ?**
25 millions de lignes ne tiennent pas confortablement dans pandas. DuckDB est une base analytique en colonnes,
sans serveur : elle lit les CSV compressés directement et agrège 25 M lignes en une fraction de seconde. PostgreSQL
aurait demandé d'installer et d'administrer un serveur pour un usage purement analytique.

**Comment avez-vous choisi les stations ?**
Automatiquement, en SQL : la station la plus complète sur 1951-2025, à moins de 40 km et 200 m d'altitude du
centre-ville. Le garde-fou d'altitude a été ajouté après avoir vu que Grenoble tombait sur une station de
montagne à 806 m. Trois villes ont une chaîne de stations imposée dans la configuration.

**Qu'est-ce que le raccordement de séries ?**
Quand une station ferme (Reims-Courcy en 2014), on prend le relais avec une autre (Reims-Prunay, ouverte en 2012).
Sur les deux ans où les deux mesurent, on calcule leur écart moyen, mois par mois, et on corrige l'ancienne série
de cet écart. Sans ça, on créerait un faux saut de température au moment du changement.

**Comment savez-vous qu'une tendance est réelle ?**
Deux tests : la régression linéaire (pente et intervalle de confiance) et Mann-Kendall, non paramétrique, qui ne
suppose pas de loi normale. Je ne déclare une tendance significative que si les deux le sont. Pour la pluie, ce
n'est pas le cas, et je le dis : un résultat non significatif est aussi un résultat.

**Pourquoi des anomalies plutôt que des températures brutes ?**
Pour comparer des villes aux climats différents et pour qu'une ville manquante une année ne fasse pas bouger la
moyenne nationale. Chaque ville est comparée à sa propre normale 1991-2020.

**Quelles sont les limites ?**
Les séries ne sont pas homogénéisées comme celles des climatologues : l'urbanisation autour d'une station peut
amplifier la tendance. Une station représente une ville, pas son territoire. Et les tests supposent des années
indépendantes, donc les p-values sont un peu optimistes.

**Pourquoi la comparaison avec ERA5 ?**
Pour valider mes séries avec une source indépendante, et parce que beaucoup d'entreprises utilisent des
réanalyses faute de stations. Conclusion : ERA5 suit très bien la météo au jour le jour (corrélation 0,987),
mais lisse les effets locaux (mer, ville) ; pour les extrêmes locaux, il faut des stations.

**Qu'est-ce que vous feriez avec plus de temps ?**
Étendre à toute la France (c'est un paramètre de configuration), ajouter l'ensoleillement et l'humidité,
planifier une mise à jour quotidienne, et déployer le dashboard en ligne.

**Comment garantissez-vous que le code fonctionne ?**
18 tests automatisés, dont un test de bout en bout sur un mini-jeu de données synthétique qui reproduit tous
les pièges rencontrés : code qualité douteux, valeur impossible, TN > TX, doublon, identifiant à zéro initial,
station qui déménage.
