# ML Healthcare Access Africa

Analyse spatiale et datamining de l'accès aux établissements de santé publics en Afrique subsaharienne, de l'exploration des données à la localisation de nouveaux sites.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mawandu/ml-healthcare-access-africa/blob/main/ml-healthcare-access-africa.ipynb)

## Données

`health_facilities.csv` : 98 745 établissements de santé **publics** géolocalisés dans 50 pays et territoires d'Afrique subsaharienne (Maina et al., 2019, *Scientific Data* 6, 134). La base contient des localisations, sans capacité, sans qualité des soins et sans population desservie.

## Contenu du notebook

| Phase | Étapes | Méthodes |
|---|---|---|
| 1. Analyse exploratoire spatiale | 1 à 5 | indice de Moran, LISA, Getis-Ord Gi*, DBSCAN |
| 2. Modélisation prédictive | 6 à 11 | variables de voisinage (arbre k-d), Random Forest, XGBoost, CNN sur patchs 11x11 |
| 3. Optimisation | 12 à 14 | localisation à couverture maximale, algorithme génétique (DEAP), glouton, tirage aléatoire |
| 4. Graphes | 15 à 17 | communautés de Louvain, Graph Transformer (GATv2) |

## Résultats principaux

- Autocorrélation spatiale positive (Moran I = 0,43), clusters LISA et DBSCAN, 154 communautés de santé (Louvain).
- CNN, pixel central masqué, test par blocs spatiaux de 5 degrés : R² d'environ 0,35, contre 0,33 pour une régression linéaire sur la moyenne des voisins.
- Couverture : 21 % de la terre ferme étudiée (5,4 millions de km2) est à plus de 50 km d'un établissement. Les 50 sites choisis par l'algorithme génétique couvrent 404 000 km2, contre 328 000 km2 pour un tirage aléatoire et 413 500 km2 pour l'algorithme glouton.

## Limites

- Le besoin est mesuré en superficie, pas en population : les grands vides arides sont favorisés.
- Les distances sont à vol d'oiseau, pas par la route.
- Un territoire non couvert peut correspondre à des données manquantes.
- La base ne contient que des établissements publics.

## Exécution

Le notebook s'exécute dans Google Colab (bouton ci-dessus). Les contours des pays (Natural Earth, 50 m) sont téléchargés automatiquement à l'étape 12.

## Fichiers

- `ml-healthcare-access-africa.ipynb` : notebook complet
- `ml_healthcare_access_africa.py` : version script du notebook
- `health_facilities.csv` : données
