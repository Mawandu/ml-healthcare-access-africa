# -*- coding: utf-8 -*-
# ======================================
# ETAPE 1 : CHARGEMENT ET VISUALISATION
# ======================================

# --- A. Installation et Importation des bibliothèques ---
print("Installation des bibliothèques...")
# pip/shell : pip install pandas geopandas matplotlib contextily -q
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx

print("Bibliothèques installées et importées.")

# --- B. Chargement de vos données d'établissements ---
url = 'https://raw.githubusercontent.com/Mawandu/ml-healthcare-access-africa/main/health_facilities.csv'
print(f"Chargement de vos données depuis : {url}")
try:
    df = pd.read_csv(url)
    print("Données chargées avec succès !")
except Exception as e:
    print(f"Erreur lors du chargement du fichier CSV : {e}")

# --- C. Conversion en objet géographique ---
print("\nConversion des données en format géographique...")
gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df['X'], df['Y']),
    crs="EPSG:4326"
)
print("Conversion réussie !")

# --- D. Création de la carte avec un fond de carte dynamique ---
print("\nCréation de la carte de distribution...")
fig, ax = plt.subplots(figsize=(13, 13))

gdf_web = gdf.to_crs(epsg=3857)
gdf_web.plot(ax=ax, marker='o', color='red', markersize=8, alpha=0.7, label="Établissements de Santé")
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

ax.set_title("Distribution des Établissements de Santé en Afrique Subsaharienne", fontsize=18)
ax.set_axis_off()

# --- E. Sauvegarde de la figure ---
print("\nSauvegarde de la carte en PNG...")
plt.savefig('carte_distribution_etablissements.png', dpi=300, bbox_inches='tight')
plt.show()

# =========================================================
# ETAPE 2 : ANALYSE D'AUTOCORRÉLATION SPATIALE (MORAN'S I)
# =========================================================

# --- A. Installation et Importation des bibliothèques spécialisées ---
print("Installation des bibliothèques d'analyse spatiale...")
# pip/shell : pip install esda libpysal -q
import esda
from libpysal import weights
import numpy as np

# --- B. Définition du voisinage spatial ---
print("\nCréation de la matrice de poids spatiaux (Voisinage)...")
wq = weights.KNN.from_dataframe(gdf, k=8)
wq.transform = 'R' # Standardisation des poids (la somme des poids pour chaque point = 1)
print("Matrice de poids créée.")

# --- C. Calcul de l'Indice de Moran Global ---
gdf['dist_voisin'] = gdf.geometry.apply(lambda g: gdf.geometry.distance(g).sort_values().iloc[1])

print("\nCalcul de l'Indice de Moran Global...")
moran = esda.moran.Moran(gdf['dist_voisin'], wq)
print("Calcul terminé.")

# --- D. Affichage et Interprétation des Résultats ---
print("\n" + "="*50)
print("        RÉSULTATS DE L'ANALYSE DE CLUSTERING GLOBAL")
print("="*50)
print(f"Indice de Moran Global (I) : {moran.I:.4f}")
print(f"P-valeur (p_sim)           : {moran.p_sim:.4f}")
print("="*50)

print("\n--- Interprétation ---")
# Une p-valeur très faible (ex: < 0.05) indique que le résultat n'est pas dû au hasard.
if moran.p_sim < 0.05:
    if moran.I > 0:
        print(" Résultat statistiquement significatif.")
        print("   L'indice est positif, ce qui indique une **autocorrélation spatiale positive**.")
        print("   En clair : les établissements ont tendance à se **REGRUPER**. Les zones à forte densité sont proches d'autres zones à forte densité.")
    elif moran.I < 0:
        print(" Résultat statistiquement significatif.")
        print("   L'indice est négatif, ce qui indique une **autocorrélation spatiale négative**.")
        print("   En clair : les établissements ont tendance à se **DISPERSER** de manière régulière.")
else:
    print(" Le résultat n'est pas statistiquement significatif.")
    print("   Nous ne pouvons pas conclure à un regroupement ou une dispersion.")
    print("   La distribution pourrait être **ALÉATOIRE**.")

print("\nÉtape 2 terminée ! Nous avons une mesure mathématique de la structure de nos données.")

# ==============================================
# ETAPE 3 : DÉTECTION DES CLUSTERS LOCAUX (LISA)
# ==============================================

# --- A. Calcul des Indicateurs Locaux (LISA) ---
# Nous utilisons la variable 'dist_voisin' calculée à l'étape 2.
# Le calcul va déterminer pour chaque point sa relation avec son voisinage.
print("Calcul des indicateurs de clusters locaux (LISA)...")
lisa = esda.moran.Moran_Local(gdf['dist_voisin'], wq)
print("Calcul terminé.")

# --- B. Identification des types de clusters ---
# Nous allons maintenant filtrer pour ne garder que les clusters statistiquement significatifs (p < 0.05).
# Et nous allons leur attribuer une étiquette claire :
# HH (High-High): Point chaud, forte densité entourée de forte densité.
# LL (Low-Low): Point froid, faible densité entourée de faible densité.
# LH (Low-High): Anomalie, faible densité entourée de forte densité.
# HL (High-Low): Anomalie, forte densité entourée de faible densité.

# Crée une nouvelle colonne pour les étiquettes des clusters
gdf['lisa_cluster_label'] = 'Non-significatif'
# Identifie les points significatifs (p < 0.05)
significant_points = lisa.p_sim < 0.05
# Attribue les étiquettes en fonction du quadrant (type de cluster)
gdf.loc[significant_points & (lisa.q == 1), 'lisa_cluster_label'] = 'Point Chaud (HH)'
gdf.loc[significant_points & (lisa.q == 2), 'lisa_cluster_label'] = 'Anomalie (LH)'
gdf.loc[significant_points & (lisa.q == 3), 'lisa_cluster_label'] = 'Point Froid (LL)'
gdf.loc[significant_points & (lisa.q == 4), 'lisa_cluster_label'] = 'Anomalie (HL)'

print("\nRépartition des points par type de cluster :")
print(gdf['lisa_cluster_label'].value_counts())

# --- C. Création de la carte des clusters ---
print("\nCréation de la carte des clusters...")
fig, ax = plt.subplots(figsize=(15, 15))

# Palette de couleurs pour chaque type de cluster
color_map = {
    'Non-significatif': 'lightgrey',
    'Point Chaud (HH)': '#d7191c',
    'Point Froid (LL)': '#2c7bb6',
    'Anomalie (LH)': '#abd9e9',
    'Anomalie (HL)': '#fdae61'
}

# On affiche d'abord tous les points non-significatifs en gris
gdf[gdf['lisa_cluster_label'] == 'Non-significatif'].plot(
    ax=ax, marker='o', color=color_map['Non-significatif'], markersize=5, label='Non-significatif'
)
# On affiche ensuite les points significatifs avec leurs couleurs respectives
for cluster_type, color in color_map.items():
    if cluster_type != 'Non-significatif':
        gdf[gdf['lisa_cluster_label'] == cluster_type].plot(
            ax=ax, marker='o', color=color, markersize=30, alpha=0.9, label=cluster_type
        )

# Ajout du fond de carte
ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)

# Amélioration de l'aspect
ax.set_title("Carte des Clusters d'Établissements de Santé (LISA)", fontsize=18)
ax.set_axis_off()
ax.legend(title='Type de Cluster', fontsize=12, loc='upper left', frameon=True)

# --- D. Sauvegarde de la carte ---
plt.savefig('carte_clusters_lisa.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nÉtape 3 terminée ! Nous avons identifié et visualisé les zones de concentration.")

# ==============================================================
# ETAPE 4 : ANALYSE DES POINTS CHAUDS ET FROIDS (Getis-Ord Gi*)
# ==============================================================

# --- A. Calcul de la statistique Gi* ---
# Nous réutilisons le GeoDataFrame 'gdf' et la matrice de poids 'wq' des étapes précédentes.
print("Calcul de la statistique Getis-Ord Gi*...")
g_star = esda.getisord.G_Local(gdf['dist_voisin'], wq)
print("Calcul terminé.")

# --- B. Identification des points chauds et froids ---
# La statistique Gi* nous donne un Z-score et une P-valeur pour chaque point.
# Un Z-score élevé et une P-valeur faible indiquent un point chaud (clustering de valeurs élevées).
# Un Z-score bas (négatif) et une P-valeur faible indiquent un point froid (clustering de valeurs faibles).

gdf['g_star_z_score'] = g_star.Zs
gdf['g_star_p_value'] = g_star.p_sim

# On définit les seuils de significativité (p < 0.05)
significant = gdf['g_star_p_value'] < 0.05
hot_spots = significant & (gdf['g_star_z_score'] > 0)
cold_spots = significant & (gdf['g_star_z_score'] < 0)

# On crée une nouvelle colonne pour étiqueter les résultats
gdf['g_star_cluster_label'] = 'Non-significatif'
gdf.loc[hot_spots, 'g_star_cluster_label'] = 'Point Chaud'
gdf.loc[cold_spots, 'g_star_cluster_label'] = 'Point Froid'

print("\nRépartition des points (Getis-Ord Gi*):")
print(gdf['g_star_cluster_label'].value_counts())

# --- C. Création de la carte des points chauds/froids ---
print("\nCréation de la carte des points chauds et froids...")
fig, ax = plt.subplots(figsize=(15, 15))

# Palette de couleurs
color_map_gstar = {
    'Non-significatif': 'lightgrey',
    'Point Chaud': '#d7191c',  # Rouge
    'Point Froid': '#2c7bb6'   # Bleu
}

# Affichage des points non-significatifs
gdf[gdf['g_star_cluster_label'] == 'Non-significatif'].plot(
    ax=ax, marker='o', color=color_map_gstar['Non-significatif'], markersize=5, label='Non-significatif'
)
# Affichage des points chauds et froids
for cluster_type, color in color_map_gstar.items():
    if cluster_type != 'Non-significatif':
        gdf[gdf['g_star_cluster_label'] == cluster_type].plot(
            ax=ax, marker='o', color=color, markersize=30, alpha=0.9, label=cluster_type
        )

# Ajout du fond de carte
ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)

# Amélioration de l'aspect
ax.set_title("Carte des Points Chauds et Froids (Getis-Ord Gi*)", fontsize=18)
ax.set_axis_off()
ax.legend(title='Type de Cluster', fontsize=12, loc='upper left', frameon=True)

# --- D. Sauvegarde de la carte ---
plt.savefig('carte_hotspots_getis_ord.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nÉtape 4 terminée ! Nous avons une confirmation robuste des zones de forte et faible densité.")

# =======================================================================
# ETAPE 5.1 : PRÉPARATION ET DÉCOUVERTE DU PARAMÈTRE EPSILON POUR DBSCAN
# =======================================================================
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import numpy as np

print("Préparation des données pour DBSCAN...")

# --- A. Projection des données ---
# Les calculs de distance pour DBSCAN doivent être faits en mètres, pas en degrés.
# On projette nos données dans un système de coordonnées approprié (Web Mercator).
gdf_proj = gdf.to_crs(epsg=3857)

# On extrait les coordonnées x, y pour l'algorithme
coords = np.array(list(gdf_proj.geometry.apply(lambda p: (p.x, p.y))))

# --- B. Calcul des distances au plus proche voisin ---
# C'est la méthode standard pour trouver un bon epsilon.
# On choisit min_samples = 5 (un choix courant pour les données en 2D)
min_samples = 5
print(f"Calcul des distances aux {min_samples-1} plus proches voisins...")

# On calcule la distance de chaque point à ses k-1 voisins
k_distances = NearestNeighbors(n_neighbors=min_samples).fit(coords).kneighbors(coords)[0][:, -1]

# On trie les distances par ordre croissant
k_distances_sorted = np.sort(k_distances)

# --- C. Visualisation de la courbe "du coude" ---
print("Création du graphique pour déterminer epsilon...")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(k_distances_sorted)
ax.set_title("Courbe de la distance au k-ième voisin (Elbow Method)")
ax.set_xlabel("Index des points (triés par distance)")
ax.set_ylabel(f"Distance au {min_samples-1}-ième voisin (en mètres)")
ax.grid(True)
plt.savefig('dbscan_epsilon_elbow_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# ==============================================================
# ETAPE 5.2 : EXÉCUTION DE DBSCAN ET VISUALISATION DES CLUSTERS
# ==============================================================

# --- A. Configuration et exécution de l'algorithme ---
# On utilise la valeur d'epsilon déterminée grâce au graphique.
epsilon_value = 10000  # en mètres
min_samples_value = 5

print(f"Exécution de DBSCAN avec epsilon = {epsilon_value}m et min_samples = {min_samples_value}...")

# Initialisation et entraînement du modèle
db = DBSCAN(eps=epsilon_value, min_samples=min_samples_value).fit(coords)

# On récupère les étiquettes de cluster pour chaque point
# Le label -1 est attribué par DBSCAN aux points considérés comme du "bruit" (isolés)
cluster_labels = db.labels_

# --- B. Analyse des résultats ---
# On ajoute les labels des clusters à notre GeoDataFrame
gdf['dbscan_cluster'] = cluster_labels
gdf_proj['dbscan_cluster'] = cluster_labels # On l'ajoute aussi au gdf projeté

# On compte le nombre de clusters trouvés (en excluant le bruit, label -1)
n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
n_noise = list(cluster_labels).count(-1)

print(f"\nAnalyse terminée !")
print(f"Nombre de clusters découverts : {n_clusters}")
print(f"Nombre de points isolés (bruit) : {n_noise}")


# --- C. Création de la carte des clusters DBSCAN ---
print("\nCréation de la carte des clusters...")
fig, ax = plt.subplots(figsize=(15, 15))

# On utilise le GeoDataFrame projeté pour la visualisation
# On affiche d'abord les points de bruit en petit et en noir
noise_points = gdf_proj[gdf_proj['dbscan_cluster'] == -1]
noise_points.plot(ax=ax, color='black', markersize=5, label='Bruit (points isolés)')

# On affiche ensuite chaque cluster avec une couleur différente
# On utilise une palette de couleurs pour distinguer les clusters
clusters = gdf_proj[gdf_proj['dbscan_cluster'] != -1]
clusters.plot(column='dbscan_cluster', ax=ax, categorical=True, legend=True,
              markersize=20, alpha=0.8, cmap='tab20',
              legend_kwds={'title': "Clusters", 'bbox_to_anchor': (1.15, 1)})

# Ajout du fond de carte
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

# Amélioration de l'aspect
ax.set_title(f"Clusters d'Établissements de Santé (DBSCAN)\n{n_clusters} clusters trouvés", fontsize=18)
ax.set_axis_off()

# --- D. Sauvegarde de la carte ---
plt.savefig('carte_clusters_dbscan.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nÉtape 5 terminée ! Nous avons une vision des 'archipels' de services de santé.")

# ===================================================================
# ETAPE 6 : INGÉNIERIE DES VARIABLES SPATIALES (FEATURE ENGINEERING)
# ===================================================================
from scipy.spatial import cKDTree
import numpy as np

print("Phase 2 - Étape 6 : Démarrage de l'ingénierie des variables...")

# --- A. Préparation des données ---
# On s'assure de travailler avec le GeoDataFrame projeté pour des calculs de distance précis (en mètres)
if 'gdf_proj' not in globals():
    gdf_proj = gdf.to_crs(epsg=3857)

# On extrait les coordonnées dans un format optimisé pour les calculs de distance
coords = np.array(list(gdf_proj.geometry.apply(lambda p: (p.x, p.y))))
tree = cKDTree(coords) # Structure de données optimisée pour la recherche de voisins

# --- B. Création des variables (Features) ---

# Feature 1 : Nombre de voisins dans un rayon de 10 km (10 000 mètres)
print("Calcul du nombre de voisins à 10km...")
gdf['neighbors_10km'] = tree.query_ball_point(coords, r=10000, return_length=True) - 1 # -1 pour ne pas se compter soi-même

# Feature 2 : Nombre de voisins dans un rayon de 50 km (50 000 mètres)
print("Calcul du nombre de voisins à 50km...")
gdf['neighbors_50km'] = tree.query_ball_point(coords, r=50000, return_length=True) - 1

# Feature 3 : Distance moyenne aux 5 plus proches voisins
print("Calcul de la distance moyenne aux 5 plus proches voisins...")
# On cherche les 6 plus proches (incluant le point lui-même), puis on exclut le premier (distance=0)
distances, _ = tree.query(coords, k=6)
# On calcule la moyenne des distances des 5 voisins (indices 1 à 5)
gdf['mean_dist_5nn'] = np.mean(distances[:, 1:], axis=1)

# --- C. Définition de la Variable Cible (Target) ---
# Notre objectif est de prédire l'accès aux soins, que nous modélisons par la densité.
# Une bonne mesure de la densité locale est l'inverse de la distance au plus proche voisin.
# Nous avons déjà calculé 'dist_voisin' à l'étape 2. C'est notre cible !
gdf['target_density'] = gdf['dist_voisin']


print("\nIngénierie des variables terminée !")
print("Aperçu des nouvelles colonnes créées :")
print(gdf[['neighbors_10km', 'neighbors_50km', 'mean_dist_5nn', 'target_density']].head())

print("\nÉtape 6 terminée ! Nos données sont prêtes pour le premier modèle prédictif.")

# =====================================================
# ETAPE 7 : MODÉLISATION PRÉDICTIVE AVEC RANDOM FOREST
# =====================================================
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import pandas as pd

print("Phase 2 - Étape 7 : Démarrage de la modélisation avec Random Forest...")

# --- A. Préparation des données pour le modèle ---
# Définition des variables explicatives (X) et de la variable cible (y)
features = ['neighbors_10km', 'neighbors_50km', 'mean_dist_5nn']
target = 'target_density'

X = gdf[features]
y = gdf[target]

# Division des données : 80% pour l'entraînement, 20% pour le test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Données divisées : {len(X_train)} points pour l'entraînement, {len(X_test)} pour le test.")

# --- B. Entraînement du modèle Random Forest ---
print("\nEntraînement du modèle...")
# n_estimators=100 : le modèle est une "forêt" de 100 arbres de décision
# random_state=42 : pour assurer que les résultats sont reproductibles
# n_jobs=-1 : pour utiliser tous les cœurs du processeur et accélérer le calcul
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
print("Modèle entraîné avec succès !")

# --- C. Évaluation du modèle ---
print("\nÉvaluation de la performance du modèle sur les données de test...")
y_pred = rf_model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("="*40)
print("        PERFORMANCE DU MODÈLE")
print("="*40)
print(f"Coefficient de Détermination (R²) : {r2:.3f}")
print(f"Erreur Absolue Moyenne (MAE)      : {mae:.3f}")
print("="*40)
print(f"\nInterprétation : Le modèle explique environ {r2:.1%} de la variance de la densité d'accès aux soins.")

# --- D. Visualisation des résultats ---
# 1. Graphique de l'importance des variables
print("\nCréation des graphiques d'analyse...")
feature_importances = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)

fig, ax1 = plt.subplots(1, 2, figsize=(18, 7))

feature_importances.plot(kind='barh', ax=ax1[0], color='skyblue')
ax1[0].set_title("Importance des Variables (Features)")
ax1[0].set_xlabel("Importance")

# 2. Graphique de comparaison Prédit vs Réel
ax1[1].scatter(y_test, y_pred, alpha=0.5, color='coral')
ax1[1].plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2) # Ligne d'identité (prédiction parfaite)
ax1[1].set_title("Valeurs Prédites vs Valeurs Réelles")
ax1[1].set_xlabel("Valeurs Réelles")
ax1[1].set_ylabel("Valeurs Prédites")
ax1[1].grid(True)

plt.tight_layout()

# --- E. Sauvegarde des graphiques ---
plt.savefig('analyse_modele_random_forest.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nÉtape 7 terminée ! Nous avons notre premier modèle prédictif fonctionnel.")

# ================================================
# ETAPE 8 : MODÉLISATION PRÉDICTIVE AVEC XGBOOST
# ================================================

# --- A. Installation de la bibliothèque XGBoost ---
print("Installation de la bibliothèque XGBoost...")
# pip/shell : pip install xgboost -q
import xgboost as xgb
from sklearn.metrics import r2_score, mean_absolute_error
import pandas as pd

print("Phase 2 - Étape 8 : Démarrage de la modélisation avec XGBoost...")

# --- B. Préparation des données ---
# Les données X_train, X_test, y_train, y_test ont déjà été créées à l'étape 7.
# Nous les réutilisons pour une comparaison équitable.
print(f"Utilisation des mêmes données : {len(X_train)} points pour l'entraînement, {len(X_test)} pour le test.")

# --- C. Entraînement du modèle XGBoost ---
print("\nEntraînement du modèle XGBoost...")
# objective='reg:squarederror' : spécifie que c'est un problème de régression.
# n_estimators=100 : le nombre d'arbres à construire.
# random_state=42 : pour la reproductibilité.
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42, n_jobs=-1)
xgb_model.fit(X_train, y_train)
print("Modèle XGBoost entraîné avec succès !")

# --- D. Évaluation du modèle ---
print("\nÉvaluation de la performance du modèle sur les données de test...")
y_pred_xgb = xgb_model.predict(X_test)

r2_xgb = r2_score(y_test, y_pred_xgb)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)

print("="*40)
print("        PERFORMANCE DU MODÈLE XGBOOST")
print("="*40)
print(f"Coefficient de Détermination (R²) : {r2_xgb:.3f}")
print(f"Erreur Absolue Moyenne (MAE)      : {mae_xgb:.3f}")
print("="*40)
print(f"\nInterprétation : Le modèle XGBoost explique environ {r2_xgb:.1%} de la variance.")

# --- E. Visualisation des résultats ---
print("\nCréation des graphiques d'analyse pour XGBoost...")
feature_importances_xgb = pd.Series(xgb_model.feature_importances_, index=features).sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Graphique de l'importance des variables
feature_importances_xgb.plot(kind='barh', ax=axes[0], color='lightgreen')
axes[0].set_title("Importance des Variables (XGBoost)")
axes[0].set_xlabel("Importance")

# Graphique de comparaison Prédit vs Réel
axes[1].scatter(y_test, y_pred_xgb, alpha=0.5, color='dodgerblue')
axes[1].plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
axes[1].set_title("Valeurs Prédites vs Valeurs Réelles (XGBoost)")
axes[1].set_xlabel("Valeurs Réelles")
axes[1].set_ylabel("Valeurs Prédites")
axes[1].grid(True)

plt.tight_layout()

# --- F. Sauvegarde des graphiques ---
plt.savefig('analyse_modele_xgboost.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nÉtape 8 terminée ! Nous avons un deuxième modèle prédictif pour comparaison.")

# ==============================================================================
# ETAPE 9 : AMÉLIORATION DU MODÈLE PAR INTÉGRATION DES COORDONNÉES
# ==============================================================================
print("Phase 2 - Étape 9 : Démarrage du modèle Random Forest avec features spatiales...")

# --- A. Préparation des données étendues ---
# On ajoute les coordonnées 'X' et 'Y' à notre liste de features
features_spatial = ['neighbors_10km', 'neighbors_50km', 'mean_dist_5nn', 'X', 'Y']
target = 'target_density'

X_spatial = gdf[features_spatial]
y_spatial = gdf[target]

# On utilise la même division entraînement/test pour une comparaison juste
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_spatial, y_spatial, test_size=0.2, random_state=42)
print(f"Utilisation de {len(features_spatial)} features pour la modélisation.")
print(f"Données divisées : {len(X_train_s)} points pour l'entraînement, {len(X_test_s)} pour le test.")

# --- B. Entraînement du modèle "Spatial" Random Forest ---
print("\nEntraînement du nouveau modèle...")
rf_model_spatial = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model_spatial.fit(X_train_s, y_train_s)
print("Modèle spatial entraîné avec succès !")

# --- C. Évaluation du nouveau modèle ---
print("\nÉvaluation de la performance du modèle spatial...")
y_pred_s = rf_model_spatial.predict(X_test_s)

r2_s = r2_score(y_test_s, y_pred_s)
mae_s = mean_absolute_error(y_test_s, y_pred_s)

print("="*50)
print("        PERFORMANCE DU MODÈLE AVEC COORDONNÉES")
print("="*50)
print(f"Nouveau Coefficient de Détermination (R²) : {r2_s:.3f}")
print(f"Ancien R² (sans coordonnées)              : 0.869")
print(f"Amélioration                              : {((r2_s - 0.869) / 0.869):.2%}")
print("="*50)

# --- D. Visualisation de l'importance des variables ---
print("\nCréation du graphique d'importance des variables...")
feature_importances_s = pd.Series(rf_model_spatial.feature_importances_, index=features_spatial).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
feature_importances_s.plot(kind='barh', color='mediumseagreen')
plt.title("Importance des Variables (Coordonnées Incluses)")
plt.xlabel("Importance")
plt.tight_layout()

# --- E. Sauvegarde du graphique ---
plt.savefig('analyse_modele_spatial_rf.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nÉtape 9 terminée !")

# ==============================================================
# ETAPE 10 : PRÉPARATION DES DONNÉES POUR LE DEEP LEARNING (CNN)
# ==============================================================
import numpy as np
import contextily as ctx # On s'assure que contextily est importé

print("Phase 2 - Étape 10 : Transformation des données en grille de densité pour CNN...")

# --- A. Définition de la grille ---
# Cette partie n'a pas changé et fonctionne parfaitement.
min_x, min_y, max_x, max_y = gdf.total_bounds
margin = 1
x_bins = np.arange(min_x - margin, max_x + margin, 0.1)
y_bins = np.arange(min_y - margin, max_y + margin, 0.1)
print(f"Création d'une grille de {len(x_bins)} x {len(y_bins)} pixels.")

# --- B. Création de l'histogramme 2D ---
density_grid, _, _ = np.histogram2d(gdf['Y'], gdf['X'], bins=[y_bins, x_bins])
density_grid_log = np.log1p(density_grid)
print("Grille de densité créée avec succès !")

# --- C. Visualisation de notre "image" de densité ---
print("\nVisualisation de la grille de densité...")
fig, ax = plt.subplots(figsize=(10, 10))

# On affiche notre grille comme une image. L'argument 'extent' est crucial
# car il cale correctement l'image sur les coordonnées géographiques.
im = ax.imshow(density_grid_log, cmap='viridis', origin='lower',
               extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]],
               alpha=0.7) # On ajoute un peu de transparence pour bien voir le fond de carte

# CORRECTION ICI : On utilise contextily pour ajouter le fond de carte
# On lui précise que les coordonnées de notre graphique sont en EPSG:4326 (Lat/Lon)
ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)

# Amélioration de l'aspect
ax.set_title("Image de la Densité des Établissements de Santé")
# Les labels des axes sont maintenant redondants avec le fond de carte
ax.set_axis_off()
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Densité (log)")

# --- D. Sauvegarde du graphique ---
plt.savefig('cnn_density_grid.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nÉtape 10 terminée ! Nos données sont prêtes à être utilisées par un CNN.")

import os, random, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, geopandas as gpd
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from scipy.spatial import cKDTree
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GroupShuffleSplit
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
os.makedirs("figures", exist_ok=True)

# Même grille que l'étape 10
grid_log = density_grid_log
P = 11; H = P // 2
padded = np.pad(grid_log, H, mode="constant", constant_values=0)
windows = sliding_window_view(padded, (P, P))
ii, jj = np.nonzero(grid_log > 0)
X_leak = windows[ii, jj].astype("float32")
y_patch = grid_log[ii, jj].astype("float32")
X_mask = X_leak.copy(); X_mask[:, H, H] = 0
groups = (ii // 50) * 1000 + (jj // 50)
print(len(y_patch), "patchs (la valeur centrale vaut y dans X_leak :", bool(np.allclose(X_leak[:, H, H], y_patch)), ")")

def cnn():
    m = Sequential([Input((P, P, 1)), Conv2D(32, 3, activation="relu"), MaxPooling2D(2),
                    Conv2D(64, 3, activation="relu"), Flatten(), Dense(100, activation="relu"), Dense(1)])
    m.compile(optimizer="adam", loss="mse"); return m

def run(X, tr, te, seed=42):
    tf.keras.utils.set_random_seed(seed)
    m = cnn(); m.fit(X[tr][..., None], y_patch[tr], epochs=10, batch_size=64, validation_split=0.1, verbose=0)
    return m, r2_score(y_patch[te], m.predict(X[te][..., None], verbose=0).ravel())

idx = np.arange(len(y_patch))
tr_r, te_r = train_test_split(idx, test_size=0.2, random_state=42)
tr_b, te_b = next(GroupShuffleSplit(1, test_size=0.2, random_state=42).split(idx, groups=groups))
m_leak, r2_A = run(X_leak, tr_r, te_r)
_, r2_B = run(X_mask, tr_r, te_r)
m_mask, r2_C = run(X_mask, tr_b, te_b)

def nb_feats(X):
    f = X.reshape(len(X), -1).copy(); f = np.delete(f, (P * P) // 2, axis=1)
    ring = X[:, H-1:H+2, H-1:H+2].reshape(len(X), -1).copy(); ring = np.delete(ring, 4, axis=1)
    return np.c_[f.mean(1), ring.mean(1)]
F_nb = nb_feats(X_mask)
r2_base = r2_score(y_patch[te_b], LinearRegression().fit(F_nb[tr_b], y_patch[tr_b]).predict(F_nb[te_b]))

cnn_res = pd.DataFrame({"Configuration": ["A. Contrôle de fuite : pixel central inclus, découpage aléatoire",
                                      "B. CNN, pixel central masqué, découpage aléatoire",
                                      "C. CNN, pixel central masqué, blocs spatiaux de 5° (résultat retenu)",
                                      "Référence : régression sur la moyenne des voisins (blocs spatiaux)"],
                    "R²": [r2_A, r2_B, r2_C, r2_base]}).round(3)
cnn_res.to_csv("figures/cnn_r2.csv", index=False); cnn_res

import os, urllib.request, shapely
POLY = "ne_50m_admin_0_countries.geojson"      # contours des pays (Natural Earth, 50 m)
if not os.path.exists(POLY):
    urllib.request.urlretrieve("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_0_countries.geojson", POLY)
R_KM = 50.0; K = 50
polys = gpd.read_file(POLY)[["ADMIN", "geometry"]]
j = gpd.sjoin(gdf[["geometry"]], polys, how="left", predicate="within")
admins = sorted(j["ADMIN"].dropna().unique())
study_polys = polys[polys["ADMIN"].isin(admins)]
study = study_polys.union_all().buffer(0.1)
print(len(admins), "polygones de pays avec au moins un établissement")

def xyz(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    return 6371.0 * np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]

xc = x_bins[:-1] + 0.05; yc = y_bins[:-1] + 0.05
PX, PY = np.meshgrid(xc, yc)
land = shapely.contains_xy(study, PX.ravel(), PY.ravel())
lon = PX.ravel()[land]; lat = PY.ravel()[land]
gi, gj = np.divmod(np.nonzero(land)[0], len(xc))
area = (0.1 * 111.32 * np.cos(np.radians(lat))) * (0.1 * 110.57)
d_exist, _ = cKDTree(xyz(gdf["X"].values, gdf["Y"].values)).query(xyz(lon, lat))
unc = d_exist > R_KM
print(f"Terre étudiée : {area.sum()/1e6:.2f} M km² | non couverte à {R_KM:.0f} km : {area[unc].sum()/1e6:.2f} M km² ({100*area[unc].sum()/area.sum():.1f} %)")

D_lon, D_lat, D_w = lon[unc], lat[unc], area[unc]
D_i, D_j = gi[unc], gj[unc]
cand = np.nonzero((D_i % 3 == 0) & (D_j % 3 == 0))[0]
treeD = cKDTree(xyz(D_lon, D_lat))
cover = treeD.query_ball_point(xyz(D_lon[cand], D_lat[cand]), r=R_KM)
cover = [np.array(c, dtype=np.int64) for c in cover]
NC = len(cand)
flat = np.concatenate(cover); starts = np.r_[0, np.cumsum([len(c) for c in cover])[:-1]]
print(NC, "sites candidats ; pixels non couverts :", len(D_w))

# pip/shell : pip install deap -q
def covered_area(sel):
    cov = np.zeros(len(D_w), bool)
    for c in sel: cov[cover[c]] = True
    return D_w[cov].sum()

# 1) tirage aléatoire
rng = np.random.default_rng(1)
rand = np.array([covered_area(rng.choice(NC, K, replace=False)) for _ in range(1000)])

# 2) glouton (gain marginal exact)
cov = np.zeros(len(D_w), bool); greedy = []
for _ in range(K):
    gain = np.add.reduceat(D_w[flat] * (~cov[flat]), starts)
    gain[greedy] = -1; b = int(np.argmax(gain)); greedy.append(b); cov[cover[b]] = True
greedy_val = covered_area(greedy)

# 3) algorithme génétique (DEAP)
from deap import base, creator, tools, algorithms
random.seed(7); np.random.seed(7)
creator.create("FitnessMax", base.Fitness, weights=(1.0,)); creator.create("Individual", list, fitness=creator.FitnessMax)
tb = base.Toolbox()
tb.register("ind", lambda: creator.Individual(random.sample(range(NC), K)))
tb.register("population", tools.initRepeat, list, tb.ind)
tb.register("evaluate", lambda ind: (covered_area(ind),))
def cx_set(a, b):
    pool = list(set(a) | set(b)); random.shuffle(pool); a[:] = pool[:K]; b[:] = pool[-K:]; return a, b
def mut_replace(ind, indpb=0.08):
    s = set(ind)
    for k in range(K):
        if random.random() < indpb:
            new = random.randrange(NC)
            while new in s: new = random.randrange(NC)
            s.discard(ind[k]); ind[k] = new; s.add(new)
    return ind,
tb.register("mate", cx_set); tb.register("mutate", mut_replace); tb.register("select", tools.selTournament, tournsize=3)
ga_pop = tb.population(n=200); hof = tools.HallOfFame(1)
ga_stats = tools.Statistics(lambda ind: ind.fitness.values[0]); ga_stats.register("max", np.max); ga_stats.register("moy", np.mean)
ga_pop, ga_log = algorithms.eaSimple(ga_pop, tb, cxpb=0.7, mutpb=0.4, ngen=150, stats=ga_stats, halloffame=hof, verbose=False)
ga = list(hof[0]); ga_val = covered_area(ga)

tot_unc = D_w.sum()
cover_res = pd.DataFrame({"Méthode": ["Tirage aléatoire (moyenne de 1 000)", "Glouton", "Algorithme génétique (150 générations)"],
                    "Superficie nouvellement couverte (milliers de km²)": [rand.mean()/1e3, greedy_val/1e3, ga_val/1e3],
                    "Part du territoire non couvert (%)": [100*rand.mean()/tot_unc, 100*greedy_val/tot_unc, 100*ga_val/tot_unc]}).round(1)
cover_res.to_csv("figures/couverture_resultats.csv", index=False)
print("Génétique : générations", int(ga_log.select("gen")[0]), "à", int(ga_log.select("gen")[-1]), "| max initial", round(ga_log.select("max")[0]/1e3,1), "-> final", round(ga_log.select("max")[-1]/1e3,1), "milliers de km²")
cover_res

best = ga if ga_val >= greedy_val else greedy
S_lon, S_lat = D_lon[cand][best], D_lat[cand][best]
sites = gpd.GeoDataFrame({"lon": S_lon, "lat": S_lat}, geometry=gpd.points_from_xy(S_lon, S_lat), crs="EPSG:4326")
sites = gpd.sjoin(sites, polys, how="left", predicate="within")
print("Sites hors de la zone terrestre étudiée :", int(sites["ADMIN"].isna().sum()), "/", K)
print(sites["ADMIN"].fillna("hors polygone (côte)").value_counts().to_string())
sites[["lon", "lat", "ADMIN"]].to_csv("figures/sites_optimaux_50.csv", index=False)

fig, ax = plt.subplots(figsize=(11, 11))
study_polys.boundary.plot(ax=ax, color="#9b9b9b", linewidth=0.5)
ax.scatter(D_lon, D_lat, s=1.2, color="#f2b8a8", label=f"Terre à plus de {R_KM:.0f} km d'un établissement", zorder=1)
ax.scatter(gdf["X"], gdf["Y"], s=0.4, color="#555555", alpha=0.5, label="Établissements existants", zorder=2)
ax.scatter(S_lon, S_lat, marker="*", s=170, color="gold", edgecolor="black", linewidth=0.7, label=f"{K} nouveaux sites", zorder=3)
ax.set_xlim(-26, 64); ax.set_ylim(-36, 26); ax.set_aspect("equal"); ax.set_axis_off()
ax.set_title(f"Localisation à couverture maximale : {K} sites, rayon {R_KM:.0f} km (terre ferme uniquement)", fontsize=13)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], marker="o", ls="", color="#f2b8a8", markersize=8, label=f"Terre à plus de {R_KM:.0f} km d'un établissement"),
                   Line2D([], [], marker="o", ls="", color="#555555", markersize=4, label="Établissements existants"),
                   Line2D([], [], marker="*", ls="", markerfacecolor="gold", markeredgecolor="black", markersize=15, label=f"{K} nouveaux sites")],
          loc="lower left", frameon=True, fontsize=10)
plt.savefig("figures/carte_solution_optimale.png", dpi=170, bbox_inches="tight"); plt.show()

# Complétude des données : densité d'établissements par pays (pour lire la carte avec prudence)
cnt = j["ADMIN"].value_counts()
surf = study_polys.set_index("ADMIN").to_crs(6933).geometry.area / 1e9      # milliers de km²
dens = (cnt / surf.reindex(cnt.index)).dropna().rename("établissements par 1 000 km²")
tab = dens.loc[[a for a in sites["ADMIN"].dropna().unique()]].round(1).to_frame()
print("Médiane des pays de la base :", round(dens.median(), 1), "établissements par 1 000 km²")
tab

# ==========================================================
# ETAPE 15 : THÉORIE DES GRAPHES - DÉTECTION DE COMMUNAUTÉS
# ==========================================================

# --- A. Installation des bibliothèques nécessaires ---
print("Installation des bibliothèques pour la théorie des graphes...")
# pip/shell : pip install networkx community -q
import networkx as nx
from community import community_louvain

print("Phase 4 - Étape 15 : Démarrage de l'analyse de réseau...")

# --- B. Construction du graphe ---
if 'wq' not in globals():
    from libysal import weights
    print("Création de la matrice de poids spatiaux (Voisinage)...")
    wq = weights.KNN.from_dataframe(gdf, k=8)

# Création du graphe
G_directed = wq.to_networkx()
G = G_directed.to_undirected()

print(f"Graphe non-dirigé créé avec {G.number_of_nodes()} nœuds et {G.number_of_edges()} liens.")

# --- C. Détection de communautés avec l'algorithme de Louvain ---
print("\nDétection des communautés de santé avec l'algorithme de Louvain...")
partition = community_louvain.best_partition(G)
n_communities = len(set(partition.values()))
print(f"{n_communities} communautés détectées.")

gdf['community_id'] = gdf.index.map(partition)

# --- D. Visualisation de la carte des communautés ---
print("\nCréation de la carte des communautés...")
fig, ax = plt.subplots(figsize=(15, 15))

gdf.plot(column='community_id', ax=ax, categorical=True, cmap='tab20',
         legend=False, markersize=10, alpha=0.9)

ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)

ax.set_title(f"Détection de {n_communities} Communautés de Santé (Louvain)", fontsize=18)
ax.set_axis_off()

# --- E. Sauvegarde de la carte ---
plt.savefig('carte_communautes_sante.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nÉtape 15 terminée ! Nous avons une vision macroscopique de l'organisation du réseau de santé.")

# =============================================================
# ETAPE 16 : PRÉPARATION DES DONNÉES POUR UN GRAPH TRANSFORMER
# =============================================================

print("Phase 4 - Étape 16 : Préparation des données pour l'analyse par Transformer...")

# --- A. Sélection des caractéristiques des nœuds ---
# Ce sont les informations que le Transformer utilisera pour décrire chaque établissement.
node_features = [
    'X',
    'Y',
    'neighbors_10km',
    'neighbors_50km',
    'mean_dist_5nn',
    'community_id'
]

# On s'assure que toutes les données sont numériques et sans valeurs manquantes
transformer_data = gdf[node_features].copy()
transformer_data = transformer_data.fillna(0) # Remplacer les NaN potentiels par 0

print(f"Données préparées pour {len(transformer_data)} nœuds avec {len(node_features)} caractéristiques chacun.")
print("\nAperçu des caractéristiques des nœuds :")
print(transformer_data.head())

# --- B. Standardisation des caractéristiques ---
# Les réseaux de neurones fonctionnent mieux quand les données sont mises à l'échelle.
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_transformer = scaler.fit_transform(transformer_data)

print("\nCaractéristiques des nœuds standardisées.")
print("Les données sont maintenant prêtes à être injectées dans un modèle de type Graph Transformer.")

# Le graphe 'G' et les caractéristiques 'X_transformer' sont les deux entrées requises.

print("\nÉtape 16 terminée ! Les données sont structurées pour l'étape la plus avancée de l'IA.")

# ==============================================================================
# ETAPE 17 : MODÉLISATION PAR GRAPH TRANSFORMER
# ==============================================================================

# --- A. Installation et Importation ---
print("Installation des bibliothèques pour PyTorch Geometric...")
# pip/shell : pip install torch torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.1.0+cu121.html -q

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import from_networkx
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

print("Phase 4 - Étape 17 : Démarrage de la modélisation par Graph Transformer...")

# --- B. Conversion des Données au format PyG ---
# On utilise le graphe 'G', les features 'X_transformer' et la cible 'target_density'
# 'G' doit être notre graphe non-dirigé de l'étape 15
# 'X_transformer' et 'gdf['target_density']' des étapes précédentes

# Conversion du graphe et des caractéristiques en Tenseurs PyTorch
pyg_data = from_networkx(G)
pyg_data.x = torch.tensor(X_transformer, dtype=torch.float)
pyg_data.y = torch.tensor(gdf['target_density'].values, dtype=torch.float)

# Création des masques pour l'entraînement et le test (80/20)
num_nodes = G.number_of_nodes()
indices = list(range(num_nodes))
train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)

train_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[train_indices] = True
test_mask[test_indices] = True

pyg_data.train_mask = train_mask
pyg_data.test_mask = test_mask

print(f"Données converties au format PyTorch Geometric.")

# --- C. Définition de l'Architecture du Modèle ---
class GraphTransformer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, 16, heads=8, dropout=0.6)
        self.conv2 = GATv2Conv(16 * 8, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_gt = GraphTransformer(pyg_data.num_node_features, 1).to(device)
data = pyg_data.to(device)
optimizer = torch.optim.Adam(model_gt.parameters(), lr=0.005, weight_decay=5e-4)

# --- D. Boucle d'Entraînement ---
print("\nEntraînement du Graph Transformer (cela peut prendre plusieurs minutes)...")
def train():
    model_gt.train()
    optimizer.zero_grad()
    out = model_gt(data.x, data.edge_index)
    loss = F.mse_loss(out[data.train_mask].squeeze(), data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test():
    model_gt.eval()
    out = model_gt(data.x, data.edge_index)
    # On calcule le R² sur l'ensemble de test
    y_true = data.y[data.test_mask].cpu().numpy()
    y_pred = out[data.test_mask].squeeze().cpu().detach().numpy()
    return r2_score(y_true, y_pred)

for epoch in range(1, 101):
    loss = train()
    if epoch % 10 == 0:
        test_r2 = test()
        print(f'Époque: {epoch:03d}, Perte: {loss:.4f}, R² Test: {test_r2:.3f}')

# --- E. Évaluation Finale ---
final_r2 = test()
print("\nEntraînement terminé !")
print("="*50)
print("        PERFORMANCE DU MODÈLE GRAPH TRANSFORMER")
print("="*50)
print(f"Coefficient de Détermination (R²) : {final_r2:.3f}")
print("Pour comparaison : CNN (pixel central masqué, blocs spatiaux) : R² = 0.35 (cibles et découpages différents)")
print("="*50)

print("\nToutes les phases de modélisation sont terminées !")

# ==============================================================================
# ETAPE FINALE : VISUALISATION DES ERREURS DU GRAPH TRANSFORMER
# ==============================================================================

print("Création de la carte des erreurs de prédiction du Graph Transformer...")

# --- A. Obtenir les prédictions finales ---
model_gt.eval() # Mettre le modèle en mode évaluation
with torch.no_grad(): # Pas besoin de calculer les gradients
    predictions = model_gt(data.x, data.edge_index).cpu().numpy().flatten()

# On ajoute les prédictions et les erreurs à notre GeoDataFrame principal
gdf['gt_prediction'] = predictions
gdf['gt_error'] = gdf['target_density'] - gdf['gt_prediction']


# --- B. Création de la carte des erreurs ---
fig, ax = plt.subplots(figsize=(13, 13))

# On utilise une palette de couleurs "divergente" :
# - Bleu : le modèle a sous-estimé la densité (erreur positive)
# - Rouge : le modèle a sur-estimé la densité (erreur négative)
# - Blanc : le modèle était proche de la réalité
gdf.plot(ax=ax, column='gt_error', cmap='coolwarm',
         markersize=10, legend=True,
         legend_kwds={'label': "Erreur de Prédiction (Réel - Prédit)", 'orientation': "horizontal"})

# Ajout du fond de carte
ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)

# Amélioration
ax.set_title("Carte des Erreurs de Prédiction du Graph Transformer", fontsize=18)
ax.set_axis_off()

# --- C. Sauvegarde de la carte ---
plt.savefig('carte_erreurs_graph_transformer.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nAnalyse technique terminée. Tous les modèles ont été construits et évalués.")

import os
import zipfile
# from google.colab import files  (Colab uniquement)

print("Préparation du téléchargement de tous les graphiques...")

# Nom de l'archive ZIP qui contiendra tous vos fichiers
zip_filename = 'rapport_final_graphiques.zip'

# Liste de tous les fichiers PNG que nous avons générés
files_to_zip = [
    'carte_distribution_etablissements.png',
    'carte_clusters_lisa.png',
    'carte_hotspots_getis_ord.png',
    'dbscan_epsilon_elbow_plot.png',
    'carte_clusters_dbscan.png',
    'analyse_modele_random_forest.png',
    'analyse_modele_xgboost.png',
    'analyse_modele_spatial_rf.png',
    'cnn_density_grid.png',
    'figures/carte_solution_optimale.png',
    'figures/cnn_r2.csv',
    'figures/couverture_resultats.csv',
    'figures/sites_optimaux_50.csv',
    'carte_communautes_sante.png',
    'carte_erreurs_graph_transformer.png'
]

# Création du fichier ZIP
with zipfile.ZipFile(zip_filename, 'w') as zipf:
    print(f"Création de l'archive '{zip_filename}'...")
    for file in files_to_zip:
        if os.path.exists(file):
            zipf.write(file)
            print(f"  -> Ajout de '{file}'")
        else:
            print(f"  -> Attention : '{file}' non trouvé, il sera ignoré.")

print("\nArchive créée avec succès !")
print("Lancement du téléchargement...")

# Commande magique de Colab pour télécharger le fichier
# files.download(zip_filename)  (Colab uniquement)
