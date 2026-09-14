# ==============================================================================
# PROJET SMART ENERGY GRID - PHASE 4 : DÉFINITION DES ENTITÉS ET FEATURES FEAST
# Fichier : feature_repository/definitions.py
# Description :
#   Ce fichier Python déclare les objets Feast qui constituent le registre :
#   1. Entity (Entité) : L'identifiant métier unique (Turbine_ID).
#   2. BatchSource (Source de données) : Le fichier Parquet historique avec horodatage.
#   3. FeatureView (Vue de caractéristiques) : Les 13 colonnes de télémétrie
#      qui alimenteront le modèle d'Intelligence Artificielle en temps réel.
# ==============================================================================

# Importation de timedelta pour définir la durée de vie (TTL) des observations
from datetime import timedelta

# Importation des classes fondamentales du framework Feast
from feast import Entity, FeatureView, Field, FileSource, ValueType

# Importation des types de données fortement typés de Feast
from feast.types import Float64


# ==============================================================================
# 1. DÉFINITION DE L'ENTITÉ (L'ÉOLIENNE)
# ==============================================================================
# Une entité est la clé primaire sur laquelle les features sont indexées.
# Ici, chaque éolienne est identifiée par son numéro unique (Turbine_ID : 1, 2...).
turbine_entity = Entity(
    # Nom logique de l'entité au sein du Feature Store
    name="turbine_id",

    # Type de la valeur : Entier 64 bits (INT64)
    value_type=ValueType.INT64,

    # Nom exact de la colonne dans le DataFrame / table servant de clé de jointure
    join_keys=["Turbine_ID"],

    # Description humaine claire de l'entité
    description="Identifiant numérique unique de chaque turbine éolienne du parc",
)


# ==============================================================================
# 2. DÉFINITION DE LA SOURCE DE DONNÉES OFFLINE (FICHIER PARQUET)
# ==============================================================================
# Cette source indique à Feast où trouver les enregistrements historiques et comment
# interpréter la chronologie des événements grâce à la colonne 'timestamp'.
turbine_source = FileSource(
    # Nom descriptif de la source de données
    name="turbine_telemetry_source",

    # Chemin relatif (depuis le dossier feature_repository) vers le fichier Parquet enrichi
    path="data/turbine_features.parquet",

    # Colonne indiquant la date et l'heure réelles de la mesure du capteur (obligatoire pour Feast)
    timestamp_field="timestamp",

    # Colonne indiquant la date et l'heure d'enregistrement dans le système
    created_timestamp_column="created_timestamp",

    # Description de la source de données
    description="Fichier Parquet contenant les mesures de capteurs et les 5 indicateurs calculés",
)


# ==============================================================================
# 3. DÉFINITION DE LA VUE DE FEATURES (FEATURE VIEW)
# ==============================================================================
# La Feature View regroupe les 13 variables physiques et calculées associées à chaque turbine.
# C'est ce groupe de variables qui sera matérialisé dans Redis et interrogé par l'API.
turbine_features_view = FeatureView(
    # Nom unique de la vue de features
    name="turbine_features",

    # Entité(s) à laquelle ces features sont rattachées (ici l'éolienne)
    entities=[turbine_entity],

    # Durée de validité temporelle (Time-To-Live) d'une mesure :
    # Si une mesure date de plus de 730 jours (2 ans), elle ne sera pas retenue
    ttl=timedelta(days=730),

    # Schéma explicite des 13 features avec leur type de données exact (Float64)
    schema=[
        # --- Les 8 mesures physiques brutes transmises par les capteurs IoT ---
        Field(name="Rotor_Speed_RPM", dtype=Float64, description="Vitesse de rotation des pales (tours/min)"),
        Field(name="Wind_Speed_mps", dtype=Float64, description="Vitesse du vent captée par l'anémomètre (m/s)"),
        Field(name="Power_Output_kW", dtype=Float64, description="Puissance électrique instantanée générée (kW)"),
        Field(name="Gearbox_Oil_Temp_C", dtype=Float64, description="Température de l'huile du multiplicateur (°C)"),
        Field(name="Generator_Bearing_Temp_C", dtype=Float64, description="Température des roulements de la génératrice (°C)"),
        Field(name="Vibration_Level_mmps", dtype=Float64, description="Niveau de vibration mécanique de la nacelle (mm/s)"),
        Field(name="Ambient_Temp_C", dtype=Float64, description="Température extérieure de l'air ambiant (°C)"),
        Field(name="Humidity_pct", dtype=Float64, description="Taux d'humidité relative de l'air ambiant (%)"),

        # --- Les 5 indicateurs calculés par notre ingénierie de variables (Feature Engineering) ---
        Field(name="Thermal_Imbalance", dtype=Float64, description="Écart de température huile multiplicateur vs roulements génératrice (°C)"),
        Field(name="Power_Efficiency", dtype=Float64, description="Efficacité énergétique : puissance produite / vitesse de rotation"),
        Field(name="Thermal_Stress", dtype=Float64, description="Stress thermique : température d'huile vs température ambiante (°C)"),
        Field(name="Vibration_per_RPM", dtype=Float64, description="Intensité vibratoire ramenée à chaque tour mécanique effectué"),
        Field(name="Heat_Humidity_Index", dtype=Float64, description="Indice hygrométrique combiné chaleur et humidité"),
    ],

    # Activation du magasin en ligne : cette vue sera synchronisée dans Redis pour l'API
    online=True,

    # Source de données batch associée
    source=turbine_source,

    # Description humaine de la Feature View
    description="Ensemble complet des 13 variables de surveillance prédictive des turbines",
)
