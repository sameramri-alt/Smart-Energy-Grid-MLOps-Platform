# ==============================================================================
# PROJET SMART ENERGY GRID - PHASE 4 : PRÉPARATION DES DONNÉES DU FEATURE STORE
# Fichier : scripts/prepare_feast_data.py
# Description : 
#   1. Charge les données de télémétrie brutes (capteurs d'éoliennes).
#   2. Applique le Feature Engineering identique à la Phase 3 (les 5 colonnes calculées).
#   3. Ajoute les horodatages temporels (event_timestamp et created_timestamp)
#      requis par Feast pour ordonner et versionner les observations.
#   4. Exporte le résultat final au format Apache Parquet optimisé
#      dans 'feature_repository/data/turbine_features.parquet'.
# ==============================================================================

# Importation du module de gestion du système et des chemins de fichiers
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

# Importation de pandas pour le chargement, la manipulation et le calcul vectorisé
import pandas as pd

# Importation de pyarrow pour la sauvegarde haute performance au format Parquet
import pyarrow as pa
import pyarrow.parquet as pq


def compute_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule exactement les 5 features synthétiques créées en Phase 3 (Machine Learning).
    Cette fonction garantit le principe de consistance MLOps :
    la même formule mathématique est utilisée pour l'entraînement et pour Feast.
    """
    # Copie locale du DataFrame pour éviter toute modification indésirable
    data = df.copy()

    # Feature 1 : Déséquilibre thermique (Thermal_Imbalance)
    # Mesure l'écart de température entre l'huile du multiplicateur et le roulement de la génératrice
    data["Thermal_Imbalance"] = (
        data["Gearbox_Oil_Temp_C"] - data["Generator_Bearing_Temp_C"]
    )

    # Feature 2 : Efficacité énergétique instantanée (Power_Efficiency)
    # Ratio de la puissance électrique générée par rapport à la vitesse de rotation des pales
    # L'ajout de +1e-5 au dénominateur empêche toute division par zéro en cas d'arrêt des pales
    data["Power_Efficiency"] = (
        data["Power_Output_kW"] / (data["Rotor_Speed_RPM"] + 1e-5)
    )

    # Feature 3 : Stress thermique sur le multiplicateur (Thermal_Stress)
    # Écart entre la température de l'huile de boîte et la température ambiante de l'air
    data["Thermal_Stress"] = (
        data["Gearbox_Oil_Temp_C"] - data["Ambient_Temp_C"]
    )

    # Feature 4 : Intensité des vibrations par tour mécanique (Vibration_per_RPM)
    # Ratio vibration mécanique rapportée à la vitesse de rotation
    data["Vibration_per_RPM"] = (
        data["Vibration_Level_mmps"] / (data["Rotor_Speed_RPM"] + 1e-5)
    )

    # Feature 5 : Indice combiné de chaleur et d'humidité (Heat_Humidity_Index)
    # Indicateur des conditions atmosphériques favorisant l'oxydation ou la condensation
    data["Heat_Humidity_Index"] = (
        data["Ambient_Temp_C"] * (data["Humidity_pct"] / 100.0)
    )

    # Renvoi du DataFrame enrichi des 5 nouvelles colonnes
    return data


def generate_time_stamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute les colonnes d'horodatage temporelles obligatoires pour Feast :
    - timestamp : l'instant précis où la mesure physique a eu lieu (event_timestamp).
    - created_timestamp : l'instant où l'enregistrement a été inséré dans le système.
    """
    # Copie locale du DataFrame
    data = df.copy()

    # Définition de la date de référence de départ (1er janvier 2024, en temps universel UTC)
    start_date = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    # Création d'une liste vide pour stocker les dates générées
    timestamps = []

    # Regroupement par turbine pour attribuer une chronologie cohérente à chaque machine
    for turbine_id, group in data.groupby("Turbine_ID", sort=False):
        # Nombre total d'enregistrements pour cette turbine
        n_records = len(group)

        # Génération d'une séquence de pas de temps de 1 heure pour chaque enregistrement
        turbine_times = [start_date + timedelta(hours=i) for i in range(n_records)]
        
        # Ajout des dates à la liste principale
        timestamps.extend(turbine_times)

    # Assignation de la série d'horodatages à la colonne 'timestamp'
    data["timestamp"] = pd.to_datetime(timestamps, utc=True)

    # Assignation de 'created_timestamp' qui correspond à la date de traitement
    now_utc = datetime.now(timezone.utc)
    data["created_timestamp"] = pd.to_datetime(now_utc, utc=True)

    # Renvoi du DataFrame doté de sa dimension temporelle
    return data


def main():
    # Affichage du titre du script
    print("\n" + "=" * 80)
    print(" PHASE 4 : PRÉPARATION DU JEU DE DONNÉES ENRICHI POUR FEAST")
    print("=" * 80)

    # Définition du chemin vers le fichier source de télémétrie
    source_csv_path = Path("data/Test.csv")

    # Vérification de l'existence du fichier source
    if not source_csv_path.exists():
        print(f"[ERREUR] Le fichier source '{source_csv_path}' est introuvable.")
        sys.exit(1)

    # Chargement du fichier CSV source via pandas
    print(f"[INFO] Lecture du fichier source '{source_csv_path}'...")
    raw_df = pd.read_csv(source_csv_path)
    print(f"[OK] {len(raw_df)} lignes lues depuis le fichier source.")

    # Application du calcul des 5 features dérivées (Feature Engineering)
    print("[INFO] Calcul des 5 features synthétiques (Thermal_Stress, Power_Efficiency...)...")
    enriched_df = compute_engineered_features(raw_df)
    print("[OK] 5 colonnes calculées avec succès.")

    # Ajout des timestamps exigés par Feast
    print("[INFO] Génération des horodatages temporels (timestamp UTC)...")
    final_df = generate_time_stamps(enriched_df)
    print(f"[OK] Période temporelle couverte : de {final_df['timestamp'].min()} à {final_df['timestamp'].max()}")

    # Sélection stricte des colonnes à stocker dans le Feature Store
    # (Nous conservons la clé d'entité, les timestamps, et les 13 colonnes de features)
    feature_columns = [
        "Turbine_ID",
        "timestamp",
        "created_timestamp",
        # Les 8 capteurs physiques bruts
        "Rotor_Speed_RPM",
        "Wind_Speed_mps",
        "Power_Output_kW",
        "Gearbox_Oil_Temp_C",
        "Generator_Bearing_Temp_C",
        "Vibration_Level_mmps",
        "Ambient_Temp_C",
        "Humidity_pct",
        # Les 5 features calculées
        "Thermal_Imbalance",
        "Power_Efficiency",
        "Thermal_Stress",
        "Vibration_per_RPM",
        "Heat_Humidity_Index",
    ]

    # Filtrage du DataFrame final avec uniquement les colonnes sélectionnées
    store_df = final_df[feature_columns].copy()

    # Cast explicite des types pour garantir la compatibilité absolue avec Feast et Redis
    store_df["Turbine_ID"] = store_df["Turbine_ID"].astype("int64")
    float_cols = [
        "Rotor_Speed_RPM", "Wind_Speed_mps", "Power_Output_kW",
        "Gearbox_Oil_Temp_C", "Generator_Bearing_Temp_C", "Vibration_Level_mmps",
        "Ambient_Temp_C", "Humidity_pct", "Thermal_Imbalance", "Power_Efficiency",
        "Thermal_Stress", "Vibration_per_RPM", "Heat_Humidity_Index"
    ]
    for col in float_cols:
        store_df[col] = store_df[col].astype("float64")

    # Définition du répertoire de destination pour le stockage Parquet
    output_dir = Path("feature_repository/data")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Définition du chemin du fichier Parquet final
    output_parquet_path = output_dir / "turbine_features.parquet"

    # Sauvegarde au format Parquet optimisé (compressé en Snappy par défaut)
    print(f"[INFO] Sauvegarde du fichier Parquet dans '{output_parquet_path}'...")
    store_df.to_parquet(output_parquet_path, index=False, engine="pyarrow")
    
    # Calcul et affichage de la taille du fichier produit
    file_size_mb = output_parquet_path.stat().st_size / (1024 * 1024)
    print(f"[OK] Fichier Parquet sauvegardé avec succès ! Taille : {file_size_mb:.2f} MB")
    print(f"[INFO] Nombre d'éoliennes uniques : {store_df['Turbine_ID'].nunique()} (IDs : {store_df['Turbine_ID'].unique().tolist()})")
    print(f"[INFO] Nombre total d'enregistrements : {len(store_df)}")
    print(f"[INFO] Nombre de colonnes de features prêtes : {len(float_cols)}")

    print("\n" + "=" * 80)
    print(" PRÉPARATION TERMINÉE : LES DONNÉES SONT PRÊTES POUR FEAST")
    print("=" * 80 + "\n")


# Point d'entrée standard du script Python
if __name__ == "__main__":
    main()
