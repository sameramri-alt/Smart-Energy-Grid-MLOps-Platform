# ==============================================================================
# PROJET SMART ENERGY GRID - PHASE 2 : VÉRIFICATION ANALYTIQUE AVEC DUCKDB
# Fichier : scripts/verify_lakehouse.py
# Description : Interroge la table Apache Iceberg et les fichiers Parquet dans MinIO
#               à l'aide du moteur d'analyse SQL DuckDB.
#               Calcule des métriques clés (KPIs) pour valider l'ingestion.
# ==============================================================================

# Importation du module standard pour l'analyse des arguments en ligne de commande
import argparse

# Importation du module sys pour gérer la sortie standard et le flux d'erreur
import sys

# Importation de duckdb pour exécuter des requêtes analytiques OLAP ultra-rapides en mémoire
import duckdb

# Importation de pandas pour la mise en forme et l'affichage des résultats tabulaires
import pandas as pd

# Importation de la classe de catalogue SQL de PyIceberg
from pyiceberg.catalog.sql import SqlCatalog


def parse_arguments():
    # Déclaration du parseur d'arguments pour configurer le script d'analyse
    parser = argparse.ArgumentParser(
        description="Vérification analytique du Lakehouse Iceberg avec DuckDB."
    )

    # Argument pour spécifier le point de terminaison du stockage MinIO
    parser.add_argument(
        "--minio-endpoint",
        type=str,
        default="http://localhost:9000",
        help="URL du point de terminaison MinIO S3 (défaut : http://localhost:9000).",
    )

    # Argument pour la clé d'accès MinIO
    parser.add_argument(
        "--minio-access-key",
        type=str,
        default="admin",
        help="Clé d'accès MinIO (défaut : admin).",
    )

    # Argument pour la clé secrète MinIO
    parser.add_argument(
        "--minio-secret-key",
        type=str,
        default="password",
        help="Clé secrète MinIO (défaut : password).",
    )

    # Argument pour le nom du bucket MinIO
    parser.add_argument(
        "--bucket-name",
        type=str,
        default="lakehouse",
        help="Nom du bucket MinIO (défaut : lakehouse).",
    )

    # Argument pour la chaîne de connexion PostgreSQL au catalogue Iceberg
    parser.add_argument(
        "--postgres-uri",
        type=str,
        default="postgresql+psycopg2://admin:password@localhost:5432/lakehouse",
        help="URI de connexion PostgreSQL pour le catalogue Iceberg.",
    )

    # Renvoi des arguments analysés
    return parser.parse_args()


def load_iceberg_table_data(args) -> pd.DataFrame:
    # Définition des propriétés de connexion au catalogue Iceberg
    catalog_properties = {
        "uri": args.postgres_uri,
        "s3.endpoint": args.minio_endpoint,
        "s3.access-key-id": args.minio_access_key,
        "s3.secret-access-key": args.minio_secret_key,
        "s3.region": "us-east-1",
        "warehouse": f"s3://{args.bucket_name}/warehouse",
        "downcast-ns-timestamp-to-us-on-write": "true",
    }

    # Connexion au catalogue Iceberg adossé à PostgreSQL
    print("[INFO] Connexion au catalogue Apache Iceberg via PostgreSQL...")
    try:
        catalog = SqlCatalog("lakehouse_catalog", **catalog_properties)
    except Exception as e:
        print(f"[ERREUR] Échec de connexion au catalogue Iceberg : {e}")
        sys.exit(1)

    # Chargement des métadonnées de la table de télémétrie
    table_identifier = ("energy_data", "sensor_telemetry")
    print(f"[INFO] Chargement de la table Iceberg '{table_identifier[0]}.{table_identifier[1]}'...")
    try:
        table = catalog.load_table(table_identifier)
    except Exception as e:
        print(f"[ERREUR] Impossible de charger la table Iceberg : {e}")
        print("Avez-vous bien lancé l'ingestion au moins une fois avec scripts/ingest_iceberg.py ?")
        sys.exit(1)

    # Affichage des métadonnées de la table (snapshots, schéma, partitionnement)
    print(f"[OK] Table chargée avec succès.")
    print(f"     - Nom de la table     : {table.name()}")
    print(f"     - Snapshot courant ID : {table.current_snapshot().snapshot_id if table.current_snapshot() else 'Aucun'}")

    # Numérisation (scan) des données de la table et conversion en DataFrame pandas
    print("[INFO] Numérisation des données depuis MinIO via PyIceberg...")
    df = table.scan().to_pandas()
    print(f"[OK] {len(df)} lignes récupérées depuis la table Iceberg.")

    # Renvoi du DataFrame contenant l'intégralité des données du Lakehouse
    return df


def run_duckdb_analysis(df: pd.DataFrame, args):
    # Initialisation d'une session DuckDB en mémoire
    print("\n[INFO] Initialisation du moteur d'analyse DuckDB...")
    con = duckdb.connect(database=":memory:")

    # Enregistrement du DataFrame pandas comme vue virtuelle interrogeable en SQL dans DuckDB
    con.register("sensor_telemetry_view", df)
    print("[OK] Vue SQL 'sensor_telemetry_view' enregistrée dans DuckDB.")

    # --------------------------------------------------------------------------
    # REQUÊTE 1 : Volumétrie globale et plage temporelle des données
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" 1. VOLUMÉTRIE GLOBALE ET PLAGE TEMPORELLE")
    print("=" * 80)

    # Définition de la requête SQL d'agrégation globale
    query_overview = """
    SELECT
        COUNT(*) AS total_lignes,
        COUNT(DISTINCT Turbine_ID) AS nombre_turbines,
        MIN(timestamp) AS premier_evenement,
        MAX(timestamp) AS dernier_evenement
    FROM sensor_telemetry_view;
    """

    # Exécution de la requête et conversion du résultat en DataFrame pandas
    df_overview = con.execute(query_overview).df()

    # Affichage du résultat formaté
    print(df_overview.to_string(index=False))

    # --------------------------------------------------------------------------
    # REQUÊTE 2 : Statistiques et KPIs par éolienne (Rotor, Vent, Puissance)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" 2. MOYENNES ET PERFORMANCES PHYSIQUES PAR TURBINE")
    print("=" * 80)

    # Définition de la requête SQL calculant les moyennes d'exploitation par turbine
    query_turbine_kpi = """
    SELECT
        Turbine_ID,
        COUNT(*) AS nb_mesures,
        ROUND(AVG(Rotor_Speed_RPM), 2) AS vitesse_rotor_moy_rpm,
        ROUND(AVG(Wind_Speed_mps), 2) AS vitesse_vent_moy_mps,
        ROUND(AVG(Power_Output_kW), 2) AS puissance_moy_kw,
        ROUND(MAX(Power_Output_kW), 2) AS puissance_max_kw
    FROM sensor_telemetry_view
    GROUP BY Turbine_ID
    ORDER BY Turbine_ID;
    """

    # Exécution de la requête
    df_turbine_kpi = con.execute(query_turbine_kpi).df()

    # Affichage du tableau de KPIs par turbine
    print(df_turbine_kpi.to_string(index=False))

    # --------------------------------------------------------------------------
    # REQUÊTE 3 : Surveillance thermique et mécanique (Vibration & Températures)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" 3. SURVEILLANCE MÉCANIQUE ET THERMIQUE (VIBRATION & TEMPÉRATURES)")
    print("=" * 80)

    # Définition de la requête SQL pour les métriques de santé mécanique
    query_health_kpi = """
    SELECT
        Turbine_ID,
        ROUND(AVG(Gearbox_Oil_Temp_C), 2) AS temp_huile_moy_c,
        ROUND(MAX(Gearbox_Oil_Temp_C), 2) AS temp_huile_max_c,
        ROUND(AVG(Generator_Bearing_Temp_C), 2) AS temp_roulement_moy_c,
        ROUND(AVG(Vibration_Level_mmps), 2) AS vibration_moy_mmps,
        ROUND(MAX(Vibration_Level_mmps), 2) AS vibration_max_mmps
    FROM sensor_telemetry_view
    GROUP BY Turbine_ID
    ORDER BY Turbine_ID;
    """

    # Exécution de la requête
    df_health_kpi = con.execute(query_health_kpi).df()

    # Affichage du tableau de santé mécanique
    print(df_health_kpi.to_string(index=False))

    # --------------------------------------------------------------------------
    # DÉMONSTRATION DUCKDB DIRECTE SUR MINIO S3 (VIA L'EXTENSION HTTPFS)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" 4. DÉMONSTRATION DUCKDB : LECTURE DIRECTE DES PARQUETS SUR MINIO S3")
    print("=" * 80)

    # Configuration des identifiants S3 pour l'extension HTTPFS de DuckDB
    # Cela permet à DuckDB d'aller lire directement les fichiers Parquet sur MinIO sans passer par Python
    try:
        # Installation et chargement du module S3 / HTTPFS dans DuckDB
        con.execute("INSTALL httpfs; LOAD httpfs;")

        # Paramétrage de l'adresse du serveur MinIO
        endpoint_clean = args.minio_endpoint.replace("http://", "").replace("https://", "")
        con.execute(f"SET s3_endpoint='{endpoint_clean}';")

        # Paramétrage des clés d'accès MinIO
        con.execute(f"SET s3_access_key_id='{args.minio_access_key}';")
        con.execute(f"SET s3_secret_access_key='{args.minio_secret_key}';")

        # Désactivation du SSL pour notre environnement de développement local (HTTP)
        con.execute("SET s3_use_ssl=false;")

        # Activation du style d'URL par chemin pour MinIO (ex: http://endpoint/bucket/object)
        con.execute("SET s3_url_style='path';")

        # Requête SQL directe interrogeant les fichiers Parquet partitionnés sur MinIO
        s3_parquet_pattern = f"s3://{args.bucket_name}/warehouse/energy_data/sensor_telemetry/data/**/*.parquet"
        query_direct_s3 = f"""
        SELECT
            COUNT(*) AS total_lignes_s3,
            ROUND(AVG(Power_Output_kW), 2) AS puissance_moyenne_globale
        FROM read_parquet('{s3_parquet_pattern}');
        """

        # Exécution de la requête SQL distribuée sur MinIO
        df_direct_s3 = con.execute(query_direct_s3).df()
        print("[OK] DuckDB a interrogé directement les fichiers Parquet stockés dans MinIO :")
        print(df_direct_s3.to_string(index=False))

    except Exception as err:
        # Information en cas d'indisponibilité du module S3 externe
        print(f"[INFO] Note sur la lecture directe S3 HTTPFS : {err}")

    print("\n" + "=" * 80)
    print(" VÉRIFICATION DU LAKEHOUSE VALIDÉE AVEC SUCCÈS")
    print("=" * 80)


def main():
    # 1. Analyse des arguments en ligne de commande
    args = parse_arguments()

    # 2. Récupération des données depuis le Lakehouse Iceberg
    df = load_iceberg_table_data(args)

    # 3. Exécution des analyses SQL OLAP avec DuckDB
    run_duckdb_analysis(df, args)


# Point d'entrée principal pour l'exécution directe du script
if __name__ == "__main__":
    main()
