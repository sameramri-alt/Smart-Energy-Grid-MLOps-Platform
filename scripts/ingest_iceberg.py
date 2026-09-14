# ==============================================================================
# PROJET SMART ENERGY GRID - PHASE 2 : INGESTION DANS LE LAKEHOUSE APACHE ICEBERG
# Fichier : scripts/ingest_iceberg.py
# Description : Consomme les événements télémétriques depuis le flux Redis Stream,
#               les valide et les persiste de manière transactionnelle (ACID)
#               dans une table Apache Iceberg hébergée sur MinIO avec catalogue PostgreSQL.
# ==============================================================================

# Importation du module standard pour l'analyse des arguments en ligne de commande
import argparse

# Importation du module datetime pour la gestion des horodatages temporels
import datetime

# Importation du module sys pour la gestion des flux d'erreurs et arrêts du programme
import sys

# Importation du module time pour les temporisations entre deux lectures de micro-lots
import time

# Importation de boto3 pour interagir avec le stockage d'objets MinIO (compatible AWS S3)
import boto3

# Importation de pandas pour la structuration intermédiaire des données tabulaires
import pandas as pd

# Importation de pyarrow pour créer les structures de données en colonnes compatibles Iceberg
import pyarrow as pa

# Importation de pyiceberg pour administrer le catalogue et la table Iceberg
from pyiceberg.catalog.sql import SqlCatalog

# Importation des classes de partitionnement pour découper la table par turbine
from pyiceberg.partitioning import PartitionField, PartitionSpec

# Importation de la classe Schema pour définir la structure de la table Iceberg
from pyiceberg.schema import Schema

# Importation de la transformation d'identité pour le partitionnement direct sur la colonne
from pyiceberg.transforms import IdentityTransform

# Importation des types de données officiels supportés par le standard Apache Iceberg
from pyiceberg.types import DoubleType, LongType, NestedField, TimestampType

# Importation de redis pour se connecter au broker et lire le flux d'événements
import redis

# ==============================================================================
# IMPORT FEAST (FEATURE STORE) — Connexion Lakehouse ↔ Feature Store en temps réel
# ==============================================================================
# Importation de FeatureStore pour écrire la dernière mesure dans le Online Store (Redis)
# C'est le maillon qui connecte le Tuyau A (MinIO Lakehouse) au Tuyau B (Feature Store Redis)
from feast import FeatureStore


def parse_arguments():
    # Déclaration du parseur d'arguments pour configurer l'ingestion
    parser = argparse.ArgumentParser(
        description="Ingestion continue de télémétrie vers Apache Iceberg et MinIO."
    )

    # Option pour ne faire que l'initialisation du bucket et de la table sans consommer
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="Initialise uniquement le bucket MinIO et la table Iceberg, puis quitte.",
    )

    # Option pour forcer la suppression et recréation propre de la table Iceberg
    parser.add_argument(
        "--recreate-table",
        action="store_true",
        help="Supprime et recrée la table Iceberg pour repartir sur un schéma neuf.",
    )

    # Taille d'un micro-lot (batch) d'ingestion (nombre d'événements par écriture Iceberg)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Nombre d'événements à regrouper par transaction Iceberg (défaut : 50).",
    )

    # Nombre maximum de micro-lots à traiter (0 = exécution infinie en continu)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Nombre maximum de micro-lots à ingérer (0 = infini, défaut : 0).",
    )

    # Hôte de connexion au conteneur MinIO
    parser.add_argument(
        "--minio-endpoint",
        type=str,
        default="http://localhost:9000",
        help="URL du point de terminaison MinIO S3 (défaut : http://localhost:9000).",
    )

    # Identifiant d'accès MinIO (configuré dans Terraform / docker-compose)
    parser.add_argument(
        "--minio-access-key",
        type=str,
        default="admin",
        help="Clé d'accès MinIO (défaut : admin).",
    )

    # Mot de passe secret MinIO
    parser.add_argument(
        "--minio-secret-key",
        type=str,
        default="password",
        help="Clé secrète MinIO (défaut : password).",
    )

    # Nom du bucket de stockage dans MinIO
    parser.add_argument(
        "--bucket-name",
        type=str,
        default="lakehouse",
        help="Nom du bucket MinIO pour le Lakehouse (défaut : lakehouse).",
    )

    # URI de connexion à la base de données PostgreSQL pour le catalogue Iceberg
    parser.add_argument(
        "--postgres-uri",
        type=str,
        default="postgresql+psycopg2://admin:password@localhost:5432/lakehouse",
        help="Chaîne de connexion SQLAlchemy vers PostgreSQL pour le catalogue Iceberg.",
    )

    # Hôte du serveur Redis
    parser.add_argument(
        "--redis-host",
        type=str,
        default="localhost",
        help="Hôte du serveur Redis (défaut : localhost).",
    )

    # Port du serveur Redis
    parser.add_argument(
        "--redis-port",
        type=int,
        default=6379,
        help="Port d'écoute du serveur Redis (défaut : 6379).",
    )

    # Clé du flux Redis Stream à lire
    parser.add_argument(
        "--stream-key",
        type=str,
        default="stream:turbines",
        help="Nom de la clé Redis Stream source (défaut : stream:turbines).",
    )

    # Renvoi des arguments analysés
    return parser.parse_args()


def ensure_minio_bucket(endpoint: str, access_key: str, secret_key: str, bucket_name: str):
    # Création du client S3 via boto3 configuré sur le port de MinIO
    s3_client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="us-east-1",
    )

    # Récupération de la liste des buckets existants sur le serveur MinIO
    existing_buckets = [b["Name"] for b in s3_client.list_buckets().get("Buckets", [])]

    # Vérification si notre bucket de Lakehouse existe déjà
    if bucket_name not in existing_buckets:
        # S'il n'existe pas encore, nous le créons immédiatement
        s3_client.create_bucket(Bucket=bucket_name)
        print(f"[OK] Bucket MinIO '{bucket_name}' créé avec succès.")
    else:
        # S'il existe déjà, confirmation dans la console
        print(f"[OK] Le bucket MinIO '{bucket_name}' existe déjà.")


def get_iceberg_catalog(postgres_uri: str, minio_endpoint: str, access_key: str, secret_key: str, bucket_name: str) -> SqlCatalog:
    # Définition des propriétés de configuration du catalogue Apache Iceberg
    catalog_properties = {
        # URI de connexion à PostgreSQL où Iceberg sauvegarde ses pointeurs et commits
        "uri": postgres_uri,
        # URL de destination pour écrire les fichiers Parquet dans MinIO
        "s3.endpoint": minio_endpoint,
        # Identifiants de connexion au stockage objet S3
        "s3.access-key-id": access_key,
        "s3.secret-access-key": secret_key,
        "s3.region": "us-east-1",
        # Chemin racine du warehouse où sont logés les dossiers et fichiers de données
        "warehouse": f"s3://{bucket_name}/warehouse",
        # Conversion automatique des timestamps haute précision en microsecondes lors de l'écriture
        "downcast-ns-timestamp-to-us-on-write": "true",
    }

    # Instanciation du catalogue SQL adossé à PostgreSQL et MinIO
    catalog = SqlCatalog("lakehouse_catalog", **catalog_properties)

    # Confirmation de l'initialisation du catalogue
    print("[OK] Catalogue Apache Iceberg (SQL sur PostgreSQL + MinIO) initialisé.")

    # Renvoi de l'objet catalogue prêt à être manipulé
    return catalog


def get_or_create_iceberg_table(catalog: SqlCatalog, recreate: bool = False):
    # Nom de l'espace de noms (namespace / schéma logique)
    namespace = "energy_data"

    # Nom de la table Iceberg dans le catalogue
    table_name = "sensor_telemetry"

    # Identifiant complet composé du namespace et du nom de la table
    table_identifier = (namespace, table_name)

    # Si l'option de recréation est activée, on supprime d'abord l'ancienne table
    if recreate:
        try:
            catalog.drop_table(table_identifier)
            print(f"[OK] Ancienne table Iceberg '{namespace}.{table_name}' supprimée.")
        except Exception:
            pass

    # Création du namespace s'il n'existe pas encore dans le catalogue PostgreSQL
    try:
        catalog.create_namespace(namespace)
        print(f"[OK] Espace de noms '{namespace}' créé dans le catalogue.")
    except Exception:
        # Le namespace existe déjà, on poursuit sans erreur
        pass

    # Définition du schéma strict d'Apache Iceberg pour notre télémétrie d'éoliennes
    # RÈGLE D'OR MLOPS : Nous ne stockons QUE les grandeurs physiques réelles (sans label de panne)
    iceberg_schema = Schema(
        # Identifiant 1 : Horodatage précis de la mesure (type timestamp Iceberg)
        NestedField(field_id=1, name="timestamp", field_type=TimestampType(), required=False),
        # Identifiant 2 : ID de l'éolienne (type entier long 64-bit)
        NestedField(field_id=2, name="Turbine_ID", field_type=LongType(), required=False),
        # Identifiant 3 : Vitesse de rotation des pales en tours/min
        NestedField(field_id=3, name="Rotor_Speed_RPM", field_type=DoubleType(), required=False),
        # Identifiant 4 : Vitesse du vent mesurée par l'anémomètre
        NestedField(field_id=4, name="Wind_Speed_mps", field_type=DoubleType(), required=False),
        # Identifiant 5 : Puissance électrique instantanée générée en kW
        NestedField(field_id=5, name="Power_Output_kW", field_type=DoubleType(), required=False),
        # Identifiant 6 : Température d'huile du multiplicateur de vitesse
        NestedField(field_id=6, name="Gearbox_Oil_Temp_C", field_type=DoubleType(), required=False),
        # Identifiant 7 : Température des roulements de la génératrice
        NestedField(field_id=7, name="Generator_Bearing_Temp_C", field_type=DoubleType(), required=False),
        # Identifiant 8 : Niveau de vibration mécanique mesuré sur la nacelle
        NestedField(field_id=8, name="Vibration_Level_mmps", field_type=DoubleType(), required=False),
        # Identifiant 9 : Température de l'air ambiant extérieur
        NestedField(field_id=9, name="Ambient_Temp_C", field_type=DoubleType(), required=False),
        # Identifiant 10 : Taux d'humidité relative dans l'atmosphère
        NestedField(field_id=10, name="Humidity_pct", field_type=DoubleType(), required=False),
    )

    # Définition de la stratégie de partitionnement : partitionner par Turbine_ID
    # Cela permet à Iceberg d'isoler physiquement les fichiers de la Turbine 1 et de la Turbine 2
    partition_spec = PartitionSpec(
        PartitionField(
            source_id=2,
            field_id=1000,
            transform=IdentityTransform(),
            name="turbine_partition",
        )
    )

    # Tentative de création de la table avec son schéma et son partitionnement
    try:
        table = catalog.create_table(
            identifier=table_identifier,
            schema=iceberg_schema,
            partition_spec=partition_spec,
        )
        print(f"[OK] Table Iceberg '{namespace}.{table_name}' créée avec succès.")
    except Exception:
        # Si la table existe déjà, nous chargeons la table existante
        table = catalog.load_table(table_identifier)
        print(f"[OK] Table Iceberg existante '{namespace}.{table_name}' chargée.")

    # Renvoi de l'objet table Iceberg
    return table


def push_to_feast_online_store(records: list, feast_repo_path: str) -> None:
    """
    Pousse la DERNIÈRE mesure de chaque turbine directement dans le Online Store Feast (Redis).

    Cette fonction est le maillon manquant entre le Lakehouse et le Feature Store :
    - Après chaque lot ingéré dans Iceberg/MinIO (Tuyau A),
    - On calcule les 5 features dérivées et on écrit le résultat dans Redis via Feast (Tuyau B).
    - Résultat : le Dashboard Streamlit voit des données fraîches à chaque rafraîchissement.

    Args :
        records      : Liste de dictionnaires représentant les événements du lot (capteurs bruts).
        feast_repo_path : Chemin vers le dossier 'feature_repository/' contenant feature_store.yaml.
    """
    import datetime

    try:
        # --- 1. Chargement du Feature Store Feast connecté à Redis ---
        store = FeatureStore(repo_path=feast_repo_path)

        # --- 2. Conversion des événements bruts en DataFrame pandas ---
        df_raw = pd.DataFrame(records)

        # Conversion des types numériques
        for col in ["Rotor_Speed_RPM", "Wind_Speed_mps", "Power_Output_kW",
                    "Gearbox_Oil_Temp_C", "Generator_Bearing_Temp_C",
                    "Vibration_Level_mmps", "Ambient_Temp_C", "Humidity_pct"]:
            df_raw[col] = df_raw[col].astype("float64")
        df_raw["Turbine_ID"] = df_raw["Turbine_ID"].astype("int64")

        # --- 3. Calcul des 5 features d'ingénierie physique (identique à Phase 3 & Phase 5) ---
        # Ces formules doivent être STRICTEMENT identiques à celles de app/main.py pour éviter
        # tout Training-Serving Skew (le grand danger en MLOps)
        df_raw["Thermal_Imbalance"] = df_raw["Gearbox_Oil_Temp_C"] - df_raw["Generator_Bearing_Temp_C"]
        df_raw["Power_Efficiency"]  = df_raw["Power_Output_kW"] / (df_raw["Rotor_Speed_RPM"] + 1e-5)
        df_raw["Thermal_Stress"]    = df_raw["Gearbox_Oil_Temp_C"] - df_raw["Ambient_Temp_C"]
        df_raw["Vibration_per_RPM"] = df_raw["Vibration_Level_mmps"] / (df_raw["Rotor_Speed_RPM"] + 1e-5)
        df_raw["Heat_Humidity_Index"] = df_raw["Ambient_Temp_C"] * (df_raw["Humidity_pct"] / 100.0)

        # --- 4. Garder uniquement la DERNIÈRE mesure de chaque turbine du lot ---
        # (La plus récente remplace la précédente dans le Online Store Redis)
        df_last = df_raw.sort_values("timestamp").groupby("Turbine_ID").last().reset_index()

        # --- 5. Ajout des colonnes d'horodatage requises par Feast ---
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        df_last["timestamp"]         = now_utc
        df_last["created_timestamp"] = now_utc

        # --- 6. Écriture dans le Online Store Feast (Redis) ---
        # C'est ici que la magie opère : Feast écrit dans Redis clé par clé
        # La clé Redis est basée sur Turbine_ID et le nom de la Feature View
        store.write_to_online_store("turbine_features", df_last)

        # Confirmation dans la console avec les valeurs fraîches de chaque turbine
        for _, row in df_last.iterrows():
            print(
                f"  [FEAST] Turbine {int(row['Turbine_ID'])} mis à jour dans Redis | "
                f"RPM={row['Rotor_Speed_RPM']:.1f} | "
                f"Puiss.={row['Power_Output_kW']:.0f} kW | "
                f"T.Huile={row['Gearbox_Oil_Temp_C']:.1f}°C | "
                f"Vib.={row['Vibration_Level_mmps']:.2f} mm/s"
            )

    except Exception as e:
        # Ne jamais faire planter l'ingestion Iceberg si Feast échoue
        # Feast est optionnel par rapport au Lakehouse (MinIO reste toujours la source de vérité)
        print(f"  [AVERTISSEMENT] Mise à jour Feast échouée (n'impacte pas l'ingestion) : {e}")


def build_pyarrow_batch(records: list) -> pa.Table:
    # Conversion de la liste de dictionnaires bruts en DataFrame pandas
    df = pd.DataFrame(records)

    # Conversion de la chaîne de caractères temporelle en datetime pandas UTC
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Suppression du fuseau horaire (tz-naive) pour correspondre au TimestampType sans fuseau d'Iceberg
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)

    # Conversion explicite du timestamp en microsecondes (us) requis par le standard Iceberg
    df["timestamp"] = df["timestamp"].astype("datetime64[us]")

    # Conversion explicite de l'identifiant de la turbine en entier 64 bits (LongType)
    df["Turbine_ID"] = df["Turbine_ID"].astype("int64")

    # Liste des colonnes de mesures physiques à convertir en nombres décimaux (float64 / DoubleType)
    numeric_columns = [
        "Rotor_Speed_RPM",
        "Wind_Speed_mps",
        "Power_Output_kW",
        "Gearbox_Oil_Temp_C",
        "Generator_Bearing_Temp_C",
        "Vibration_Level_mmps",
        "Ambient_Temp_C",
        "Humidity_pct",
    ]

    # Application de la conversion float64 pour chaque métrique physique
    for col in numeric_columns:
        df[col] = df[col].astype("float64")

    # Ordre strict des colonnes aligné avec le schéma Iceberg
    ordered_cols = ["timestamp", "Turbine_ID"] + numeric_columns
    df = df[ordered_cols]

    # Transformation du DataFrame pandas en table PyArrow optimisée
    arrow_table = pa.Table.from_pandas(df, preserve_index=False)

    # Renvoi de la table PyArrow prête pour l'insertion
    return arrow_table


def run_ingest():
    # 1. Analyse des paramètres passés en ligne de commande
    args = parse_arguments()

    # 2. Vérification / création du bucket S3 dans MinIO
    ensure_minio_bucket(
        endpoint=args.minio_endpoint,
        access_key=args.minio_access_key,
        secret_key=args.minio_secret_key,
        bucket_name=args.bucket_name,
    )

    # 3. Initialisation du catalogue Iceberg
    catalog = get_iceberg_catalog(
        postgres_uri=args.postgres_uri,
        minio_endpoint=args.minio_endpoint,
        access_key=args.minio_access_key,
        secret_key=args.minio_secret_key,
        bucket_name=args.bucket_name,
    )

    # 4. Création ou chargement de la table Iceberg partitionnée
    table = get_or_create_iceberg_table(catalog, recreate=args.recreate_table)

    # Si l'utilisateur a demandé uniquement l'initialisation (--init-only), on s'arrête ici
    if args.init_only:
        print("[INFO] Initialisation complète terminée (--init-only). Arrêt.")
        return

    # 5. Connexion au broker Redis pour lire le flux d'événements
    print(f"[INFO] Connexion à Redis Stream sur {args.redis_host}:{args.redis_port}...")
    r = redis.Redis(host=args.redis_host, port=args.redis_port, decode_responses=True)

    # Vérification de la connectivité avec Redis
    try:
        r.ping()
    except redis.ConnectionError as e:
        print(f"[ERREUR] Impossible de joindre Redis : {e}")
        sys.exit(1)

    # Message d'information sur le démarrage de l'écoute du flux
    print("=" * 70)
    print(" DÉMARRAGE DE L'INGESTION CONTINUE DANS LE LAKEHOUSE APACHE ICEBERG")
    print(f" - Flux Redis surveillé : {args.stream_key}")
    print(f" - Taille des micro-lots : {args.batch_size} événements par transaction")
    print(f" - Limite de micro-lots  : {'Infinie' if args.max_batches <= 0 else f'{args.max_batches} lots'}")
    print(f" - Cible MinIO / Table   : s3://{args.bucket_name}/.../sensor_telemetry")
    print("=" * 70)

    # Compteur de micro-lots ingérés avec succès
    batches_processed = 0

    # Compteur cumulatif d'enregistrements écrits dans Iceberg
    total_records_ingested = 0

    try:
        # Boucle principale de consommation et d'ingestion
        while True:
            # Vérification de la condition d'arrêt par nombre maximal de lots
            if args.max_batches > 0 and batches_processed >= args.max_batches:
                print(f"\n[FIN] Limite de {batches_processed} micro-lots atteinte.")
                break

            # Lecture d'un lot d'événements depuis le flux Redis avec XRANGE
            # On lit au maximum 'batch_size' messages depuis le début du flux ('-')
            raw_entries = r.xrange(args.stream_key, min="-", max="+", count=args.batch_size)

            # Si aucun message n'est disponible dans le flux
            if not raw_entries:
                # Affichage discret d'attente
                print("[ATTENTE] En attente de nouveaux événements depuis le simulateur...", end="\r")
                # Pause d'une demi-seconde avant de sonder à nouveau Redis
                time.sleep(0.5)
                continue

            # Extraction des identifiants Redis et des corps d'événements (payloads)
            message_ids = [entry[0] for entry in raw_entries]
            records = [entry[1] for entry in raw_entries]

            # Construction de la table PyArrow typée pour Iceberg
            arrow_batch = build_pyarrow_batch(records)

            # Écriture transactionnelle (ACID) dans la table Iceberg sur MinIO
            # Iceberg écrit les fichiers Parquet compressés et met à jour le catalogue dans PostgreSQL
            table.append(arrow_batch)

            # Suppression des messages traités dans Redis (XDEL) pour libérer la mémoire vive
            r.xdel(args.stream_key, *message_ids)

            # Mise à jour des compteurs statistiques
            batches_processed += 1
            current_batch_size = len(records)
            total_records_ingested += current_batch_size

            # Affichage du succès de l'ingestion du micro-lot dans Iceberg/MinIO
            print(
                f"[INGESTION] Lot #{batches_processed:04d} validé : "
                f"+{current_batch_size} événements persistés dans Iceberg | "
                f"Total cumulé : {total_records_ingested} lignes"
            )

            # ==================================================================
            # PUSH STREAMING VERS FEAST (FEATURE STORE TEMPS RÉEL)
            # Connexion automatique Lakehouse → Feature Store après chaque lot
            # Les dernières mesures de chaque turbine sont mises à jour dans Redis
            # pour que le Dashboard Streamlit les affiche en temps réel
            # ==================================================================
            push_to_feast_online_store(
                records=records,
                feast_repo_path="feature_repository",
            )

    except KeyboardInterrupt:
        # Arrêt sécurisé si l'utilisateur appuie sur Ctrl+C
        print("\n[INFO] Arrêt de l'ingestion demandé par l'utilisateur (Ctrl+C).")

    # Bilan récapitulatif de fin d'exécution
    print("-" * 70)
    print(f"[BILAN] Total micro-lots traités : {batches_processed}")
    print(f"[BILAN] Total lignes persistées  : {total_records_ingested}")
    print(f"[BILAN] État table Iceberg      : Données partitionnées et synchronisées.")
    print("-" * 70)


# Exécution principale du script
if __name__ == "__main__":
    run_ingest()
