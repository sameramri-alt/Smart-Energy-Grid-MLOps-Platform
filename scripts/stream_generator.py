# ==============================================================================
# PROJET SMART ENERGY GRID - PHASE 2 : SIMULATEUR IOT TEMPS RÉEL
# Fichier : scripts/stream_generator.py
# Description : Simule les capteurs physiques des turbines éoliennes (1 et 2).
#               Lit les mesures du fichier data/Test.csv (données non vues par l'IA),
#               génère un horodatage précis et publie les événements dans Redis Stream.
# ==============================================================================

# Importation du module standard pour analyser les arguments passés en ligne de commande
import argparse

# Importation du module datetime pour manipuler et générer les horodatages précis
import datetime

# Importation du module os pour vérifier l'existence des fichiers et manipuler les chemins
import os

# Importation du module sys pour gérer la sortie propre et l'encodage du terminal
import sys

# Importation du module time pour contrôler la cadence d'émission (50 événements/seconde)
import time

# Importation de pandas pour charger et itérer efficacement sur le fichier CSV
import pandas as pd

# Importation de redis pour se connecter au serveur Redis et publier dans Redis Stream
import redis


def parse_arguments():
    # Création du parseur d'arguments avec une description claire du script
    parser = argparse.ArgumentParser(
        description="Simulateur IoT de télémétrie pour éoliennes connectées."
    )

    # Argument pour définir le débit d'émission en événements par seconde (défaut : 50)
    parser.add_argument(
        "--rate",
        type=float,
        default=50.0,
        help="Cadence d'émission en événements par seconde (défaut : 50)."
    )

    # Argument pour limiter le nombre total d'événements à émettre (utile pour les tests)
    # Une valeur de 0 ou négative signifie une exécution en continu (infinie)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Nombre maximum d'événements à émettre (0 = infini, défaut : 0)."
    )

    # Argument pour spécifier le chemin vers le fichier de données sources
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/Test.csv",
        help="Chemin vers le fichier CSV de télémétrie (défaut : data/Test.csv)."
    )

    # Argument pour configurer l'hôte du serveur Redis
    parser.add_argument(
        "--redis-host",
        type=str,
        default="localhost",
        help="Hôte du serveur Redis (défaut : localhost)."
    )

    # Argument pour configurer le port d'écoute du serveur Redis
    parser.add_argument(
        "--redis-port",
        type=int,
        default=6379,
        help="Port du serveur Redis (défaut : 6379)."
    )

    # Argument pour spécifier le nom de la clé Redis Stream utilisée
    parser.add_argument(
        "--stream-key",
        type=str,
        default="stream:turbines",
        help="Nom de la clé Redis Stream de destination (défaut : stream:turbines)."
    )

    # Analyse et renvoi des arguments fournis par l'utilisateur
    return parser.parse_args()


def connect_redis(host: str, port: int) -> redis.Redis:
    # Création de l'instance client Redis avec décodage automatique des réponses en texte UTF-8
    client = redis.Redis(host=host, port=port, decode_responses=True)

    # Tentative d'envoi d'un signal PING pour vérifier que le conteneur Redis est joignable
    try:
        client.ping()
        # Message de confirmation si la connexion est réussie
        print(f"[OK] Connexion réussie à Redis sur {host}:{port}")
    except redis.ConnectionError as err:
        # En cas d'échec, message explicite et arrêt immédiat du script
        print(f"[ERREUR] Impossible de se connecter à Redis ({host}:{port}) : {err}")
        print("Veuillez vérifier que le conteneur Docker 'redis' est bien démarré.")
        sys.exit(1)

    # Retourne le client Redis prêt à être utilisé
    return client


def load_telemetry_data(csv_path: str) -> pd.DataFrame:
    # Vérification que le fichier CSV source existe bien sur le disque
    if not os.path.exists(csv_path):
        print(f"[ERREUR] Le fichier source '{csv_path}' est introuvable.")
        print("Veuillez vérifier que le fichier est bien présent dans le dossier data/.")
        sys.exit(1)

    # Lecture complète du fichier CSV dans un DataFrame pandas
    print(f"[INFO] Chargement du fichier de télémétrie : {csv_path}...")
    df = pd.read_csv(csv_path)

    # Liste des colonnes de capteurs physiques que nous voulons transmettre
    expected_sensor_cols = [
        "Turbine_ID",
        "Rotor_Speed_RPM",
        "Wind_Speed_mps",
        "Power_Output_kW",
        "Gearbox_Oil_Temp_C",
        "Generator_Bearing_Temp_C",
        "Vibration_Level_mmps",
        "Ambient_Temp_C",
        "Humidity_pct",
    ]

    # Vérification que toutes les colonnes indispensables sont bien présentes dans le CSV
    missing_cols = [col for col in expected_sensor_cols if col not in df.columns]
    if missing_cols:
        print(f"[ERREUR] Colonnes manquantes dans le CSV : {missing_cols}")
        sys.exit(1)

    # RÈGLE D'OR MLOPS : On filtre expressément pour ne garder QUE les capteurs physiques
    # On supprime Maintenance_Label car un vrai capteur physique ne connaît pas le diagnostic de panne
    df_sensors = df[expected_sensor_cols].copy()

    # Séparation des données de la Turbine 1 et de la Turbine 2
    df_t1 = df_sensors[df_sensors["Turbine_ID"] == 1].reset_index(drop=True)
    df_t2 = df_sensors[df_sensors["Turbine_ID"] == 2].reset_index(drop=True)

    # Entrelacement (interleaving) ligne par ligne pour simuler deux éoliennes émettant en parallèle
    # Turbine 1, puis Turbine 2, puis Turbine 1, puis Turbine 2...
    df_interleaved = pd.concat([df_t1, df_t2]).sort_index(kind="merge").reset_index(drop=True)

    # Affichage du nombre de lignes chargées en mémoire
    print(f"[OK] {len(df_interleaved)} enregistrements physiques chargés (Turbine 1 & 2 entrelacées).")

    # Retourne le DataFrame final prêt pour la simulation temps réel
    return df_interleaved


def run_stream_generator():
    # 1. Récupération des paramètres en ligne de commande
    args = parse_arguments()

    # 2. Initialisation de la connexion avec le broker Redis
    r = connect_redis(host=args.redis_host, port=args.redis_port)

    # 3. Chargement des données de télémétrie depuis le fichier CSV
    telemetry_df = load_telemetry_data(args.data_path)

    # Calcul de l'intervalle de pause entre deux événements consécutifs pour respecter le débit
    # Par exemple, pour 50 événements/s, pause = 1 / 50 = 0.02 seconde (20 millisecondes)
    delay_between_events = 1.0 / args.rate if args.rate > 0 else 0.0

    # Affichage du récapitulatif de configuration du simulateur
    print("=" * 70)
    print(" DÉMARRAGE DU SIMULATEUR IOT DE TÉLÉMÉTRIE")
    print(f" - Débit cible        : {args.rate} événements/seconde")
    print(f" - Intervalle temps   : {delay_between_events * 1000:.2f} ms par événement")
    print(f" - Limite d'émission  : {'Infinie (en boucle)' if args.limit <= 0 else f'{args.limit} événements'}")
    print(f" - Clé Redis Stream   : {args.stream_key}")
    print("=" * 70)

    # Initialisation du compteur d'événements publiés
    published_count = 0

    # Mémorisation du temps de départ pour mesurer le débit réel
    start_time = time.time()

    # Récupération du nombre total de lignes disponibles dans le jeu de données
    total_rows = len(telemetry_df)

    # Index de position dans le jeu de données (permet de boucler si la simulation est infinie)
    row_index = 0

    try:
        # Boucle principale d'émission des données
        while True:
            # Vérification de la condition d'arrêt si une limite a été spécifiée
            if args.limit > 0 and published_count >= args.limit:
                print(f"\n[FIN] Limite atteinte : {published_count} événements publiés.")
                break

            # Extraction de la ligne courante sous forme de dictionnaire Python
            current_row = telemetry_df.iloc[row_index].to_dict()

            # Génération d'un horodatage UTC précis au moment exact de l'émission par le capteur
            # Format standard ISO 8601 en microsecondes (compatible Lakehouse / Iceberg)
            now_utc = datetime.datetime.now(datetime.timezone.utc)
            timestamp_str = now_utc.isoformat()

            # Construction du payload de l'événement à envoyer dans Redis Stream
            # Toutes les valeurs sont converties en chaîne de caractères pour Redis
            event_payload = {
                "timestamp": timestamp_str,
                "Turbine_ID": str(int(current_row["Turbine_ID"])),
                "Rotor_Speed_RPM": str(float(current_row["Rotor_Speed_RPM"])),
                "Wind_Speed_mps": str(float(current_row["Wind_Speed_mps"])),
                "Power_Output_kW": str(float(current_row["Power_Output_kW"])),
                "Gearbox_Oil_Temp_C": str(float(current_row["Gearbox_Oil_Temp_C"])),
                "Generator_Bearing_Temp_C": str(float(current_row["Generator_Bearing_Temp_C"])),
                "Vibration_Level_mmps": str(float(current_row["Vibration_Level_mmps"])),
                "Ambient_Temp_C": str(float(current_row["Ambient_Temp_C"])),
                "Humidity_pct": str(float(current_row["Humidity_pct"])),
            }

            # Envoi effectif de l'événement dans le flux Redis avec la commande XADD
            # Le symbole '*' indique à Redis de générer automatiquement un ID unique d'événement
            r.xadd(args.stream_key, event_payload)

            # Incrémentation du compteur d'événements transmis
            published_count += 1

            # Progression de l'index dans le jeu de données avec retour au début (modulo) si nécessaire
            row_index = (row_index + 1) % total_rows

            # Affichage périodique de l'avancement toutes les 50 émissions (ou au premier événement)
            if published_count % 50 == 0 or published_count == 1:
                elapsed = time.time() - start_time
                current_rate = published_count / elapsed if elapsed > 0 else 0
                turbine_id = event_payload["Turbine_ID"]
                power = float(event_payload["Power_Output_kW"])
                vibr = float(event_payload["Vibration_Level_mmps"])
                print(
                    f"[STREAM] Émis : {published_count:6d} msgs | "
                    f"Débit réel : {current_rate:5.1f} ev/s | "
                    f"Dernier : Turbine {turbine_id} - Puissance: {power:7.1f} kW - Vibration: {vibr:4.2f} mm/s"
                )

            # Pause calibrée pour respecter scrupuleusement la cadence demandée
            if delay_between_events > 0:
                time.sleep(delay_between_events)

    except KeyboardInterrupt:
        # Capture de l'interruption manuelle de l'utilisateur (Ctrl+C) pour un arrêt en douceur
        print("\n[INFO] Arrêt manuel du simulateur demandé par l'utilisateur (Ctrl+C).")

    # Calcul et affichage des statistiques finales de session de streaming
    total_elapsed = time.time() - start_time
    avg_rate = published_count / total_elapsed if total_elapsed > 0 else 0
    print("-" * 70)
    print(f"[BILAN] Total événements publiés : {published_count}")
    print(f"[BILAN] Durée totale d'émission   : {total_elapsed:.2f} secondes")
    print(f"[BILAN] Débit moyen effectif     : {avg_rate:.2f} événements/seconde")
    print("-" * 70)


# Point d'entrée principal pour l'exécution directe du script depuis le terminal
if __name__ == "__main__":
    run_stream_generator()
