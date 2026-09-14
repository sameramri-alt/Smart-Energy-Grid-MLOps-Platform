# ==============================================================================
# PROJET SMART ENERGY GRID - PHASE 3 : ENTRAÎNEMENT DU MODÈLE MACHINE LEARNING
# Fichier : scripts/train_model.py
# Description : Charge les données de Train.csv et Test.csv, effectue l'ingénierie
#               des features (création de 5 nouvelles colonnes synthétiques),
#               entraîne un classifieur XGBoost pour détecter les pannes d'éoliennes,
#               évalue ses performances, affiche les features les plus importantes,
#               et sauvegarde le modèle entraîné dans app/model.pkl.
# ==============================================================================

# Importation du module os pour vérifier l'existence des fichiers et gérer les chemins
import os

# Importation de numpy pour les calculs numériques efficaces sur les tableaux
import numpy as np

# Importation de pandas pour charger et manipuler les données tabulaires CSV
import pandas as pd

# Importation de joblib pour sauvegarder le modèle entraîné sous forme de fichier binaire
import joblib

# Importation du classifieur XGBoost - l'algorithme d'apprentissage automatique principal
from xgboost import XGBClassifier

# Importation des fonctions d'évaluation pour mesurer la qualité du modèle
from sklearn.metrics import (
    accuracy_score,       # Mesure le pourcentage global de bonnes prédictions
    classification_report, # Rapport détaillé (precision, recall, f1-score) par classe
    confusion_matrix,      # Matrice montrant les erreurs de classification par classe
)


# ==============================================================================
# CONSTANTES DE CONFIGURATION DU PROJET
# ==============================================================================

# Chemin vers le fichier CSV utilisé pour l'ENTRAÎNEMENT du modèle (avec labels)
TRAIN_PATH = "data/Train.csv"

# Chemin vers le fichier CSV utilisé pour l'ÉVALUATION finale du modèle (avec labels)
TEST_PATH = "data/Test.csv"

# Chemin de sauvegarde du modèle entraîné pour une utilisation future par l'API FastAPI
MODEL_OUTPUT_PATH = "app/model.pkl"

# Nom de la colonne cible que le modèle doit apprendre à prédire
TARGET_COLUMN = "Maintenance_Label"

# Liste des 8 colonnes de mesures physiques brutes issues des capteurs des éoliennes
BASE_FEATURE_COLUMNS = [
    "Rotor_Speed_RPM",           # Vitesse de rotation des pales en tours/minute
    "Wind_Speed_mps",            # Vitesse du vent en mètres par seconde
    "Power_Output_kW",           # Puissance électrique générée en kilowatts
    "Gearbox_Oil_Temp_C",        # Température de l'huile du multiplicateur (°C)
    "Generator_Bearing_Temp_C",  # Température des roulements de la génératrice (°C)
    "Vibration_Level_mmps",      # Niveau de vibration mécanique en mm/s
    "Ambient_Temp_C",            # Température ambiante extérieure (°C)
    "Humidity_pct",              # Taux d'humidité relative de l'air en %
]

# Correspondance des labels numériques vers leur signification métier
LABEL_NAMES = {
    0: "Normal (Healthy)",
    1: "Maintenance Suggested",
    2: "Immediate Maintenance Required",
}


# ==============================================================================
# ÉTAPE 1 : INGÉNIERIE DES FEATURES (FEATURE ENGINEERING)
# ==============================================================================

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute 5 nouvelles colonnes synthétiques calculées à partir des mesures brutes.
    Cette fonction est appliquée IDENTIQUEMENT sur Train.csv et Test.csv pour garantir
    que le modèle voit exactement les mêmes types de signaux lors de l'entraînement
    et de l'évaluation.
    """

    # Copie défensive du DataFrame pour ne pas modifier l'original en mémoire
    df = df.copy()

    # ── Feature 1 : Déséquilibre thermique entre les deux capteurs de chaleur ──
    # Un écart croissant entre la température du multiplicateur et celle des roulements
    # est un signal précoce de friction mécanique anormale ou de problème de lubrification.
    df["Thermal_Imbalance"] = df["Gearbox_Oil_Temp_C"] - df["Generator_Bearing_Temp_C"]

    # ── Feature 2 : Efficacité de conversion du vent en électricité ──
    # Si le rotor tourne vite (vent disponible) mais que la puissance générée est faible,
    # cela suggère des pertes mécaniques ou électriques à l'intérieur de la machine.
    # On ajoute 0.001 au dénominateur pour éviter une division par zéro si RPM = 0
    df["Power_Efficiency"] = df["Power_Output_kW"] / (df["Rotor_Speed_RPM"] + 0.001)

    # ── Feature 3 : Indice de stress thermique global ──
    # La somme des deux températures critiques capture la surchauffe globale de la machine.
    # Un stress thermique combiné élevé augmente fortement le risque de défaillance.
    df["Thermal_Stress"] = df["Gearbox_Oil_Temp_C"] + df["Generator_Bearing_Temp_C"]

    # ── Feature 4 : Intensité vibratoire rapportée à la vitesse de rotation ──
    # Une vibration normale augmente proportionnellement à la vitesse de rotation.
    # Si la vibration augmente plus vite que la vitesse, c'est un signe de déséquilibre
    # mécanique (roulement usé, pale déformée, désalignement d'arbre...).
    # On ajoute 0.001 pour éviter une division par zéro si RPM = 0
    df["Vibration_per_RPM"] = df["Vibration_Level_mmps"] / (df["Rotor_Speed_RPM"] + 0.001)

    # ── Feature 5 : Indice de stress chaleur-humidité ──
    # La combinaison d'une haute température et d'une haute humidité accélère fortement
    # la corrosion des composants électriques et la dégradation des lubrifiants.
    # On divise par 100 pour normaliser le taux d'humidité entre 0 et 1.
    df["Heat_Humidity_Index"] = df["Gearbox_Oil_Temp_C"] * df["Humidity_pct"] / 100.0

    # Retourne le DataFrame enrichi des 5 nouvelles colonnes synthétiques
    return df


# ==============================================================================
# ÉTAPE 2 : CHARGEMENT ET PRÉPARATION DES DONNÉES
# ==============================================================================

def load_and_prepare(csv_path: str, label: str = "Train") -> tuple:
    """
    Charge un fichier CSV, vérifie son contenu, applique l'ingénierie des features,
    et retourne le tuple (X, y) prêt pour l'entraînement ou l'évaluation.
    """

    # Vérification que le fichier CSV existe bien sur le disque
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"[ERREUR] Le fichier '{csv_path}' est introuvable. "
            f"Vérifiez que le dossier data/ contient bien {csv_path}."
        )

    # Chargement du fichier CSV complet dans un DataFrame pandas
    print(f"\n[{label}] Chargement de '{csv_path}'...")
    df = pd.read_csv(csv_path)

    # Affichage du nombre de lignes et de colonnes chargées
    print(f"[{label}] Dimensions du fichier : {df.shape[0]} lignes × {df.shape[1]} colonnes.")

    # Affichage de la répartition des labels dans le jeu de données (équilibre des classes)
    print(f"[{label}] Répartition des labels (Maintenance_Label) :")
    label_counts = df[TARGET_COLUMN].value_counts().sort_index()
    for lbl_num, count in label_counts.items():
        # Affichage de chaque label avec son nom métier, son effectif et son pourcentage
        pct = count / len(df) * 100
        name = LABEL_NAMES.get(lbl_num, "Inconnu")
        print(f"         Label {lbl_num} ({name}) : {count:6d} lignes ({pct:.1f}%)")

    # Application de l'ingénierie des features (identique pour Train et Test)
    print(f"[{label}] Calcul des 5 nouvelles features synthétiques...")
    df = add_features(df)

    # Construction de la liste complète des features (8 brutes + 5 calculées = 13)
    all_feature_columns = BASE_FEATURE_COLUMNS + [
        "Thermal_Imbalance",
        "Power_Efficiency",
        "Thermal_Stress",
        "Vibration_per_RPM",
        "Heat_Humidity_Index",
    ]

    # Extraction de la matrice de features X (les 13 colonnes d'entrée)
    X = df[all_feature_columns]

    # Extraction du vecteur cible y (la colonne Maintenance_Label à prédire)
    y = df[TARGET_COLUMN]

    # Confirmation du nombre de features utilisées
    print(f"[{label}] Features utilisées ({len(all_feature_columns)}) : {all_feature_columns}")

    # Retour du tuple (X, y) sous forme de DataFrames / Series pandas
    return X, y


# ==============================================================================
# ÉTAPE 3 : ENTRAÎNEMENT DU MODÈLE XGBOOST
# ==============================================================================

def train_xgboost_model(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    """
    Instancie et entraîne un classifieur XGBoost sur la totalité de Train.csv.
    XGBoost construit des centaines d'arbres de décision en cascade et les combine
    pour produire des prédictions robustes et précises sur des données tabulaires.
    """

    print("\n" + "=" * 80)
    print(" ÉTAPE 3 : ENTRAÎNEMENT DU MODÈLE XGBOOST")
    print("=" * 80)

    # Instanciation du classifieur XGBoost avec ses hyperparamètres
    model = XGBClassifier(

        # Nombre d'arbres de décision construits en cascade (plus = plus précis mais plus lent)
        n_estimators=300,

        # Profondeur maximale de chaque arbre (trop profond = sur-apprentissage du bruit)
        max_depth=6,

        # Taux d'apprentissage : à quel rythme chaque nouvel arbre corrige les erreurs du précédent
        # Une valeur faible donne généralement de meilleures performances finales
        learning_rate=0.1,

        # Fraction des colonnes de features choisies aléatoirement pour chaque arbre
        # Introduit de la diversité et prévient le sur-apprentissage
        colsample_bytree=0.8,

        # Fraction des lignes d'entraînement choisies aléatoirement pour chaque arbre
        # Accélère l'entraînement et prévient le sur-apprentissage
        subsample=0.8,

        # Paramètre de régularisation L1 (pénalise les poids excessifs pour simplifier le modèle)
        reg_alpha=0.1,

        # Paramètre de régularisation L2 (pénalise les poids excessifs pour simplifier le modèle)
        reg_lambda=1.0,

        # CORRECTION DU DÉSÉQUILIBRE DES CLASSES
        # Le jeu de données est fortement déséquilibré : 83.5% "Normal" vs 16.5% "Pannes".
        # Sans correction, le modèle apprend à tout prédire comme "Normal" (83% d'accuracy
        # facile, mais 0% de détection de pannes — inutile pour un système d'alerte !).
        # La stratégie "balanced" calcule automatiquement des poids inversement proportionnels
        # à la fréquence de chaque classe : les classes rares (Label 1 et 2) sont ainsi
        # pénalisées plus fortement quand le modèle se trompe sur elles.
        # On simule ce comportement via sample_weight calculé dans le fit().
        # Pour XGBoost multi-classe, on utilise l'objectif "multi:softprob" avec des poids
        # de classe pour équilibrer l'entraînement.
        objective="multi:softmax",

        # Nombre de classes distinctes à prédire (0 = Normal, 1 = Attention, 2 = Urgent)
        num_class=3,

        # Graine du générateur aléatoire pour garantir la reproductibilité des résultats
        random_state=42,

        # Utilisation de tous les processeurs disponibles pour paralléliser l'entraînement
        n_jobs=-1,

        # Niveau de verbosité de XGBoost pendant l'entraînement (0 = silencieux)
        verbosity=0,
    )

    # Affichage du message de départ de l'entraînement
    print(f"[TRAIN] Démarrage de l'entraînement sur {len(X_train)} lignes avec {X_train.shape[1]} features...")

    # CORRECTION DU DÉSÉQUILIBRE DE CLASSES VIA LES POIDS INDIVIDUELS DE LIGNES
    # Le problème : 83.5% des données sont "Normal" (Label 0). Sans correction, le modèle
    # prédit tout comme "Normal" car c'est la réponse facile qui minimise l'erreur globale.
    # La solution : attribuer un poids plus élevé aux classes rares (Label 1 et 2) pour que
    # le modèle soit PLUS FORTEMENT pénalisé quand il se trompe sur une panne !
    #
    # Calcul du poids inversement proportionnel à la fréquence de chaque classe :
    # Poids de la classe i = Nombre total de lignes / (Nombre de classes × Nombre de lignes de la classe i)
    from sklearn.utils.class_weight import compute_sample_weight
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    print(f"[TRAIN] Poids de rééquilibrage appliqués (Label 0 < Label 1, Label 2) pour corriger le déséquilibre.")

    # Entraînement effectif du modèle sur l'intégralité de Train.csv
    # sample_weight passe à chaque ligne un poids individuel : les pannes (Label 1 et 2)
    # ont un poids environ 5x plus élevé que le Label 0 (Normal), forçant le modèle à les détecter
    model.fit(X_train, y_train, sample_weight=sample_weights)

    # Confirmation de la fin de l'entraînement
    print(f"[TRAIN] Entraînement terminé ! {model.n_estimators} arbres de décision construits.")

    # Retourne le modèle entraîné, prêt pour l'évaluation
    return model


# ==============================================================================
# ÉTAPE 4 : ÉVALUATION DU MODÈLE SUR TEST.CSV
# ==============================================================================

def evaluate_model(model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Utilise le modèle entraîné pour prédire les labels sur le jeu de test (Test.csv),
    puis calcule et affiche les métriques de performance complètes.
    """

    print("\n" + "=" * 80)
    print(" ÉTAPE 4 : ÉVALUATION DU MODÈLE SUR TEST.CSV")
    print("=" * 80)

    # Application du modèle entraîné pour prédire les labels de chaque ligne de Test.csv
    # y_pred contiendra des valeurs 0, 1 ou 2 pour chaque ligne de X_test
    y_pred = model.predict(X_test)

    # Calcul de l'accuracy globale : le pourcentage de prédictions exactes
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n[EVAL] Accuracy globale sur Test.csv : {accuracy * 100:.2f}%")

    # Affichage de la matrice de confusion
    # Chaque ligne = la vraie classe, chaque colonne = la classe prédite
    # Les valeurs sur la diagonale = bonnes prédictions
    # Les valeurs hors diagonale = erreurs de classification
    print("\n[EVAL] Matrice de Confusion :")
    print("       (Ligne = Vraie classe | Colonne = Classe prédite)\n")
    cm = confusion_matrix(y_test, y_pred)
    print(f"          Prédit 0   Prédit 1   Prédit 2")
    for i, row in enumerate(cm):
        # Affichage de chaque ligne avec le nom de la vraie classe
        label_name = list(LABEL_NAMES.values())[i]
        print(f"  Vrai {i} : {row[0]:8d}   {row[1]:8d}   {row[2]:8d}   <- {label_name}")

    # Affichage du rapport de classification complet (précision, rappel et F1 par classe)
    print("\n[EVAL] Rapport de Classification Détaillé :")
    print("       (Precision = taux de bonnes alertes | Recall = taux de détection | F1 = équilibre)")
    target_names = [f"Label {k} ({v})" for k, v in LABEL_NAMES.items()]
    report = classification_report(y_test, y_pred, target_names=target_names)
    print(report)


# ==============================================================================
# ÉTAPE 5 : IMPORTANCE DES FEATURES (FEATURE IMPORTANCES)
# ==============================================================================

def display_feature_importances(model: XGBClassifier, feature_names: list):
    """
    Extrait et affiche le classement des features par ordre d'importance.
    XGBoost calcule pour chaque feature combien de fois elle a été utilisée
    par les arbres de décision pour améliorer les prédictions.
    """

    print("\n" + "=" * 80)
    print(" ÉTAPE 5 : IMPORTANCE DES FEATURES (CONTRIBUTION À LA PRÉDICTION)")
    print("=" * 80)

    # Extraction du tableau des importances calculées par XGBoost pour chaque feature
    importances = model.feature_importances_

    # Création d'un DataFrame pandas pour faciliter le tri et l'affichage
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
    })

    # Tri par ordre décroissant d'importance pour voir les plus prédictives en premier
    importance_df = importance_df.sort_values("Importance", ascending=False).reset_index(drop=True)

    # Calcul du pourcentage d'importance relatif de chaque feature
    importance_df["Importance_pct"] = importance_df["Importance"] / importance_df["Importance"].sum() * 100

    # Affichage du tableau des importances avec une barre de progression textuelle
    print(f"\n{'Rang':<5} {'Feature':<35} {'Importance':>12} {'%':>8}  {'Barre visuelle'}")
    print("-" * 90)
    for idx, row in importance_df.iterrows():
        # Calcul de la longueur de la barre proportionnelle à l'importance
        bar_length = int(row["Importance_pct"] / 2)
        bar = "#" * bar_length

        # Affichage de la ligne avec rang, nom de la feature, valeur brute, % et barre
        print(
            f"  {idx + 1:<4} {row['Feature']:<35} {row['Importance']:>12.4f} "
            f"{row['Importance_pct']:>7.1f}%  {bar}"
        )

    # Affichage du message de lecture du graphique
    print("\n[INFO] Lecture : La feature avec la barre la plus longue est la plus déterminante")
    print("       pour distinguer Normal / Maintenance conseillée / Maintenance urgente.")


# ==============================================================================
# ÉTAPE 6 : SAUVEGARDE DU MODÈLE ENTRAÎNÉ
# ==============================================================================

def save_model(model: XGBClassifier, output_path: str):
    """
    Sérialise (transforme en fichier binaire) le modèle XGBoost entraîné avec joblib.
    Le fichier .pkl généré pourra être rechargé instantanément par l'API FastAPI (Phase 5)
    sans avoir besoin de ré-entraîner le modèle à chaque démarrage.
    """

    print("\n" + "=" * 80)
    print(" ÉTAPE 6 : SAUVEGARDE DU MODÈLE ENTRAÎNÉ")
    print("=" * 80)

    # Extraction du dossier de destination (ex: "app/" dans "app/model.pkl")
    output_dir = os.path.dirname(output_path)

    # Création du dossier de destination s'il n'existe pas encore sur le disque
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"[SAVE] Dossier '{output_dir}/' créé automatiquement.")

    # Sauvegarde du modèle sur le disque avec la compression par défaut de joblib
    # joblib est plus efficace que pickle pour les objets NumPy / sklearn / XGBoost
    joblib.dump(model, output_path)

    # Calcul et affichage de la taille du fichier généré en kilo-octets
    size_kb = os.path.getsize(output_path) / 1024.0
    print(f"[SAVE] Modèle sauvegardé avec succès dans '{output_path}' ({size_kb:.1f} KB)")
    print(f"[SAVE] Ce fichier sera rechargé par l'API FastAPI avec :")
    print(f"       model = joblib.load('{output_path}')")


# ==============================================================================
# FONCTION PRINCIPALE D'ORCHESTRATION
# ==============================================================================

def main():
    # Affichage de l'en-tête de démarrage du script
    print("=" * 80)
    print(" PHASE 3 : ENTRAÎNEMENT DU MODÈLE MACHINE LEARNING - SMART ENERGY GRID")
    print("=" * 80)

    # ── Étape 1 & 2 : Chargement, feature engineering et préparation de Train.csv ──
    print("\n[ÉTAPE 1/6] Chargement et préparation de Train.csv (ENTRAÎNEMENT)...")
    X_train, y_train = load_and_prepare(TRAIN_PATH, label="TRAIN")

    # ── Étape 1 & 2 : Chargement, feature engineering et préparation de Test.csv ──
    print("\n[ÉTAPE 2/6] Chargement et préparation de Test.csv (ÉVALUATION FINALE)...")
    X_test, y_test = load_and_prepare(TEST_PATH, label="TEST ")

    # Extraction de la liste des noms de features pour les affichages ultérieurs
    feature_names = list(X_train.columns)

    # ── Étape 3 : Entraînement du modèle XGBoost sur la totalité de Train.csv ──
    print("\n[ÉTAPE 3/6] Entraînement de XGBoost sur Train.csv...")
    model = train_xgboost_model(X_train, y_train)

    # ── Étape 4 : Évaluation honnête sur Test.csv (données jamais vues par le modèle) ──
    print("\n[ÉTAPE 4/6] Évaluation du modèle sur Test.csv...")
    evaluate_model(model, X_test, y_test)

    # ── Étape 5 : Affichage des importances des 13 features ──
    print("\n[ÉTAPE 5/6] Analyse des features les plus importantes...")
    display_feature_importances(model, feature_names)

    # ── Étape 6 : Sauvegarde du cerveau artificiel dans app/model.pkl ──
    print("\n[ÉTAPE 6/6] Sauvegarde du modèle entraîné...")
    save_model(model, MODEL_OUTPUT_PATH)

    # Message de fin de la Phase 3
    print("\n" + "=" * 80)
    print(" PHASE 3 TERMINEE AVEC SUCCES !")
    print(f" Le modele est pret dans '{MODEL_OUTPUT_PATH}'.")
    print(" Prochaine etape : Phase 4 - Feature Store Feast + Redis")
    print("=" * 80)


# Point d'entrée principal pour l'exécution directe du script depuis le terminal
if __name__ == "__main__":
    main()
