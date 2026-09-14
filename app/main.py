# ==============================================================================
# PROJET SMART ENERGY GRID - PHASE 5 : API REST FASTAPI
# Fichier : app/main.py
# Description :
#   Serveur API REST qui expose les capacités prédictives du projet via HTTP.
#   Trois routes disponibles :
#     1. GET  /health             → Vérification de l'état de santé du serveur
#     2. GET  /turbine/{id}       → Diagnostic automatique via Feast + Redis + XGBoost
#     3. POST /predict            → Prédiction manuelle à partir de données saisies
#
#   Chaque route utilise la logique de décision probabiliste (app/decision.py) :
#     → Si prob_urgente >= 20%          → Alerte Urgente
#     → Si prob_conseillée >= 25%       → Surveillance Requise
#     → Si risque cumulé >= 35%         → Surveillance Requise (Risque Cumulé)
#
#   Démarrage du serveur :
#     uvicorn app.main:app --reload --port 8000
# ==============================================================================

# Importation des modules fondamentaux de FastAPI
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Importation de Pydantic pour la validation et la sérialisation des données entrantes/sortantes
from pydantic import BaseModel, Field

# Importation du module standard Python pour les chemins de fichiers
from pathlib import Path

# Importation du module de logging pour tracer les événements
import logging

# Importation de numpy pour la gestion des types numériques
import numpy as np

# Importation de pandas pour construire les DataFrames de features
import pandas as pd

# Importation de joblib pour charger le modèle ML sérialisé depuis le disque
import joblib

# Importation de Feast pour accéder à l'Online Store Redis
from feast import FeatureStore

# Importation de notre module de décision probabiliste centralisé
from app.decision import compute_smart_decision, DecisionResult


# ==============================================================================
# CONFIGURATION DU LOGGING
# ==============================================================================

# Initialisation du logger de l'application (pour tracer les requêtes et les erreurs)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("smart_energy_grid")


# ==============================================================================
# CONSTANTES ET CHEMINS DE FICHIERS
# ==============================================================================

# Chemin vers le modèle XGBoost sérialisé, sauvegardé en Phase 3
MODEL_PATH = Path("app/model.pkl")

# Chemin vers le dossier du Feature Store Feast, configuré en Phase 4
FEAST_REPO_PATH = Path("feature_repository")

# Ordre exact des 13 colonnes de features attendu par le modèle XGBoost
# (Doit correspondre exactement à l'ordre utilisé pendant l'entraînement en Phase 3)
FEATURE_COLUMNS = [
    "Rotor_Speed_RPM",
    "Wind_Speed_mps",
    "Power_Output_kW",
    "Gearbox_Oil_Temp_C",
    "Generator_Bearing_Temp_C",
    "Vibration_Level_mmps",
    "Ambient_Temp_C",
    "Humidity_pct",
    "Thermal_Imbalance",
    "Power_Efficiency",
    "Thermal_Stress",
    "Vibration_per_RPM",
    "Heat_Humidity_Index",
]

# Noms des features attendues depuis Feast
# Syntaxe : "<nom_feature_view>:<nom_feature>"
FEAST_FEATURES = [f"turbine_features:{col}" for col in FEATURE_COLUMNS]


# ==============================================================================
# INITIALISATION DE L'APPLICATION FASTAPI
# ==============================================================================

# Création de l'instance principale de l'application FastAPI
app = FastAPI(
    title="Smart Energy Grid — API de Maintenance Prédictive",
    description=(
        "API REST pour la surveillance prédictive des éoliennes. "
        "Utilise un modèle XGBoost entraîné sur des données de capteurs physiques, "
        "avec un Feature Store Feast alimenté en temps réel depuis Redis."
    ),
    version="1.0.0",
)

# Ajout du middleware CORS pour autoriser le Dashboard Streamlit à interroger l'API
# (Streamlit tourne sur un port différent, sans CORS le navigateur bloquerait les requêtes)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # En production, remplacer par l'URL exacte du dashboard
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# CHARGEMENT DES RESSOURCES AU DÉMARRAGE (UNE SEULE FOIS EN MÉMOIRE)
# ==============================================================================

# Variable globale pour stocker le modèle chargé
# (chargé une seule fois au démarrage → évite de relire le fichier à chaque requête)
_model = None

# Variable globale pour le client Feast
_feast_store = None


def get_model():
    """
    Charge le modèle XGBoost depuis le disque lors du premier appel.
    Les appels suivants retournent la version en cache mémoire.
    """
    global _model
    if _model is None:
        # Vérification que le fichier du modèle existe
        if not MODEL_PATH.exists():
            raise RuntimeError(f"Modèle introuvable : '{MODEL_PATH}'. Exécutez d'abord 'python scripts/train_model.py'.")
        # Chargement du modèle sérialisé
        _model = joblib.load(MODEL_PATH)
        logger.info(f"Modèle XGBoost chargé depuis '{MODEL_PATH}'.")
    return _model


def get_feast_store():
    """
    Initialise le client FeatureStore Feast lors du premier appel.
    Les appels suivants retournent l'instance en cache.
    """
    global _feast_store
    if _feast_store is None:
        # Vérification que le dossier Feast existe
        if not FEAST_REPO_PATH.exists():
            raise RuntimeError(f"Dossier Feast introuvable : '{FEAST_REPO_PATH}'.")
        # Initialisation du client Feast connecté à Redis
        _feast_store = FeatureStore(repo_path=str(FEAST_REPO_PATH))
        logger.info(f"Client Feast connecté depuis '{FEAST_REPO_PATH}'.")
    return _feast_store


# ==============================================================================
# FONCTIONS UTILITAIRES DE FEATURE ENGINEERING
# ==============================================================================

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les 5 features dérivées à partir des 8 capteurs bruts.
    Fonction identique à celle de Phase 3 et Phase 4 → garantit la cohérence.
    """
    data = df.copy()
    # Écart de température entre huile du multiplicateur et roulement de la génératrice
    data["Thermal_Imbalance"] = data["Gearbox_Oil_Temp_C"] - data["Generator_Bearing_Temp_C"]
    # Ratio puissance / vitesse de rotation
    data["Power_Efficiency"] = data["Power_Output_kW"] / (data["Rotor_Speed_RPM"] + 1e-5)
    # Écart de température huile vs air ambiant
    data["Thermal_Stress"] = data["Gearbox_Oil_Temp_C"] - data["Ambient_Temp_C"]
    # Ratio vibration par tour mécanique
    data["Vibration_per_RPM"] = data["Vibration_Level_mmps"] / (data["Rotor_Speed_RPM"] + 1e-5)
    # Indice combiné chaleur/humidité
    data["Heat_Humidity_Index"] = data["Ambient_Temp_C"] * (data["Humidity_pct"] / 100.0)
    return data


# ==============================================================================
# MODÈLES DE DONNÉES (SCHÉMAS PYDANTIC)
# ==============================================================================

class ManualSensorInput(BaseModel):
    """
    Schéma de validation pour les données d'entrée de la route POST /predict.
    Le technicien fournit les 8 mesures brutes des capteurs IoT.
    FastAPI valide automatiquement les types et les plages de valeurs.
    """
    Rotor_Speed_RPM: float = Field(..., ge=0, le=50, description="Vitesse de rotation des pales (RPM)")
    Wind_Speed_mps: float = Field(..., ge=0, le=40, description="Vitesse du vent (m/s)")
    Power_Output_kW: float = Field(..., ge=0, le=3000, description="Puissance générée (kW)")
    Gearbox_Oil_Temp_C: float = Field(..., ge=0, le=150, description="Température huile multiplicateur (°C)")
    Generator_Bearing_Temp_C: float = Field(..., ge=0, le=150, description="Température roulements génératrice (°C)")
    Vibration_Level_mmps: float = Field(..., ge=0, le=20, description="Niveau de vibration (mm/s)")
    Ambient_Temp_C: float = Field(..., ge=-30, le=60, description="Température ambiante (°C)")
    Humidity_pct: float = Field(..., ge=0, le=100, description="Humidité relative (%)")

    model_config = {
        # Exemple de données affiché dans la documentation Swagger automatique
        "json_schema_extra": {
            "example": {
                "Rotor_Speed_RPM": 22.0,
                "Wind_Speed_mps": 12.5,
                "Power_Output_kW": 1800.0,
                "Gearbox_Oil_Temp_C": 95.0,
                "Generator_Bearing_Temp_C": 88.0,
                "Vibration_Level_mmps": 4.2,
                "Ambient_Temp_C": 15.0,
                "Humidity_pct": 80.0,
            }
        }
    }


class PredictionResponse(BaseModel):
    """
    Schéma de la réponse retournée par les routes de prédiction.
    Toutes les informations utiles au technicien et au dashboard.
    """
    # Identifiant de la turbine concernée (None si prédiction manuelle)
    turbine_id: int | None

    # Label numérique de la décision finale (0, 1 ou 2)
    predicted_label: int

    # Statut textuel lisible
    status: str

    # Emoji d'alerte correspondant au statut
    icon: str

    # Description de la règle de décision qui a été déclenchée (transparence)
    rule_triggered: str

    # Dictionnaire des probabilités brutes de XGBoost
    probabilities: dict

    # Risque anormal cumulé en pourcentage (somme prob_urgente + prob_conseillée)
    cumulated_risk_pct: float

    # Valeurs des 13 features qui ont alimenté le modèle
    features_used: dict


# ==============================================================================
# ROUTE 1 : SANTÉ DU SERVEUR
# ==============================================================================

@app.get("/health", summary="Vérification de l'état du serveur")
def health_check():
    """
    Route de diagnostic rapide.
    Vérifie que l'API tourne, que le modèle est chargé et que Feast est accessible.
    Utile pour les systèmes de monitoring (Kubernetes liveness probe, etc.).
    """
    status = {"api": "ok", "model": "not_loaded", "feast": "not_loaded"}

    # Vérification du modèle
    try:
        get_model()
        status["model"] = "ok"
    except Exception as e:
        status["model"] = f"error: {e}"

    # Vérification de Feast
    try:
        get_feast_store()
        status["feast"] = "ok"
    except Exception as e:
        status["feast"] = f"error: {e}"

    return status


# ==============================================================================
# ROUTE 2 : DIAGNOSTIC AUTOMATIQUE PAR TURBINE (GET /turbine/{id})
# ==============================================================================

@app.get(
    "/turbine/{turbine_id}",
    response_model=PredictionResponse,
    summary="Diagnostic automatique d'une éolienne via Feast + Redis",
)
def predict_turbine(turbine_id: int):
    """
    Interroge Feast pour obtenir les dernières features de l'éolienne demandée
    (stockées en mémoire vive dans Redis), puis applique le modèle XGBoost
    et la logique de décision probabiliste pour produire le diagnostic final.

    Paramètre :
        turbine_id : Identifiant de l'éolienne (ex: 1, 2)
    """
    logger.info(f"[GET /turbine/{turbine_id}] Requête reçue.")

    # Chargement du modèle et du client Feast (depuis le cache en mémoire)
    model = get_model()
    store = get_feast_store()

    # Interrogation de Redis via Feast : récupération des 13 features de la turbine
    try:
        response = store.get_online_features(
            features=FEAST_FEATURES,
            entity_rows=[{"Turbine_ID": turbine_id}],
        )
        features_dict = response.to_dict()
    except Exception as e:
        logger.error(f"Erreur Feast pour turbine {turbine_id}: {e}")
        raise HTTPException(status_code=503, detail=f"Erreur de connexion Feast/Redis : {e}")

    # Vérification que les données ont bien été trouvées dans Redis
    if not features_dict or features_dict.get("Rotor_Speed_RPM", [None])[0] is None:
        raise HTTPException(
            status_code=404,
            detail=f"Turbine ID={turbine_id} introuvable dans le Feature Store. Vérifiez que 'feast materialize' a été exécuté.",
        )

    # Construction du DataFrame d'une ligne avec les 13 features dans le bon ordre
    features_df = pd.DataFrame(features_dict)[FEATURE_COLUMNS]

    # Prédiction XGBoost : obtention des probabilités pour chaque classe
    proba_array = model.predict_proba(features_df)[0]
    prob_normal    = float(proba_array[0])
    prob_suggested = float(proba_array[1])
    prob_urgent    = float(proba_array[2])

    # Application de la logique de décision probabiliste (le cœur de votre remarque !)
    decision: DecisionResult = compute_smart_decision(prob_normal, prob_suggested, prob_urgent)

    logger.info(f"[GET /turbine/{turbine_id}] Décision : {decision.icon} {decision.status} | Règle : {decision.rule_triggered}")

    # Construction et renvoi de la réponse complète
    return PredictionResponse(
        turbine_id=turbine_id,
        predicted_label=decision.label,
        status=decision.status,
        icon=decision.icon,
        rule_triggered=decision.rule_triggered,
        probabilities=decision.probabilities,
        cumulated_risk_pct=decision.cumulated_risk_pct,
        features_used=features_df.iloc[0].to_dict(),
    )


# ==============================================================================
# ROUTE 3 : PRÉDICTION MANUELLE (POST /predict)
# ==============================================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Prédiction manuelle à partir de mesures de capteurs saisies",
)
def predict_manual(data: ManualSensorInput):
    """
    Reçoit les 8 mesures brutes des capteurs IoT via le corps de la requête JSON,
    calcule automatiquement les 5 features d'ingénierie physique,
    puis applique le modèle XGBoost et la logique de décision probabiliste.

    Utile pour les techniciens qui souhaitent tester des scénarios hypothétiques
    ou pour les appareils qui n'ont pas d'entrée dans le Feature Store.
    """
    logger.info(f"[POST /predict] Prédiction manuelle reçue pour RPM={data.Rotor_Speed_RPM}, Puissance={data.Power_Output_kW} kW.")

    # Chargement du modèle depuis le cache mémoire
    model = get_model()

    # Conversion des données d'entrée en DataFrame pandas (une seule ligne)
    raw_df = pd.DataFrame([data.model_dump()])

    # Calcul des 5 features dérivées (Feature Engineering, identique à Phase 3 & 4)
    enriched_df = add_engineered_features(raw_df)

    # Sélection des 13 colonnes dans le bon ordre pour XGBoost
    features_df = enriched_df[FEATURE_COLUMNS]

    # Prédiction XGBoost : obtention des probabilités pour chaque classe
    proba_array = model.predict_proba(features_df)[0]
    prob_normal    = float(proba_array[0])
    prob_suggested = float(proba_array[1])
    prob_urgent    = float(proba_array[2])

    # Application de la logique de décision probabiliste
    decision: DecisionResult = compute_smart_decision(prob_normal, prob_suggested, prob_urgent)

    logger.info(f"[POST /predict] Décision : {decision.icon} {decision.status} | Règle : {decision.rule_triggered}")

    # Construction et renvoi de la réponse complète
    return PredictionResponse(
        turbine_id=None,
        predicted_label=decision.label,
        status=decision.status,
        icon=decision.icon,
        rule_triggered=decision.rule_triggered,
        probabilities=decision.probabilities,
        cumulated_risk_pct=decision.cumulated_risk_pct,
        features_used=features_df.iloc[0].to_dict(),
    )
