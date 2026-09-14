# ==============================================================================
# PROJET SMART ENERGY GRID - PHASE 6 : TESTS DU MODÈLE MACHINE LEARNING
# Fichier : tests/test_model.py
# Description :
#   Suite de tests pytest vérifiant l'intégrité du modèle XGBoost sérialisé
#   dans 'app/model.pkl'.
#
#   Ces tests garantissent que :
#     1. Le fichier de modèle existe et n'est pas corrompu.
#     2. Il peut être chargé correctement en mémoire avec joblib.
#     3. Il produit des inférences valides (classes 0, 1, 2).
#     4. Ses probabilités prédictives somment bien à 100%.
# ==============================================================================

import pathlib
import joblib
import numpy as np
import pandas as pd
import pytest
from app.main import MODEL_PATH, FEATURE_COLUMNS, get_model


def test_model_file_exists():
    """
    Test 1 : Vérifie que le fichier sérialisé 'app/model.pkl' existe
    et que sa taille est supérieure à zéro (fichier non vide).
    """
    assert MODEL_PATH.exists(), f"Le fichier modèle '{MODEL_PATH}' doit exister sur le disque"
    file_size_bytes = MODEL_PATH.stat().st_size
    assert file_size_bytes > 1000, f"Le modèle semble corrompu ou trop petit ({file_size_bytes} octets)"


def test_model_loading_and_instance():
    """
    Test 2 : Vérifie que le modèle peut être désérialisé par joblib
    et possède bien les méthodes fondamentales scikit-learn/xgboost
    'predict' et 'predict_proba'.
    """
    model = get_model()
    assert model is not None, "La fonction get_model() ne doit pas renvoyer None"
    assert hasattr(model, "predict"), "Le modèle doit implémenter la méthode 'predict'"
    assert hasattr(model, "predict_proba"), "Le modèle doit implémenter la méthode 'predict_proba'"


def test_model_inference_pipeline():
    """
    Test 3 : Inférence sur un échantillon synthétique de 13 features.
    Vérifie la forme et la cohérence de la sortie.
    """
    model = get_model()

    # Création d'une ligne synthétique avec exactement les 13 features ordonnées
    synthetic_input = pd.DataFrame([{
        "Rotor_Speed_RPM": 15.0,
        "Wind_Speed_mps": 8.5,
        "Power_Output_kW": 1200.0,
        "Gearbox_Oil_Temp_C": 68.0,
        "Generator_Bearing_Temp_C": 62.0,
        "Vibration_Level_mmps": 2.1,
        "Ambient_Temp_C": 18.0,
        "Humidity_pct": 55.0,
        "Thermal_Imbalance": 6.0,
        "Power_Efficiency": 80.0,
        "Thermal_Stress": 50.0,
        "Vibration_per_RPM": 0.14,
        "Heat_Humidity_Index": 9.9,
    }])[FEATURE_COLUMNS]

    # 1. Test de la prédiction de classe
    pred = model.predict(synthetic_input)
    assert len(pred) == 1, "La prédiction doit renvoyer un élément unique pour 1 ligne"
    predicted_label = int(pred[0])
    assert predicted_label in [0, 1, 2], f"Le label prédit ({predicted_label}) doit être 0, 1 ou 2"

    # 2. Test des probabilités
    probas = model.predict_proba(synthetic_input)
    assert probas.shape == (1, 3), f"La forme des probabilités doit être (1, 3), obtenu {probas.shape}"

    # Vérification que la somme des probabilités vaut exactement 1.0 (à 1e-5 près)
    sum_prob = float(np.sum(probas[0]))
    assert abs(sum_prob - 1.0) < 1e-4, f"La somme des probabilités ({sum_prob}) doit être égale à 1.0"
