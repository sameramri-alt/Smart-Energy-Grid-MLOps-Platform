# ==============================================================================
# PROJET SMART ENERGY GRID - PHASE 6 : TESTS DE L'API FASTAPI & DES SCHÉMAS
# Fichier : tests/test_api_schemas.py
# Description :
#   Suite de tests pytest validant :
#     1. L'instanciation propre de l'application FastAPI.
#     2. La présence et la configuration des routes requises (/health, /turbine, /predict).
#     3. La validation Pydantic des entrées de données IoT (ManualSensorInput).
#     4. Le rejet automatique des données invalides ou aberrantes.
# ==============================================================================

import pytest
from pydantic import ValidationError
from app.main import app, ManualSensorInput, PredictionResponse


def test_fastapi_app_and_routes_registry():
    """
    Test 1 : Vérifie que l'application FastAPI est bien initialisée
    et expose bien les routes essentielles du projet.
    """
    assert app is not None, "L'objet FastAPI 'app' doit être instancié"

    # Extraction des chemins d'accès (routes) enregistrés
    route_paths = [route.path for route in app.routes]

    # Vérification des endpoints obligatoires
    assert "/health" in route_paths, "La route '/health' doit exister"
    assert "/turbine/{turbine_id}" in route_paths, "La route '/turbine/{turbine_id}' doit exister"
    assert "/predict" in route_paths, "La route '/predict' doit exister"


def test_manual_sensor_input_valid_data():
    """
    Test 2 : Vérifie qu'une saisie de mesures de capteurs valides
    est correctement acceptée par le modèle Pydantic.
    """
    valid_payload = {
        "Rotor_Speed_RPM": 18.5,
        "Wind_Speed_mps": 11.2,
        "Power_Output_kW": 1650.0,
        "Gearbox_Oil_Temp_C": 72.0,
        "Generator_Bearing_Temp_C": 67.5,
        "Vibration_Level_mmps": 2.8,
        "Ambient_Temp_C": 16.0,
        "Humidity_pct": 65.0,
    }

    # Ne doit pas lever d'exception
    item = ManualSensorInput(**valid_payload)
    assert item.Rotor_Speed_RPM == 18.5
    assert item.Power_Output_kW == 1650.0


def test_manual_sensor_input_rejection_negative_rpm():
    """
    Test 3 : Contrôle de sécurité — Vitesse de rotation négative.
    Pydantic doit rejeter automatiquement la requête avec une ValidationError.
    """
    invalid_payload = {
        "Rotor_Speed_RPM": -5.0,  # Valeur impossible !
        "Wind_Speed_mps": 10.0,
        "Power_Output_kW": 1000.0,
        "Gearbox_Oil_Temp_C": 60.0,
        "Generator_Bearing_Temp_C": 55.0,
        "Vibration_Level_mmps": 2.0,
        "Ambient_Temp_C": 20.0,
        "Humidity_pct": 50.0,
    }

    with pytest.raises(ValidationError):
        ManualSensorInput(**invalid_payload)


def test_manual_sensor_input_rejection_excessive_wind():
    """
    Test 4 : Contrôle de sécurité — Vitesse du vent au-delà des limites physiques (> 40 m/s).
    """
    excessive_payload = {
        "Rotor_Speed_RPM": 20.0,
        "Wind_Speed_mps": 85.0,  # Valeur aberrante (ouragan catégorie 5 > 40 m/s)
        "Power_Output_kW": 1000.0,
        "Gearbox_Oil_Temp_C": 60.0,
        "Generator_Bearing_Temp_C": 55.0,
        "Vibration_Level_mmps": 2.0,
        "Ambient_Temp_C": 20.0,
        "Humidity_pct": 50.0,
    }

    with pytest.raises(ValidationError):
        ManualSensorInput(**excessive_payload)
