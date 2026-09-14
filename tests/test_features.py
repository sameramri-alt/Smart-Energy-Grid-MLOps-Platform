# ==============================================================================
# PROJET SMART ENERGY GRID - PHASE 6 : TESTS DE L'INGÉNIERIE DES FEATURES
# Fichier : tests/test_features.py
# Description :
#   Suite de tests pytest validant le calcul mathématique des 5 variables
#   physiques dérivées ajoutées aux 8 capteurs bruts (soit 13 features au total).
#
#   Ces tests vérifient :
#     1. L'exactitude des calculs selon les formules physiques.
#     2. La présence de toutes les colonnes requises par le modèle XGBoost.
#     3. La robustesse aux valeurs limites (évitement de division par zéro si RPM = 0).
# ==============================================================================

import pandas as pd
import numpy as np
import pytest
from app.main import add_engineered_features, FEATURE_COLUMNS


@pytest.fixture
def sample_raw_sensor_data() -> pd.DataFrame:
    """
    Fixture pytest fournissant un DataFrame avec les 8 capteurs physiques bruts.
    Valeurs représentatives d'une turbine en fonctionnement normal.
    """
    return pd.DataFrame([{
        "Rotor_Speed_RPM": 20.0,
        "Wind_Speed_mps": 10.0,
        "Power_Output_kW": 1500.0,
        "Gearbox_Oil_Temp_C": 70.0,
        "Generator_Bearing_Temp_C": 65.0,
        "Vibration_Level_mmps": 2.5,
        "Ambient_Temp_C": 20.0,
        "Humidity_pct": 50.0,
    }])


def test_engineered_features_columns_presence(sample_raw_sensor_data):
    """
    Test 1 : Vérifie que les 5 nouvelles colonnes sont bien créées
    et que les 13 colonnes attendues par XGBoost sont toutes présentes.
    """
    df_transformed = add_engineered_features(sample_raw_sensor_data)

    # Vérification des 5 features ajoutées
    expected_new_cols = [
        "Thermal_Imbalance",
        "Power_Efficiency",
        "Thermal_Stress",
        "Vibration_per_RPM",
        "Heat_Humidity_Index",
    ]
    for col in expected_new_cols:
        assert col in df_transformed.columns, f"La colonne dérivée '{col}' doit être présente"

    # Vérification que toutes les 13 features du modèle sont bien couvertes
    for col in FEATURE_COLUMNS:
        assert col in df_transformed.columns, f"La colonne requise '{col}' manque pour XGBoost"


def test_mathematical_formulas_accuracy(sample_raw_sensor_data):
    """
    Test 2 : Vérifie point par point l'exactitude mathématique des 5 formules.
    """
    df_transformed = add_engineered_features(sample_raw_sensor_data)
    row = df_transformed.iloc[0]

    # 1. Thermal_Imbalance = Gearbox_Oil_Temp_C - Generator_Bearing_Temp_C = 70.0 - 65.0 = 5.0
    assert abs(row["Thermal_Imbalance"] - 5.0) < 1e-4

    # 2. Power_Efficiency = Power_Output_kW / (Rotor_Speed_RPM + 1e-5) = 1500.0 / 20.00001 = 75.0
    assert abs(row["Power_Efficiency"] - 75.0) < 1e-2

    # 3. Thermal_Stress = Gearbox_Oil_Temp_C - Ambient_Temp_C = 70.0 - 20.0 = 50.0
    assert abs(row["Thermal_Stress"] - 50.0) < 1e-4

    # 4. Vibration_per_RPM = Vibration_Level_mmps / (Rotor_Speed_RPM + 1e-5) = 2.5 / 20.0 = 0.125
    assert abs(row["Vibration_per_RPM"] - 0.125) < 1e-4

    # 5. Heat_Humidity_Index = Ambient_Temp_C * (Humidity_pct / 100.0) = 20.0 * 0.50 = 10.0
    assert abs(row["Heat_Humidity_Index"] - 10.0) < 1e-4


def test_zero_rpm_division_by_zero_safety():
    """
    Test 3 : Cas limite critique — Éolienne à l'arrêt (Rotor_Speed_RPM = 0.0).
    Vérifie que la constante epsilon (1e-5) empêche toute exception ZeroDivisionError
    ou génération de valeurs 'NaN' ou 'Infini'.
    """
    df_stopped_turbine = pd.DataFrame([{
        "Rotor_Speed_RPM": 0.0,  # Turbine immobile !
        "Wind_Speed_mps": 3.0,
        "Power_Output_kW": 0.0,
        "Gearbox_Oil_Temp_C": 25.0,
        "Generator_Bearing_Temp_C": 25.0,
        "Vibration_Level_mmps": 0.1,
        "Ambient_Temp_C": 18.0,
        "Humidity_pct": 60.0,
    }])

    # Le calcul ne doit pas planter
    df_result = add_engineered_features(df_stopped_turbine)

    # Aucune valeur infinie ni NaN ne doit être présente
    assert not np.isnan(df_result["Power_Efficiency"].iloc[0]), "Power_Efficiency ne doit pas être NaN"
    assert not np.isinf(df_result["Power_Efficiency"].iloc[0]), "Power_Efficiency ne doit pas être Infini"
    assert not np.isnan(df_result["Vibration_per_RPM"].iloc[0]), "Vibration_per_RPM ne doit pas être NaN"
    assert not np.isinf(df_result["Vibration_per_RPM"].iloc[0]), "Vibration_per_RPM ne doit pas être Infini"
