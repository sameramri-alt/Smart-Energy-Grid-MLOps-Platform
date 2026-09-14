# ==============================================================================
# PROJET SMART ENERGY GRID - PHASE 6 : TESTS DU FEATURE STORE FEAST
# Fichier : tests/test_feast_definitions.py
# Description :
#   Suite de tests pytest validant la déclaration du référentiel Feast :
#     1. L'entité métier unique (Turbine_ID).
#     2. Le schéma de la Feature View (exactement les 13 colonnes attendues).
#     3. La concordance parfaite avec les features du modèle XGBoost.
# ==============================================================================

from feature_repository.definitions import turbine_entity, turbine_features_view
from app.main import FEATURE_COLUMNS


def test_feast_entity_definition():
    """
    Test 1 : Vérifie que l'entité Feast est bien typée et indexée sur Turbine_ID.
    """
    assert turbine_entity.name == "turbine_id", "Le nom de l'entité doit être 'turbine_id'"
    assert turbine_entity.join_key == "Turbine_ID", "La clé de jointure doit être 'Turbine_ID'"


def test_feast_feature_view_schema():
    """
    Test 2 : Vérifie que la Feature View Feast déclare exactement les 13 features
    requises par le modèle XGBoost, avec activation du stockage en ligne (Redis).
    """
    assert turbine_features_view.name == "turbine_features", "Le nom de la Feature View doit être 'turbine_features'"
    assert turbine_features_view.online is True, "Le mode en ligne (Redis) doit être activé"

    # Extraction des noms de champs déclarés dans le schéma Feast
    declared_feature_names = [f.name for f in turbine_features_view.schema]
    assert len(declared_feature_names) == 13, f"La Feature View doit comporter 13 features, trouvé {len(declared_feature_names)}"

    # Vérification que toutes les 13 colonnes du modèle sont bien couvertes par Feast
    for feat in FEATURE_COLUMNS:
        assert feat in declared_feature_names, f"La feature '{feat}' manque dans le schéma Feast"
