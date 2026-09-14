# ==============================================================================
# PROJET SMART ENERGY GRID - PHASE 6 : TESTS UNITAIRES DE LA LOGIQUE DE DÉCISION
# Fichier : tests/test_decision.py
# Description :
#   Suite de tests pytest vérifiant la robustesse de la logique probabiliste
#   centralisée dans 'app/decision.py'.
#
#   Ces tests garantissent que les seuils métier définis avec l'opérateur
#   (20% urgence, 25% maintenance suggérée, 35% risque cumulé) sont toujours
#   respectés même si d'autres parties du code évoluent.
# ==============================================================================

import pytest
# Importation de la fonction de décision et de la classe de résultat
from app.decision import (
    compute_smart_decision,
    THRESHOLD_URGENT,
    THRESHOLD_SUGGESTED,
    THRESHOLD_CUMULATED,
    DecisionResult,
)


def test_constants_thresholds():
    """
    Test 1 : Vérifie que les constantes de seuil métier correspondent
    exactement aux spécifications validées lors de la Phase 5.
    """
    # Le seuil d'urgence doit être fixé à 20% (0.20)
    assert THRESHOLD_URGENT == 0.20, "Le seuil d'urgence doit être de 0.20"
    # Le seuil de maintenance suggérée doit être fixé à 25% (0.25)
    assert THRESHOLD_SUGGESTED == 0.25, "Le seuil conseillé doit être de 0.25"
    # Le seuil de risque cumulé doit être fixé à 35% (0.35)
    assert THRESHOLD_CUMULATED == 0.35, "Le seuil cumulé doit être de 0.35"


def test_nominal_healthy_turbine():
    """
    Test 2 : Cas nominal où l'éolienne est en parfaite santé.
    Probabilité normale très élevée (ex: 90%), aucun seuil dépassé.
    Le système doit déclarer l'éolienne comme Normale (Label 0).
    """
    # Entrée : 90% normal, 6% conseillé, 4% urgent
    result = compute_smart_decision(prob_normal=0.90, prob_suggested=0.06, prob_urgent=0.04)

    # Vérifications
    assert isinstance(result, DecisionResult), "Le retour doit être une instance de DecisionResult"
    assert result.label == 0, "Le label doit être 0 (Normal)"
    assert result.status == "Normal (Healthy)", "Le statut textuel doit être 'Normal (Healthy)'"
    assert result.icon == "🟢", "L'icône doit être la pastille verte"
    # Risque cumulé = 6% + 4% = 10%
    assert abs(result.cumulated_risk_pct - 10.0) < 1e-4, "Le risque cumulé doit être de 10.0%"


def test_urgent_threshold_trigger():
    """
    Test 3 : Déclenchement de l'alerte Urgente.
    Même si la probabilité normale est dominante (ex: 75%),
    si la probabilité d'urgence atteint ou dépasse 20%,
    le système DOIT classer l'événement en Urgence (Label 2).
    """
    # Entrée : 75% normal, 4% conseillé, 21% urgent (>= 20%)
    result = compute_smart_decision(prob_normal=0.75, prob_suggested=0.04, prob_urgent=0.21)

    assert result.label == 2, "Le label doit basculer en 2 (Urgence)"
    assert result.icon == "🔴", "L'icône doit être rouge"
    assert "urgence" in result.rule_triggered.lower(), "La règle doit mentionner l'urgence"


def test_suggested_threshold_trigger():
    """
    Test 4 : Déclenchement de l'alerte Maintenance Suggérée.
    Si la probabilité d'urgence est faible (< 20%),
    mais que la probabilité conseillée atteint ou dépasse 25%,
    le système DOIT déclarer une Maintenance Conseillée (Label 1).
    """
    # Entrée : 70% normal, 26% conseillé (>= 25%), 4% urgent
    result = compute_smart_decision(prob_normal=0.70, prob_suggested=0.26, prob_urgent=0.04)

    assert result.label == 1, "Le label doit basculer en 1 (Maintenance conseillée)"
    assert result.icon == "🟡", "L'icône doit être jaune"
    assert "maintenance conseill" in result.rule_triggered.lower()


def test_cumulated_risk_overrides_xgboost_normal():
    """
    Test 5 : Cas critique découvert sur la Turbine 1 !
    XGBoost prédit "Normal" car Normal est la classe majoritaire (60.26%).
    Mais urgence (18.18%) + conseillée (21.56%) = 39.74% >= 35%.
    Le système DOIT corriger XGBoost et déclencher la surveillance (Label 1).
    """
    # Probabilités réelles mesurées sur la Turbine 1
    result = compute_smart_decision(
        prob_normal=0.6026,
        prob_suggested=0.2156,
        prob_urgent=0.1818,
    )

    # Le statut ne doit PAS rester Normal
    assert result.label == 1, "Le risque cumulé >= 35% doit forcer le label 1"
    assert result.icon == "🟡", "L'icône doit être jaune pour alerter l'opérateur"
    assert "Risque cumulé anormal" in result.rule_triggered
    assert "XGBoost disait Normal" in result.rule_triggered


def test_priority_urgent_over_suggested():
    """
    Test 6 : Priorité de sécurité.
    Si les DEUX seuils sont dépassés simultanément (ex: 22% urgent et 28% conseillé),
    l'alerte d'URGENCE doit impérativement prévaloir sur la simple suggestion.
    """
    # Entrée : 50% normal, 28% conseillé, 22% urgent
    result = compute_smart_decision(prob_normal=0.50, prob_suggested=0.28, prob_urgent=0.22)

    assert result.label == 2, "L'urgence doit être prioritaire sur la suggestion"
    assert result.icon == "🔴"


def test_invalid_probabilities_raise_error():
    """
    Test 7 : Vérification des cas d'erreur de saisie (entrées corrompues).
    Une probabilité négative ou une somme incohérente doit lever une ValueError.
    """
    # Cas d'une probabilité négative
    with pytest.raises(ValueError, match="doivent être comprises entre 0 et 1"):
        compute_smart_decision(prob_normal=-0.1, prob_suggested=0.6, prob_urgent=0.5)

    # Cas d'une somme de probabilités aberrante (ex: 200%)
    with pytest.raises(ValueError, match="doit être égale à 1.0"):
        compute_smart_decision(prob_normal=0.8, prob_suggested=0.8, prob_urgent=0.8)
