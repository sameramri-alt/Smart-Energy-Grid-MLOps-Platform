# ==============================================================================
# PROJET SMART ENERGY GRID - PHASE 5 : LOGIQUE DE DÉCISION PROBABILISTE
# Fichier : app/decision.py
# Description :
#   Ce module centralise la logique de classification "intelligente" qui combine :
#   1. La prédiction brute du modèle XGBoost (argmax des probabilités)
#   2. Des seuils probabilistes configurables pour détecter les cas ANORMAUX
#      même quand XGBoost dit "Normal" — compensant ainsi son manque de sensibilité
#      sur les classes minoritaires (pannes).
#
#   Logique de cascade (par ordre de priorité) :
#     Règle 1 : prob_urgent    >= THRESHOLD_URGENT    → Alerte Urgente
#     Règle 2 : prob_suggested >= THRESHOLD_SUGGESTED → Surveillance Requise
#     Règle 3 : (urgent + suggested) >= THRESHOLD_CUMULATED → Surveillance (Risque Cumulé)
#     Règle 4 : Tout sous les seuils → Normal Confirmé
# ==============================================================================

from dataclasses import dataclass


# ==============================================================================
# SEUILS DE DÉCLENCHEMENT DES ALERTES (configurables)
# ==============================================================================

# Seuil unique d'alerte urgente : si la probabilité de panne urgente (Label 2)
# dépasse ce pourcentage, on déclare une alerte urgente MALGRÉ la prédiction brute
THRESHOLD_URGENT: float = 0.20       # 20%

# Seuil de maintenance conseillée : si la probabilité de Label 1 dépasse ce seuil
THRESHOLD_SUGGESTED: float = 0.25    # 25%

# Seuil de risque cumulé : si la somme (Label 1 + Label 2) dépasse ce seuil,
# la turbine est considérée comme anormale même si Label 0 est dominant
THRESHOLD_CUMULATED: float = 0.35    # 35%


# ==============================================================================
# STRUCTURE DE RÉSULTAT
# ==============================================================================

@dataclass
class DecisionResult:
    """
    Conteneur structuré représentant la décision finale pour une éolienne.
    Encapsule à la fois le label numérique, le statut lisible, l'emoji d'alerte,
    la règle qui a déclenché la décision, et les probabilités détaillées.
    """
    # Identifiant numérique de la classe : 0 = Normal, 1 = Surveillance, 2 = Urgence
    label: int

    # Statut lisible destiné à l'affichage dans l'API et le Dashboard
    status: str

    # Émoji de couleur correspondant au niveau d'alerte
    icon: str

    # Description de la règle qui a déclenché la décision (pour la transparence)
    rule_triggered: str

    # Dictionnaire des probabilités brutes issues de XGBoost
    probabilities: dict

    # Niveau de risque global (somme des probabilités anormales) en pourcentage
    cumulated_risk_pct: float


# ==============================================================================
# FONCTION PRINCIPALE DE DÉCISION
# ==============================================================================

def compute_smart_decision(
    prob_normal: float,
    prob_suggested: float,
    prob_urgent: float,
) -> DecisionResult:
    """
    Applique la cascade de règles probabilistes pour déterminer le statut final
    d'une éolienne à partir des sorties brutes du modèle XGBoost.

    Arguments :
        prob_normal    : Probabilité XGBoost que l'éolienne soit en état Normal (Label 0)
        prob_suggested : Probabilité XGBoost que l'éolienne nécessite une maintenance (Label 1)
        prob_urgent    : Probabilité XGBoost que l'éolienne nécessite une intervention urgente (Label 2)

    Retourne :
        DecisionResult : Objet de décision complet avec statut, règle et probabilités
    """
    # Validation défensive des probabilités en entrée
    for name, p in [("prob_normal", prob_normal), ("prob_suggested", prob_suggested), ("prob_urgent", prob_urgent)]:
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"Les probabilités doivent être comprises entre 0 et 1 (obtenu {name}={p})")

    total_prob = prob_normal + prob_suggested + prob_urgent
    if abs(total_prob - 1.0) > 0.05:
        raise ValueError(f"La somme des probabilités doit être égale à 1.0 (obtenu {total_prob:.4f})")

    # Calcul du risque cumulé des deux scénarios anormaux
    cumulated_risk = prob_suggested + prob_urgent

    # Dictionnaire des probabilités brutes pour l'affichage
    probs = {
        "normal":               round(prob_normal,    4),
        "maintenance_suggested": round(prob_suggested, 4),
        "urgent":               round(prob_urgent,    4),
    }

    # -----------------------------------------------------------------
    # RÈGLE 1 : Probabilité d'urgence seule dépasse le seuil de 20%
    # → Même si XGBoost dit "Normal", on déclare une Alerte Urgente
    # -----------------------------------------------------------------
    if prob_urgent >= THRESHOLD_URGENT:
        return DecisionResult(
            label=2,
            status="Immediate Maintenance Required",
            icon="🔴",
            rule_triggered=f"Seuil d'urgence probabiliste atteint ({prob_urgent*100:.1f}% >= {THRESHOLD_URGENT*100:.0f}%)",
            probabilities=probs,
            cumulated_risk_pct=round(cumulated_risk * 100, 2),
        )

    # -----------------------------------------------------------------
    # RÈGLE 2 : Probabilité de maintenance conseillée dépasse 25%
    # → Déclaration d'une Surveillance Requise
    # -----------------------------------------------------------------
    if prob_suggested >= THRESHOLD_SUGGESTED:
        return DecisionResult(
            label=1,
            status="Maintenance Suggested",
            icon="🟡",
            rule_triggered=f"Seuil de maintenance conseillée atteint ({prob_suggested*100:.1f}% >= {THRESHOLD_SUGGESTED*100:.0f}%)",
            probabilities=probs,
            cumulated_risk_pct=round(cumulated_risk * 100, 2),
        )

    # -----------------------------------------------------------------
    # RÈGLE 3 : Risque cumulé (urgence + conseillée) dépasse 35%
    # → La machine est considérée anormale même si XGBoost dit "Normal"
    # → C'est la règle qui protège contre les cas ambigus (ex: Turbine 1 : 39.74%)
    # -----------------------------------------------------------------
    if cumulated_risk >= THRESHOLD_CUMULATED:
        return DecisionResult(
            label=1,
            status="Maintenance Suggested",
            icon="🟡",
            rule_triggered=f"Risque cumulé anormal ({cumulated_risk*100:.1f}% >= {THRESHOLD_CUMULATED*100:.0f}%) — XGBoost disait Normal",
            probabilities=probs,
            cumulated_risk_pct=round(cumulated_risk * 100, 2),
        )

    # -----------------------------------------------------------------
    # RÈGLE 4 (par défaut) : Tous les seuils sous contrôle → Normal Confirmé
    # -----------------------------------------------------------------
    return DecisionResult(
        label=0,
        status="Normal (Healthy)",
        icon="🟢",
        rule_triggered="Tous les seuils probabilistes sous contrôle",
        probabilities=probs,
        cumulated_risk_pct=round(cumulated_risk * 100, 2),
    )
