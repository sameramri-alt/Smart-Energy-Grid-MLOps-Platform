# ==============================================================================
# PROJET SMART ENERGY GRID - PHASE 5 : DASHBOARD STREAMLIT
# Fichier : app/dashboard.py
# Description :
#   Interface visuelle interactive pour le suivi en temps réel des éoliennes.
#   Le Dashboard interroge l'API FastAPI (app/main.py) pour obtenir les diagnostics,
#   puis affiche les résultats sous forme de tableaux colorés, de jauges et de graphiques.
#
#   Sections du Dashboard :
#     1. Tableau de bord global (statut de toutes les éoliennes)
#     2. Détail d'une éolienne sélectionnée (features + probabilités)
#     3. Simulateur manuel (POST /predict) avec sliders interactifs
#
#   Démarrage :
#     streamlit run app/dashboard.py
# ==============================================================================

# Importation de Streamlit pour la création de l'interface visuelle
import streamlit as st

# Importation de httpx pour faire des requêtes HTTP vers l'API FastAPI
import httpx

# Importation de pandas pour manipuler et afficher les données tabulaires
import pandas as pd

# Importation de plotly pour les graphiques interactifs (jauges, barres)
import plotly.graph_objects as go

# Importation du module standard pour les pauses temporelles (rafraîchissement)
import time


# ==============================================================================
# CONFIGURATION DE LA PAGE STREAMLIT
# ==============================================================================

# Configuration de la page : titre, icône, et disposition plein écran
st.set_page_config(
    page_title="Smart Energy Grid — Surveillance des Éoliennes",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Injection de CSS personnalisé pour améliorer l'apparence du dashboard
st.markdown("""
<style>
    /* Fond principal légèrement grisé */
    .main { background-color: #0e1117; }

    /* Cartes de métriques avec fond sombre et bordure arrondie */
    .metric-card {
        background-color: #1e2130;
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
        border-left: 4px solid #4ade80;
    }
    .metric-card.warning { border-left-color: #facc15; }
    .metric-card.danger  { border-left-color: #f87171; }

    /* Titre principal */
    h1 { color: #e2e8f0; font-size: 2.2rem !important; }

    /* Sous-titre de section */
    h2 { color: #94a3b8; font-size: 1.4rem !important; }

    /* Texte standard */
    p { color: #cbd5e1; }

    /* Badge de statut */
    .status-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# CONSTANTES
# ==============================================================================

# URL de base de l'API FastAPI
API_BASE_URL = "http://localhost:8000"

# Liste des ID de turbines à surveiller
TURBINE_IDS = [1, 2]

# Couleurs correspondant à chaque niveau de statut
STATUS_COLORS = {
    0: "#4ade80",    # Vert → Normal
    1: "#facc15",    # Jaune → Maintenance conseillée
    2: "#f87171",    # Rouge → Urgence
}


# ==============================================================================
# FONCTIONS D'ACCÈS À L'API
# ==============================================================================

def fetch_turbine_data(turbine_id: int) -> dict | None:
    """
    Interroge l'API FastAPI pour obtenir le diagnostic complet d'une éolienne.
    Retourne le dictionnaire JSON de réponse, ou None en cas d'erreur réseau.
    """
    try:
        # Requête HTTP GET avec un délai d'expiration de 5 secondes
        response = httpx.get(f"{API_BASE_URL}/turbine/{turbine_id}", timeout=5.0)
        response.raise_for_status()
        return response.json()
    except httpx.ConnectError:
        # L'API n'est pas démarrée ou inaccessible sur le port 8000
        return None
    except Exception as e:
        # Autre erreur réseau ou HTTP
        st.warning(f"Erreur lors de la récupération de Turbine {turbine_id}: {e}")
        return None


def fetch_manual_prediction(sensor_values: dict) -> dict | None:
    """
    Envoie les valeurs manuelles des capteurs à la route POST /predict
    et retourne la prédiction complète de l'API.
    """
    try:
        response = httpx.post(f"{API_BASE_URL}/predict", json=sensor_values, timeout=5.0)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Erreur lors de l'appel de l'API : {e}")
        return None


def check_api_health() -> bool:
    """
    Vérifie si l'API FastAPI est démarrée et opérationnelle.
    Retourne True si l'API répond, False sinon.
    """
    try:
        resp = httpx.get(f"{API_BASE_URL}/health", timeout=3.0)
        return resp.status_code == 200
    except Exception:
        return False


# ==============================================================================
# COMPOSANTS VISUELS
# ==============================================================================

def build_gauge_chart(label: int, probs: dict, turbine_id: int) -> go.Figure:
    """
    Crée une jauge Plotly montrant le risque anormal cumulé (urgence + conseillée).
    La jauge passe du vert au rouge selon le pourcentage de risque.
    """
    # Calcul du risque cumulé en pourcentage
    cumulated = (probs.get("urgent", 0) + probs.get("maintenance_suggested", 0)) * 100

    # Création de la figure Plotly Gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=cumulated,
        title={"text": f"Turbine {turbine_id} — Risque Cumulé", "font": {"color": "#e2e8f0"}},
        number={"suffix": "%", "font": {"color": "#e2e8f0", "size": 32}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#94a3b8", "tickfont": {"color": "#94a3b8"}},
            "bar": {"color": STATUS_COLORS.get(label, "#4ade80"), "thickness": 0.25},
            "bgcolor": "#1e2130",
            "bordercolor": "#334155",
            "steps": [
                {"range": [0, 35],  "color": "#14532d"},   # Vert sombre → zone sûre
                {"range": [35, 50], "color": "#713f12"},   # Orange sombre → vigilance
                {"range": [50, 100],"color": "#7f1d1d"},   # Rouge sombre → danger
            ],
            "threshold": {
                "line": {"color": "#f87171", "width": 3},
                "thickness": 0.8,
                "value": 35,  # Ligne marquant le seuil de décision probabiliste
            },
        },
    ))
    # Fond transparent pour s'intégrer au style du dashboard
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=220,
        margin=dict(t=50, b=10, l=10, r=10),
    )
    return fig


def build_probabilities_chart(probs: dict) -> go.Figure:
    """
    Crée un graphique en barres horizontales montrant la décomposition
    des probabilités XGBoost pour les trois classes.
    """
    labels = ["🟢 Normal", "🟡 Maintenance\nConseillée", "🔴 Urgence"]
    values = [
        probs.get("normal", 0) * 100,
        probs.get("maintenance_suggested", 0) * 100,
        probs.get("urgent", 0) * 100,
    ]
    colors = ["#4ade80", "#facc15", "#f87171"]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker=dict(color=colors, line=dict(color="#1e2130", width=1)),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(color="#e2e8f0", size=13),
    ))
    # Ajout de lignes verticales pour les seuils
    fig.add_vline(x=20, line_dash="dash", line_color="#f87171", annotation_text="Seuil Urgence (20%)", annotation_font_color="#f87171")
    fig.add_vline(x=25, line_dash="dash", line_color="#facc15", annotation_text="Seuil Conseillée (25%)", annotation_font_color="#facc15")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 110], ticksuffix="%", color="#94a3b8"),
        yaxis=dict(color="#e2e8f0"),
        height=200,
        margin=dict(t=10, b=10, l=10, r=10),
        showlegend=False,
    )
    return fig


# ==============================================================================
# SIDEBAR : NAVIGATION ET CONFIGURATION
# ==============================================================================

with st.sidebar:
    st.image("https://img.icons8.com/fluency/100/wind-turbine.png", width=80)
    st.title("⚡ Smart Energy Grid")
    st.caption("Surveillance Prédictive des Éoliennes")
    st.divider()

    # Vérification de l'état de l'API dans la sidebar
    api_online = check_api_health()
    if api_online:
        st.success("🟢 API FastAPI connectée")
    else:
        st.error("🔴 API hors ligne — Lancez : `uvicorn app.main:app --port 8000`")

    st.divider()

    # Sélection de la page à afficher
    page = st.radio(
        "Navigation",
        options=["📊 Tableau de Bord Global", "🔍 Détail d'une Éolienne", "🧪 Simulateur Manuel"],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("Seuils de décision probabilistes :")
    st.caption("🔴 Urgente : prob ≥ **20%**")
    st.caption("🟡 Conseillée : prob ≥ **25%**")
    st.caption("🟡 Risque cumulé : ≥ **35%**")

    st.divider()
    st.subheader("⏱️ Actualisation en Direct")
    auto_refresh = st.toggle("Rafraîchissement Auto", value=True)
    refresh_sec = st.slider("Toutes les X secondes :", min_value=2, max_value=30, value=5, step=1)


# ==============================================================================
# PAGE 1 : TABLEAU DE BORD GLOBAL
# ==============================================================================

if page == "📊 Tableau de Bord Global":

    st.title("📊 Tableau de Bord — Surveillance du Parc Éolien")
    st.caption("Données temps réel issues de Redis via Feast → Inférence XGBoost → Logique probabiliste")

    from datetime import datetime
    now_str = datetime.now().strftime("%H:%M:%S")

    # Barre d'action et indicateur de rafraîchissement
    col_refresh, col_info = st.columns([1, 4])
    with col_refresh:
        refresh = st.button("🔄 Rafraîchir maintenant", use_container_width=True)
    with col_info:
        if auto_refresh:
            st.info(f"🟢 Rafraîchissement automatique actif (toutes les **{refresh_sec}s**) — Dernière MAJ : **{now_str}**")
        else:
            st.caption(f"Dernière mise à jour manuelle : {now_str}")

    # Affichage d'un avertissement si l'API est hors ligne
    if not api_online:
        st.warning("⚠️ L'API FastAPI est hors ligne. Démarrez-la avec : `uvicorn app.main:app --port 8000`")
        st.stop()

    # Collecte des données pour toutes les éoliennes
    all_data = {}
    for tid in TURBINE_IDS:
        data = fetch_turbine_data(tid)
        if data:
            all_data[tid] = data

    if not all_data:
        st.error("Impossible de récupérer les données des éoliennes. Vérifiez l'API et Feast.")
        st.stop()

    st.divider()

    # Affichage des jauges de risque pour chaque turbine
    cols_gauges = st.columns(len(TURBINE_IDS))
    for i, tid in enumerate(TURBINE_IDS):
        if tid in all_data:
            d = all_data[tid]
            with cols_gauges[i]:
                st.plotly_chart(
                    build_gauge_chart(d["predicted_label"], d["probabilities"], tid),
                    use_container_width=True,
                )

    st.divider()

    # Tableau récapitulatif coloré de toutes les éoliennes
    st.subheader("📋 Récapitulatif de l'État des Turbines")

    # Construction des données du tableau
    rows = []
    for tid, d in all_data.items():
        rows.append({
            "Éolienne": f"Turbine {tid}",
            "Statut": f"{d['icon']} {d['status']}",
            "Vitesse (RPM)": f"{d['features_used'].get('Rotor_Speed_RPM', 0):.2f}",
            "Puissance (kW)": f"{d['features_used'].get('Power_Output_kW', 0):.1f}",
            "Stress Thermique (°C)": f"{d['features_used'].get('Thermal_Stress', 0):.1f}",
            "Vibration (mm/s)": f"{d['features_used'].get('Vibration_Level_mmps', 0):.2f}",
            "Risque Cumulé (%)": f"{d['cumulated_risk_pct']:.1f}%",
            "Règle Déclenchée": d["rule_triggered"],
        })

    df_summary = pd.DataFrame(rows)

    # Affichage du tableau avec coloration conditionnelle des cellules de statut
    st.dataframe(
        df_summary,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Statut": st.column_config.TextColumn("Statut", width="medium"),
            "Règle Déclenchée": st.column_config.TextColumn("Règle de Décision", width="large"),
        },
    )

    # Rafraîchissement automatique après l'intervalle configuré
    if auto_refresh:
        time.sleep(refresh_sec)
        st.rerun()


# ==============================================================================
# PAGE 2 : DÉTAIL D'UNE ÉOLIENNE
# ==============================================================================

elif page == "🔍 Détail d'une Éolienne":

    st.title("🔍 Diagnostic Détaillé d'une Éolienne")

    if not api_online:
        st.warning("⚠️ L'API FastAPI est hors ligne. Démarrez-la avec : `uvicorn app.main:app --port 8000`")
        st.stop()

    # Menu déroulant pour choisir la turbine
    selected_tid = st.selectbox("Sélectionnez une éolienne :", TURBINE_IDS, format_func=lambda x: f"Turbine {x}")

    # Chargement des données de la turbine sélectionnée
    with st.spinner(f"Interrogation de Redis pour Turbine {selected_tid}..."):
        data = fetch_turbine_data(selected_tid)

    if not data:
        st.error(f"Impossible de récupérer les données de Turbine {selected_tid}.")
        st.stop()

    # Affichage du statut principal
    label = data["predicted_label"]
    color = STATUS_COLORS.get(label, "#94a3b8")
    st.markdown(
        f"<div style='background:{color}22; border-left: 6px solid {color}; padding:16px; border-radius:8px;'>"
        f"<h2 style='color:{color}; margin:0;'>{data['icon']} {data['status']}</h2>"
        f"<p style='color:#94a3b8; margin:4px 0 0 0; font-size:0.9rem;'>Règle : {data['rule_triggered']}</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    # Graphique des probabilités
    col_proba, col_features = st.columns(2)
    with col_proba:
        st.subheader("Répartition des Probabilités XGBoost")
        st.plotly_chart(build_probabilities_chart(data["probabilities"]), use_container_width=True)
        st.caption(f"⚠️ Risque Anormal Cumulé : **{data['cumulated_risk_pct']}%**")

    # Métriques des capteurs physiques
    with col_features:
        st.subheader("Capteurs Physiques (dernière mesure)")
        feats = data["features_used"]
        m1, m2 = st.columns(2)
        m1.metric("Vitesse Rotor (RPM)", f"{feats.get('Rotor_Speed_RPM', 0):.2f}")
        m2.metric("Vitesse Vent (m/s)", f"{feats.get('Wind_Speed_mps', 0):.2f}")
        m1.metric("Puissance (kW)", f"{feats.get('Power_Output_kW', 0):.1f}")
        m2.metric("Humidité (%)", f"{feats.get('Humidity_pct', 0):.1f}")
        m1.metric("Temp. Huile (°C)", f"{feats.get('Gearbox_Oil_Temp_C', 0):.1f}")
        m2.metric("Temp. Roulement (°C)", f"{feats.get('Generator_Bearing_Temp_C', 0):.1f}")
        m1.metric("Vibration (mm/s)", f"{feats.get('Vibration_Level_mmps', 0):.2f}")
        m2.metric("Temp. Ambiante (°C)", f"{feats.get('Ambient_Temp_C', 0):.1f}")

    st.divider()
    st.subheader("Indicateurs Dérivés (Feature Engineering)")
    cols_fe = st.columns(5)
    cols_fe[0].metric("Stress Thermique (°C)", f"{feats.get('Thermal_Stress', 0):.2f}")
    cols_fe[1].metric("Déséq. Thermique (°C)", f"{feats.get('Thermal_Imbalance', 0):.2f}")
    cols_fe[2].metric("Efficacité Énergetique", f"{feats.get('Power_Efficiency', 0):.2f}")
    cols_fe[3].metric("Vibration / RPM", f"{feats.get('Vibration_per_RPM', 0):.4f}")
    cols_fe[4].metric("Indice Chaleur/Humidité", f"{feats.get('Heat_Humidity_Index', 0):.2f}")


# ==============================================================================
# PAGE 3 : SIMULATEUR MANUEL
# ==============================================================================

elif page == "🧪 Simulateur Manuel":

    st.title("🧪 Simulateur de Diagnostic Manuel")
    st.caption("Ajustez les valeurs des capteurs pour simuler différents scénarios et observer la réponse du modèle en temps réel.")

    if not api_online:
        st.warning("⚠️ L'API FastAPI est hors ligne. Démarrez-la avec : `uvicorn app.main:app --port 8000`")
        st.stop()

    # Formulaire avec sliders pour chaque capteur
    with st.form("manual_predict_form"):
        st.subheader("🎛️ Valeurs des Capteurs")

        col1, col2 = st.columns(2)

        with col1:
            rotor_rpm  = st.slider("Vitesse Rotor (RPM)", 0.0, 50.0, 16.0, 0.1)
            wind_speed = st.slider("Vitesse Vent (m/s)", 0.0, 40.0, 8.0, 0.1)
            power_kw   = st.slider("Puissance Produite (kW)", 0.0, 3000.0, 1500.0, 10.0)
            gearbox_t  = st.slider("Temp. Huile Multiplicateur (°C)", 0.0, 150.0, 65.0, 0.5)

        with col2:
            bearing_t  = st.slider("Temp. Roulement Génératrice (°C)", 0.0, 150.0, 72.0, 0.5)
            vibration  = st.slider("Niveau Vibration (mm/s)", 0.0, 20.0, 2.0, 0.05)
            ambient_t  = st.slider("Temp. Ambiante (°C)", -30.0, 60.0, 12.0, 0.5)
            humidity   = st.slider("Humidité Relative (%)", 0.0, 100.0, 65.0, 1.0)

        # Bouton de soumission du formulaire
        submitted = st.form_submit_button("▶️ Lancer la Prédiction", use_container_width=True, type="primary")

    if submitted:
        # Construction du dictionnaire de données à envoyer à l'API
        sensor_data = {
            "Rotor_Speed_RPM": rotor_rpm,
            "Wind_Speed_mps": wind_speed,
            "Power_Output_kW": power_kw,
            "Gearbox_Oil_Temp_C": gearbox_t,
            "Generator_Bearing_Temp_C": bearing_t,
            "Vibration_Level_mmps": vibration,
            "Ambient_Temp_C": ambient_t,
            "Humidity_pct": humidity,
        }

        # Envoi de la requête POST à l'API
        with st.spinner("Envoi à l'API FastAPI → Calcul XGBoost..."):
            result = fetch_manual_prediction(sensor_data)

        if result:
            # Affichage du résultat
            label = result["predicted_label"]
            color = STATUS_COLORS.get(label, "#94a3b8")

            st.markdown(
                f"<div style='background:{color}22; border-left: 6px solid {color}; padding:20px; border-radius:8px; margin-top:16px;'>"
                f"<h2 style='color:{color}; margin:0;'>{result['icon']} {result['status']}</h2>"
                f"<p style='color:#94a3b8; margin:8px 0 0 0;'>Règle déclenchée : {result['rule_triggered']}</p>"
                f"<p style='color:#94a3b8; margin:4px 0 0 0;'>Risque Cumulé Anormal : <strong style=\"color:{color}\">{result['cumulated_risk_pct']}%</strong></p>"
                "</div>",
                unsafe_allow_html=True,
            )
            st.markdown("")

            # Graphique des probabilités
            st.subheader("Répartition des Probabilités")
            st.plotly_chart(build_probabilities_chart(result["probabilities"]), use_container_width=True)

            # Features calculées automatiquement
            with st.expander("📐 Voir les 5 features calculées automatiquement par l'API"):
                feats = result["features_used"]
                fe_data = {
                    "Feature Calculée": [
                        "Thermal_Imbalance (Déséq. Thermique)",
                        "Power_Efficiency (Efficacité Énergetique)",
                        "Thermal_Stress (Stress Thermique)",
                        "Vibration_per_RPM",
                        "Heat_Humidity_Index (Chaleur/Humidité)",
                    ],
                    "Valeur": [
                        f"{feats.get('Thermal_Imbalance', 0):.4f}",
                        f"{feats.get('Power_Efficiency', 0):.4f}",
                        f"{feats.get('Thermal_Stress', 0):.4f}",
                        f"{feats.get('Vibration_per_RPM', 0):.6f}",
                        f"{feats.get('Heat_Humidity_Index', 0):.4f}",
                    ],
                }
                st.table(pd.DataFrame(fe_data))
