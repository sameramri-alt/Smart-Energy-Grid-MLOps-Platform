# ⚡ Smart Energy Grid — Plateforme MLOps de Maintenance Prédictive pour Éoliennes

[![Smart Energy Grid CI Pipeline](https://github.com/sameramri-alt/projet/actions/workflows/ci.yml/badge.svg)](https://github.com/sameramri-alt/projet/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![Feast](https://img.shields.io/badge/Feature%20Store-Feast-38ef7d.svg)](https://feast.dev/)
[![Terraform](https://img.shields.io/badge/IaC-Terraform-7B42BC.svg)](https://www.terraform.io/)
[![Apache Iceberg](https://img.shields.io/badge/Lakehouse-Apache%20Iceberg-1f77b4.svg)](https://iceberg.apache.org/)

---

## 📖 Présentation du Projet

**Smart Energy Grid** est une solution MLOps industrielle de bout en bout dédiée à la **surveillance en temps réel et à la maintenance prédictive d'un parc d'éoliennes**.

En exploitant les flux continus de capteurs physiques IoT (vitesse des pales, température d'huile du multiplicateur, vibrations de la nacelle, puissance générée...), la plateforme ingère, stocke, calcule des indicateurs physiques avancés et prédit automatiquement les pannes potentielles avant qu'elles ne causent des arrêts critiques ou des dommages irréversibles.

---

## 🏗️ Architecture Globale du Système

```text
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             ARCHITECTURE FLUX DE DONNÉES                         │
└──────────────────────────────────────────────────────────────────────────────────┘

 [Capteurs IoT Éoliennes]
     │ (50 événements/s)
     ▼
 [Redis Stream Broker] ──────► [Ingestion Apache Iceberg] ──► [MinIO S3 + PostgreSQL Catalog]
                                          │                               │
                                          ▼                               ▼
                              [Entraînement XGBoost]           [Analyse SQL DuckDB]
                                          │
                                          ▼
                            [Feature Store Feast] ──────────► [Redis Online Store]
                                                               (Latence: 1.44 ms)
                                                                      │
                                                                      ▼
                                                            [FastAPI REST Engine]
                                                           (Logique Probabiliste)
                                                                      │
                                       ┌──────────────────────────────┴─────────────────────────────┐
                                       ▼                                                            ▼
                         [Dashboard Streamlit]                                     [Clients Externes / API]
                      (Monitoring Live & Alertes)                                   (JSON REST /predict)
```

---

## 🌟 Points Forts & Innovations Techniques

1. **Infrastructure as Code (IaC) avec Terraform** : Déploiement automatisé et reproductible des conteneurs MinIO (S3), PostgreSQL (catalogue Iceberg) et Redis (Cache & Feature Store).
2. **Lakehouse Moderne (Apache Iceberg + DuckDB)** : Ingestion ACID de flux IoT partitionnés par date et turbine, stockés en format Parquet sur MinIO et requêtables à chaud via DuckDB.
3. **Ingénierie de Variables Physiques (5 Features Métier)** :
   * `Thermal_Imbalance` : Écart thermique entre huile du multiplicateur et roulement de la génératrice.
   * `Power_Efficiency` : Ratio de conversion puissance / vitesse de rotation.
   * `Thermal_Stress` : Gradient de température d'huile vs air ambiant.
   * `Vibration_per_RPM` : Intensité vibratoire par tour mécanique effectué.
   * `Heat_Humidity_Index` : Indice combiné hygrométrie et chaleur.
4. **Feature Store Temps Réel (Feast + Redis)** : Centralisation des 13 variables avec synchronisation continue vers la mémoire vive Redis, offrant une latence de récupération de **1.44 milliseconde**.
5. **Logique de Décision Probabiliste (Surveillance Avancée)** :
   * Résout le problème des modèles de ML déséquilibrés en milieu industriel : si le risque cumulé de panne dépasse **35%**, une alerte de maintenance est levée même si XGBoost classe l'événement comme "Normal" (cas réel validé sur la Turbine 1).
6. **Double Interface Utilisateur** :
   * API REST FastAPI auto-documentée via Swagger (`/docs`).
   * Dashboard interactif Streamlit avec jauges de santé Plotly et rafraîchissement automatique réglable (toutes les X secondes).
7. **Intégration Continue (CI/CD)** : Pipeline GitHub Actions automatisé exécutant 19 tests unitaires Pytest, la validation Terraform et le contrôle de conformité Feast.

---

## 🗂️ Structure du Répertoire

```text
projet/
├── .github/
│   └── workflows/
│       └── ci.yml               # Pipeline d'Intégration Continue GitHub Actions
├── app/
│   ├── __init__.py
│   ├── decision.py              # Logique métier probabiliste (seuils 20%, 25%, 35%)
│   ├── main.py                  # Serveur API REST FastAPI
│   ├── dashboard.py             # Interface de monitoring interactif Streamlit
│   └── model.pkl                # Modèle Machine Learning XGBoost entraîné
├── data/
│   ├── Train.csv                # Données historiques d'entraînement
│   └── Test.csv                 # Données d'évaluation
├── feature_repository/
│   ├── feature_store.yaml       # Configuration Feast (Redis Online Store)
│   ├── definitions.py           # Déclaration de l'Entity et de la FeatureView
│   └── data/                    # Registre Feast et Parquet offline
├── scripts/
│   ├── stream_generator.py      # Générateur IoT de simulation de flux temps réel
│   ├── ingest_iceberg.py        # Ingestion Lakehouse + push automatique Feast/Redis
│   ├── verify_lakehouse.py      # Requêtage analytique DuckDB
│   ├── train_model.py           # Pipeline de feature engineering et d'entraînement XGBoost
│   └── prepare_feast_data.py    # Préparation des 13 features avec horodatages pour Feast
├── terraform/
│   ├── main.tf                  # Définition des conteneurs, volumes et réseaux Docker
│   └── variables.tf             # Variables de configuration de l'infrastructure
├── tests/
│   ├── test_decision.py         # Tests des règles de décision probabilistes
│   ├── test_features.py         # Tests des formules d'ingénierie physique
│   ├── test_model.py            # Tests de chargement et d'inférence XGBoost
│   ├── test_api_schemas.py      # Tests des schémas Pydantic et de l'API FastAPI
│   └── test_feast_definitions.py# Tests de conformité du Feature Store Feast
├── .gitignore                   # Exclusion des fichiers temporaires et caches
├── requirements.txt             # Dépendances Python du projet
└── README.md                    # Documentation officielle du projet
```

---

## ⚙️ Prérequis & Installation

### Prérequis
* **Python 3.10+**
* **Docker Desktop** (actif, moteur requis par Terraform)
* **Terraform** (gestion de l'infrastructure via le provider Docker)
* **Git**

### Installation Rapide

```powershell
# 1. Cloner le projet
git clone https://github.com/sameramri-alt/projet.git
cd projet

# 2. Créer et activer l'environnement virtuel
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Installer les dépendances
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Guide de Démarrage Étape par Étape

### 1. Démarrer l'Infrastructure (IaC)

```powershell
# Déploiement des conteneurs MinIO, PostgreSQL et Redis via Terraform
cd terraform
terraform init
terraform apply -auto-approve
cd ..
```

### 2. Entraîner le Modèle Machine Learning

```powershell
python scripts/train_model.py
# Produit le fichier sérialisé app/model.pkl
```

### 3. Initialiser le Feature Store Feast

```powershell
# Préparation du fichier Parquet offline (features avec horodatages)
python scripts/prepare_feast_data.py

# Enregistrement des définitions dans le Feature Store
cd feature_repository
feast apply
cd ..
```

> **Note :** La synchronisation vers Redis (Online Store) est désormais **automatique** :
> `ingest_iceberg.py` pousse les dernières features à chaque lot ingéré via `push_to_feast_online_store()`.
> Il n'est plus nécessaire de lancer `feast materialize` manuellement.

### 4. Démarrer le Pipeline de Streaming (2 terminaux)

```powershell
# Terminal A — Générer les données des capteurs en temps réel
python -u scripts/stream_generator.py --rate 5

# Terminal B — Ingérer dans Iceberg/MinIO ET mettre à jour Feast/Redis automatiquement
python -u scripts/ingest_iceberg.py --batch-size 10
```

### 5. Lancer l'API REST FastAPI

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
* **Documentation interactive Swagger** : [http://localhost:8000/docs](http://localhost:8000/docs)
* **Vérification de santé** : [http://localhost:8000/health](http://localhost:8000/health)

### 6. Lancer le Dashboard Visuel Streamlit

Dans un terminal séparé :
```powershell
streamlit run app/dashboard.py
```
* **Accès au Dashboard en direct** : [http://localhost:8501](http://localhost:8501)

---

## 📡 Spécification des Endpoints de l'API

| Méthode | Route | Description | Exemple de Réponse |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Vérifie l'état de l'API, du modèle et de Redis | `{"api": "ok", "model": "ok", "feast": "ok"}` |
| `GET` | `/turbine/{turbine_id}` | Diagnostic automatique d'une éolienne via Feast + XGBoost | Diagnostic complet, probabilités et règle déclenchée |
| `POST` | `/predict` | Inférence manuelle sur les 8 mesures physiques brutes | Inférence immédiate avec calcul des 5 features dérivées |

---

## 🧪 Tests Automatisés & Qualité

Pour exécuter localement l'ensemble des **19 tests unitaires** :

```powershell
pytest tests/ -v
```

Les tests valident automatiquement :
* La protection contre la division par zéro ($RPM = 0$).
* La prééminence des alertes de sécurité sur les décisions par défaut.
* Le calcul exact des features physiques.
* La compatibilité des schémas Pydantic.
* L'intégrité du modèle et du Feature Store.

---

## 🛠️ Stack Technologique

* **Orchestration d'Infrastructure** : Terraform, Docker Desktop
* **Stockage & Lakehouse** : MinIO (S3), Apache Iceberg, PostgreSQL, DuckDB, Parquet
* **Feature Store & Cache** : Feast 0.40+, Redis 7
* **Intelligence Artificielle** : XGBoost, Scikit-learn, Pandas, NumPy, Joblib
* **Backend & API** : FastAPI, Uvicorn, Pydantic v2, HTTPX
* **Visualisation & UI** : Streamlit, Plotly
* **Tests & CI/CD** : Pytest, GitHub Actions, Flake8

---

## 👤 Auteur & Licence

* **Auteur** : Samer Amri ([@sameramri-alt](https://github.com/sameramri-alt))
* **Projet** : Smart Energy Grid MLOps Platform
* **Licence** : MIT
