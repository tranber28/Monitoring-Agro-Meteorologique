🌿 Système Open-Source de Monitoring Agro-Météorologique

Ce projet propose une solution complète et modulaire de monitoring agricole de précision, optimisée pour les terrains vastes (10+ hectares) et les environnements à fortes contraintes climatiques. Le système repose sur une architecture LoRaWAN pour la transmission longue portée et intègre une couche d'Intelligence Artificielle locale pour la correction micro-climatique.

🚀 Objectifs du Projet

- **Gestion Hydrique Prédictive** : Monitoring de l'humidité du sol sur plusieurs profondeurs (10, 30, 60 cm) pour optimiser les ressources sans irrigation.
- **Suivi de Croissance (Dendrométrie)** : Mesure des variations de diamètre des troncs pour détecter le stress hydrique invisible à l'œil nu.
- **IA de Correction Micro-Climatique** : Comparaison entre prévisions météo globales (Open-Meteo) et données terrain pour affiner les alertes (gel, canicule, vent).
- **Autonomie Totale** : Nœuds basse consommation alimentés par énergie solaire.

🏗️ Architecture du Système

Le projet est conçu de manière modulaire pour être déployé via Docker sur un Raspberry Pi 4.

### 1. Stack Logicielle (Gateway)

| Service | Description | Port |
|---------|-------------|------|
| **Home Assistant** | Interface de monitoring et agrégation des données | 8123 |
| **Mosquitto MQTT** | Broker de messages pour la communication entre les nœuds | 1883 |
| **n8n** | Automation et workflows | 5678 |
| **ChirpStack** | Serveur LoRaWAN (gateway, network server) | 8080 |
| **MariaDB** | Base de données pour Agro-Monitoring | 3306 |
| **PostgreSQL (ChirpStack)** | Base de données ChirpStack | - |
| **PostgreSQL (n8n)** | Base de données n8n | - |
| **Redis** | Cache pour ChirpStack | - |
| **Portainer** | Gestion des containers Docker | 9000 |
| **Dockge** | Gestionnaire de stacks Docker | 5001 |
| **WeeWX** | Station météo (Vantage/USB) | - |
| **Open-Meteo** | API météo locale | 5000 |
| **MQTT Explorer** | Explorateur de messages MQTT | 4000 |
| **Code Server** | Éditeur de code dans le navigateur | 3218 |
| **Container Monitor** | Monitoring des containers via MQTT | - |
| **AI Engine** | Scripts Python de Machine Learning & Corrélation | - |
| **MQTT to DB** | Script Python pour sauvegarder les données MQTT en base | - |

### 2. Matériel (Nodes)

- **Microcontrôleurs** : ESP32, ESP32-S3 (Heltec V3), ESP8266 (D1 Mini)
- **Radio** : Modules LoRa 868 MHz (SX1276 / LLCC68)
- **Passerelle LoRa** : SX1302/SX1250 USB (dragino, seeed, etc.)
- **Capteurs** :
  - BME680 (Qualité air, pression, temp, hum)
  - LTR390 (UV Index & Lux)
  - Sondes capacitives V1.2 (Humidité sol)
  - GY-521 / Potentiomètres (Dendrométrie)

📂 Structure du Répertoire

```
.
├── ai-engine/                     # Scripts Python de ML (à compléter)
├── chirpstack/                    # ChirpStack + PostgreSQL + Redis
├── container-monitor/             # Monitoring des containers
├── dockge/                        # Gestionnaire Docker
├── gateway-bridge/                # Bridge MQTT <-> ChirpStack
├── ha-auto-discovery/             # Auto-discovery Home Assistant
├── homeassistant/                 # HA config + code-server + mqtt-explorer
├── mariadb-agro/                  # MariaDB pour agro-monitoring
├── mosquitto-sx1302-lora-receiver/ # Config LoRa gateway
├── mqtt-to-db/                    # Script Python MQTT -> BDD
├── n8n/                           # n8n + PostgreSQL
├── openmeteo/                     # API météo locale
├── packet-multiplexer/            # Multiplexeur de paquets LoRa
├── portainer/                     # Gestion Docker
├── weewx/                         # Station météo
└── CONFIG.md                      # ⚠️ Configuration à modifier
```

🤖 Intelligence Artificielle & Analyse

Le module ai-engine utilise les données historiques pour réduire le "biais" des prévisions météo publiques.

Exemple de logique :
1. Récupération de la prévision $T_{prev}$ via API Open-Meteo
2. Capture de la température réelle $T_{real}$ via le capteurs BME680 local
3. Entraînement d'un modèle de régression pour identifier l'influence du relief et de la végétation
4. Génération d'une prévision corrigée plus précise pour les décisions agricoles

⚠️ CONFIGURATION OBLIGATOIRE AVANT DÉMARRAGE

⚠️ IMPORTANT : Tous les fichiers ci-dessous contiennent des valeurs marquées `XXXX` qui DOIVENT être remplacées par vos propres identifiants avant de lancer les containers.

| Fichier | Éléments à modifier |
|---------|---------------------|
| `homeassistant/docker-compose.yml` | `PASSWORD` (code-server), `HTTP_USER`, `HTTP_PASSWORD` (MQTT Explorer) |
| `homeassistant/config/secrets.yaml` | `some_password` |
| `n8n/docker-compose.yml` | `N8N_BASIC_AUTH_USER`, `N8N_BASIC_AUTH_PASSWORD`, `DB_POSTGRESDB_*`, `POSTGRES_*` |
| `mariadb-agro/docker-compose.yml` | `MARIADB_ROOT_PASSWORD`, `MARIADB_USER`, `MARIADB_PASSWORD` |
| `mariadb-agro/Dockerfile` | `MARIADB_ROOT_PASSWORD`, `MARIADB_USER`, `MARIADB_PASSWORD` |
| `chirpstack/docker-compose.yml` | `POSTGRES_USER`, `POSTGRES_PASSWORD` |
| `chirpstack/config/chirpstack.toml` | `dsn` (mot de passe BDD), `secret` (API key) |
| `gateway-bridge/config.toml` | `api_token` (token API ChirpStack) |
| `mqtt-to-db/docker-compose.yml` | `DB_USER`, `DB_PASSWORD` |
| `mqtt-to-db/mqtt-to-db.py` | `DB_USER`, `DB_PASSWORD` (valeurs par défaut) |
| `ai-engine/docker-compose.yml` | `DB_USER`, `DB_PASSWORD` |
| `ai-engine/brain.py` | `DB_USER`, `DB_PASSWORD` (valeurs par défaut) |
| `dockge/compose.yaml` | `DOCKGE_STACKS_DIR` (chemin vers vos stacks) |
| `openmeteo/docker-compose.yml` | `LAT`, `LON` (coordonnées GPS) |
| `mosquito-sx1302-lora-receiver/config/custom/global_conf.json` | `ref_latitude`, `ref_longitude`, `ref_altitude` (GPS) |
| `mosquito-sx1302-lora-receiver/packet_forwarder/global_conf.json` | `ref_latitude`, `ref_longitude`, `ref_altitude` (GPS) |

🔧 Installation Rapide

```bash
# 1. Cloner le projet
git clone https://github.com/ton-pseudo/projet-agro-monitoring.git
cd projet-agro-monitoring

# 2.Configurer les identifiants (voir CONFIG.md)
#    Remplacer tous les XXXX par vos mots de passe

# 3. Créer le réseau Docker partagé
docker network create lora-network

# 4. Lancer les services uno par uno ou par groupe

# Gateway de base (recommandé en premier)
cd homeassistant && docker-compose up -d
cd ..

# LoRaWAN
cd chirpstack && docker-compose up -d
cd packet-multiplexer && docker-compose up -d
cd gateway-bridge && docker-compose up -d

# Bases de données
cd mariadb-agro && docker-compose up -d

# Automation
cd n8n && docker-compose up -d

# Monitoring
cd portainer && docker-compose up -d
cd dockge && docker-compose up -d
```

🛠️ Services Optionnels

- **WeeWX** : Brancher une station météo USB (Vantage, etc.)
- **MQTT Explorer** : Accessible via http://localhost:4000
- **Code Server** : Accessible via https://localhost:3218 (login: admin, password: XXXX)

📄 Licence

Ce projet est sous licence MIT. N'hésitez pas à l'utiliser, le modifier et partager vos améliorations.
