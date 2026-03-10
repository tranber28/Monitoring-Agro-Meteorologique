# ⚠️ Configuration Requise

Avant de démarrer ce projet, vous DEVEZ modifier les valeurs marquées `XXXX` par vos propres identifiants.

## Mots de passe à changer

| Fichier | Variable | Description |
|---------|----------|-------------|
| `homeassistant/docker-compose.yml` | `PASSWORD` | Mot de passe code-server |
| `homeassistant/docker-compose.yml` | `HTTP_USER`, `HTTP_PASSWORD` | Identifiants MQTT Explorer |
| `n8n/docker-compose.yml` | `N8N_BASIC_AUTH_USER`, `N8N_BASIC_AUTH_PASSWORD` | Identifiants n8n |
| `n8n/docker-compose.yml` | `DB_POSTGRESDB_USER`, `DB_POSTGRESDB_PASSWORD` | BDD n8n |
| `n8n/docker-compose.yml` | `POSTGRES_USER`, `POSTGRES_PASSWORD` | BDD PostgreSQL |
| `mariadb-agro/docker-compose.yml` | `MARIADB_ROOT_PASSWORD` | Mot de passe root MariaDB |
| `mariadb-agro/docker-compose.yml` | `MARIADB_USER`, `MARIADB_PASSWORD` | Utilisateur BDD |
| `chirpstack/docker-compose.yml` | `POSTGRES_USER`, `POSTGRES_PASSWORD` | BDD ChirpStack |
| `mqtt-to-db/docker-compose.yml` | `DB_USER`, `DB_PASSWORD` | Utilisateur BDD |
| `ai-engine/docker-compose.yml` | `DB_USER`, `DB_PASSWORD` | Utilisateur BDD |
| `homeassistant/config/secrets.yaml` | `some_password` | Secrets Home Assistant |

## Chemins personnalisés

| Fichier | Variable | Description |
|---------|----------|-------------|
| `dockge/compose.yaml` | `DOCKGE_STACKS_DIR` | Chemin vers vos stacks Docker |

## Données GPS (confidentielles)

| Fichier | Variable | Description |
|---------|----------|-------------|
| `mosquito-sx1302-lora-receiver/config/custom/global_conf.json` | `ref_latitude`, `ref_longitude`, `ref_altitude` | Position de la gateway |
| `mosquito-sx1302-lora-receiver/packet_forwarder/global_conf.json` | `ref_latitude`, `ref_longitude`, `ref_altitude` | Position de la gateway |

---

**IMPORTANT**: Ces valeurs contiennent des informations personnelles/géographiques sensibles. Ne jamais les partager publiquement.
