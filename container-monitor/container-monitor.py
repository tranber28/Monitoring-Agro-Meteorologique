import paho.mqtt.client as mqtt
import json
import os
import time
import docker
from datetime import datetime

MQTT_BROKER = os.environ.get("MQTT_BROKER", "mosquitto")
MQTT_USER = os.environ.get("MQTT_USER", "")
MQTT_PASS = os.environ.get("MQTT_PASS", "")
INTERVAL = int(os.environ.get("CHECK_INTERVAL", "30"))

EXCLUDE = ["container-monitor", "mqtt-to-db", "ha-auto-discovery"]

client_docker = docker.from_env()

def get_all_containers():
    try:
        containers = client_docker.containers.list()
        return [c.name for c in containers if c.name not in EXCLUDE]
    except Exception as e:
        print(f"[ERROR Docker] {e}")
        return []

def get_container_status(container_name):
    try:
        container = client_docker.containers.get(container_name)
        status = container.status
        
        health = "unknown"
        if hasattr(container, 'health') and container.health:
            health = container.health
        elif status == "running":
            health = "healthy"
        
        return status, health
    except Exception as e:
        return "unknown", "unknown"

def publish_status(client, containers):
    for container in containers:
        if not container:
            continue
            
        status, health = get_container_status(container)
        
        safe_name = container.replace("-", "_").replace(".", "_")
        
        topic = f"container/{container}/status"
        
        # Plain text pour HA binary_sensor
        client.publish(topic, status)
        
        # JSON complet pour autres usages
        client.publish(f"{topic}/json", json.dumps({
            "status": status,
            "health": health,
            "timestamp": datetime.now().isoformat()
        }))
        
        # HA Auto-discovery binary_sensor
        ha_topic = f"homeassistant/binary_sensor/{safe_name}_status/config"
        payload = {
            "name": f"{container} Status",
            "state_topic": topic,
            "unique_id": f"container_{safe_name}",
            "device": {
                "name": "Container Monitor",
                "identifiers": ["container_monitor"]
            },
            "payload_on": "running",
            "payload_off": "stopped"
        }
        client.publish(ha_topic, json.dumps(payload), retain=True)
        
        print(f"[CONTAINER] {container}: {status} ({health})")

def on_connect(client, userdata, flags, rc):
    print(f"[MONITOR] Connecté à MQTT")

client = mqtt.Client()
if MQTT_USER:
    client.username_pw_set(MQTT_USER, MQTT_PASS)
client.on_connect = on_connect

print(f"[MQTT] Connexion à {MQTT_BROKER}:1883...")
client.connect(MQTT_BROKER, 1883, 60)
client.loop_start()

print(f"[MONITOR] Démarrage, interval={INTERVAL}s")

while True:
    try:
        containers = get_all_containers()
        print(f"[MONITOR] {len(containers)} containers détectés")
        publish_status(client, containers)
        time.sleep(INTERVAL)
    except Exception as e:
        print(f"[ERROR] {e}")
        time.sleep(INTERVAL)
