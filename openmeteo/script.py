import paho.mqtt.client as mqtt
import requests
import json
import os
import time
import sys

# Configuration
MQTT_HOST = os.getenv("MQTT_HOST", "mosquitto")
# Coordonnées précises pour Coggia
LAT = os.getenv("LAT", "42.1197")  
LON = os.getenv("LON", "8.733781")  

TOPIC_HA = "agro/meteo/openmeteo/state"
TOPIC_RAW = "agro/meteo/openmeteo/raw" # Le flux complet pour ton IA

client = mqtt.Client()

def setup_discovery():
    sensors = {
        "temp": {"name": "OM Temp", "unit": "°C", "class": "temperature", "key": "temperature_2m"},
        "hum": {"name": "OM Humidité", "unit": "%", "class": "humidity", "key": "relative_humidity_2m"},
        "dew": {"name": "OM Point Rosée", "unit": "°C", "class": "temperature", "key": "dew_point_2m"},
        "apparent": {"name": "OM Temp Ressentie", "unit": "°C", "class": "temperature", "key": "apparent_temperature"},
        "rain": {"name": "OM Pluie", "unit": "mm", "class": "precipitation", "key": "precipitation"},
        "pressure": {"name": "OM Pression", "unit": "hPa", "class": "pressure", "key": "pressure_msl"},
        "wind": {"name": "OM Vent", "unit": "km/h", "class": "wind_speed", "key": "wind_speed_10m"},
        "gusts": {"name": "OM Rafales", "unit": "km/h", "class": "wind_speed", "key": "wind_gusts_10m"},
        "uv": {"name": "OM UV", "unit": "index", "class": "irradiance", "key": "uv_index"},
        "radiation": {"name": "OM Rayonnement", "unit": "W/m²", "class": "irradiance", "key": "shortwave_radiation"},
        "et0": {"name": "OM ET0", "unit": "mm", "class": "precipitation", "key": "et0_fao_evapotranspiration"},
        "soil_0": {"name": "OM Sol Temp 0cm", "unit": "°C", "class": "temperature", "key": "soil_temperature_0cm"},
    }
    for s_id, info in sensors.items():
        config = {
            "name": info["name"],
            "state_topic": TOPIC_HA,
            "unit_of_measurement": info["unit"],
            "value_template": f"{{{{ value_json.{info['key']} }}}}",
            "unique_id": f"coggia_om_{s_id}",
            "device": {"identifiers": ["openmeteo_coggia"], "name": "Station Virtuelle Coggia"}
        }
        if info["class"]: config["device_class"] = info["class"]
        client.publish(f"homeassistant/sensor/coggia_om_{s_id}/config", json.dumps(config), retain=True)

def fetch_and_broadcast():
    params = (
        f"latitude={LAT}&longitude={LON}"
        "&current=temperature_2m,relative_humidity_2m,dew_point_2m,apparent_temperature,"
        "precipitation,rain,weather_code,cloud_cover,pressure_msl,"
        "wind_speed_10m,wind_direction_10m,wind_gusts_10m,"
        "shortwave_radiation,direct_radiation,diffuse_radiation,"
        "vapor_pressure_deficit,et0_fao_evapotranspiration,uv_index,"
        "soil_temperature_0cm,soil_temperature_6cm"
        "&timezone=auto"
    )
    url = f"https://api.open-meteo.com/v1/forecast?{params}"
    
    try:
        r = requests.get(url, timeout=30)
        print(f"[{time.strftime('%H:%M:%S')}] API status: {r.status_code}", flush=True)
        data = r.json()
        if "current" in data:
            client.publish(TOPIC_RAW, json.dumps(data))
            client.publish(TOPIC_HA, json.dumps(data["current"]))
            print(f"[{time.strftime('%H:%M:%S')}] Données Coggia envoyées", flush=True)
        else:
            print(f"[{time.strftime('%H:%M:%S')}] Pas de données current", flush=True)
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] Erreur : {e}", flush=True)

print(f"[{time.strftime('%H:%M:%S')}] Connexion à MQTT {MQTT_HOST}...", flush=True)
client.connect(MQTT_HOST, 1883, 60)
print(f"[{time.strftime('%H:%M:%S')}] MQTT connecté, setup discovery...", flush=True)
setup_discovery()
print(f"[{time.strftime('%H:%M:%S')}] Démarrage boucle principale...", flush=True)
client.loop_start()

while True:
    fetch_and_broadcast()
    time.sleep(900)
