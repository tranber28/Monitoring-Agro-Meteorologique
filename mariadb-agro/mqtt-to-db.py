import paho.mqtt.client as mqtt
import mysql.connector
import json
import os
import time

MQTT_BROKER = os.environ.get("MQTT_BROKER", "mosquitto")
MQTT_USER = os.environ.get("MQTT_USER", "")
MQTT_PASS = os.environ.get("MQTT_PASS", "")

DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "mariadb-agro"),
    "user": os.environ.get("DB_USER", "XXXX"),
    "password": os.environ.get("DB_PASSWORD", "XXXX"),
    "database": os.environ.get("DB_NAME", "agro_monitoring")
}

SENSOR_AUTO_CREATE = os.environ.get("SENSOR_AUTO_CREATE", "true").lower() == "true"

SENSOR_MAP = {
    "sensor/meteo/temperature": ("temp_meteo", "temperature", "meteo_station", "°C"),
    "sensor/meteo/humidity": ("humidity_meteo", "humidity", "meteo_station", "%"),
    "sensor/meteo/pressure": ("pressure_meteo", "pressure", "meteo_station", "hPa"),
    "sensor/soil/10cm": ("soil_10cm", "soil_moisture", "verger_10cm", "%"),
    "sensor/soil/30cm": ("soil_30cm", "soil_moisture", "verger_30cm", "%"),
    "sensor/soil/60cm": ("soil_60cm", "soil_moisture", "verger_60cm", "%"),
    "sensor/verger/dendrometer": ("dendrometer", "diameter", "verger_tronc", "mm"),
    "sensor/meteo/uv": ("uv_index", "uv", "meteo_station", "index"),
    "sensor/meteo/lux": ("lux", "luminosity", "meteo_station", "lux"),
    "api/openmeteo/temperature": ("temp_meteo", "temperature", "openmeteo", "°C"),
    "ia/meteo/temp_corrigee": ("temp_meteo_ia", "temperature", "ia_correction", "°C"),
}

def ensure_sensor(cursor, topic):
    if not SENSOR_AUTO_CREATE:
        return SENSOR_MAP.get(topic, (None, None, None, None))[0]
    
    mapped = SENSOR_MAP.get(topic)
    if not mapped:
        name = topic.replace("/", "_").replace("#", "")
        sensor_type = "unknown"
        location = "auto"
        unit = ""
    else:
        name, sensor_type, location, unit = mapped
    
    cursor.execute("SELECT id FROM sensors WHERE name = %s", (name,))
    if not cursor.fetchone():
        cursor.execute(
            "INSERT INTO sensors (name, type, location, unit) VALUES (%s, %s, %s, %s)",
            (name, sensor_type, location, unit)
        )
        print(f"Capteur créé: {name} ({sensor_type})")
    return name

def get_db_connection():
    for i in range(10):
        try:
            return mysql.connector.connect(**DB_CONFIG)
        except mysql.connector.Error as e:
            print(f"Connexion DB échouée (tentative {i+1}/10): {e}")
            time.sleep(3)
    raise Exception("Impossible de se connecter à la DB")

def get_sensor_id(cursor, sensor_name):
    cursor.execute("SELECT id FROM sensors WHERE name = %s", (sensor_name,))
    result = cursor.fetchone()
    return result[0] if result else None

def insert_measurement(cursor, sensor_name, value, forecast_value=None, corrected_value=None):
    sensor_id = get_sensor_id(cursor, sensor_name)
    if sensor_id is None:
        print(f"Capteur inconnu: {sensor_name}")
        return
    
    cursor.execute(
        "INSERT INTO measurements (sensor_id, value, forecast_value, corrected_value) VALUES (%s, %s, %s, %s)",
        (sensor_id, value, forecast_value, corrected_value)
    )

def on_message(client, userdata, msg):
    try:
        db = get_db_connection()
        cursor = db.cursor()
        
        sensor_name = ensure_sensor(cursor, msg.topic)
        if not sensor_name:
            print(f"Topic non supporté: {msg.topic}")
            return
        
        payload = json.loads(msg.payload.decode()) if b'{' in msg.payload else {"value": float(msg.payload.decode())}
        value = payload.get("value") or payload.get("temperature") or payload.get("humidity") or payload.get("pressure")
        
        if value is None:
            print(f"Pas de valeur extraite du payload: {payload}")
            return
        
        forecast = payload.get("original")
        corrected = payload.get("value") if "corrigee" in msg.topic else None
        
        insert_measurement(cursor, sensor_name, value, forecast, corrected)
        db.commit()
        print(f"[{msg.topic}] {value} -> insert DB")
        
        cursor.close()
        db.close()
    except Exception as e:
        print(f"Erreur: {e}")

client = mqtt.Client()
if MQTT_USER:
    client.username_pw_set(MQTT_USER, MQTT_PASS)
client.on_message = on_message

print(f"Connexion à {MQTT_BROKER}...", flush=True)
client.connect(MQTT_BROKER, 1883)

for topic in SENSOR_MAP.keys():
    client.subscribe(topic)
    print(f"  - {topic}", flush=True)

print("En attente des messages...", flush=True)
client.loop_forever()
