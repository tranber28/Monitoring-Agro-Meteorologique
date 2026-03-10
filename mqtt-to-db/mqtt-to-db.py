import paho.mqtt.client as mqtt
import mysql.connector
import json
import os
import time
import re
import base64

MQTT_BROKER = os.environ.get("MQTT_BROKER", "mosquitto")
MQTT_USER = os.environ.get("MQTT_USER", "")
MQTT_PASS = os.environ.get("MQTT_PASS", "")

DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "mariadb-agro"),
    "user": os.environ.get("DB_USER", "XXXX"),
    "password": os.environ.get("DB_PASSWORD", "XXXX"),
    "database": os.environ.get("DB_NAME", "agro_monitoring")
}

def get_db_connection():
    for i in range(10):
        try:
            return mysql.connector.connect(**DB_CONFIG)
        except Exception as e:
            print(f"Connexion DB échouée (tentative {i+1}/10): {e}")
            time.sleep(3)
    raise Exception("Impossible de se connecter à la DB")

def sanitize_name(topic):
    name = re.sub(r'[^a-zA-Z0-9_]', '_', topic)
    return re.sub(r'_+', '_', name).strip('_')

def detect_unit(topic, value):
    topic_lower = topic.lower()
    if 'temp' in topic_lower or 'temperature' in topic_lower:
        return "°C"
    elif 'humidity' in topic_lower or 'hum' in topic_lower:
        return "%"
    elif 'pressure' in topic_lower:
        return "hPa"
    elif 'soil' in topic_lower or 'moisture' in topic_lower:
        return "%"
    elif 'uv' in topic_lower:
        return "index"
    elif 'lux' in topic_lower:
        return "lux"
    elif 'dendrometer' in topic_lower or 'diameter' in topic_lower:
        return "mm"
    elif 'battery' in topic_lower:
        return "%"
    elif 'voltage' in topic_lower:
        return "V"
    return ""

def detect_type(topic):
    topic_lower = topic.lower()
    if 'temp' in topic_lower:
        return "temperature"
    elif 'hum' in topic_lower:
        return "humidity"
    elif 'pressure' in topic_lower or 'barometer' in topic_lower:
        return "pressure"
    elif 'soil' in topic_lower or 'moisture' in topic_lower:
        return "soil_moisture"
    elif 'uv' in topic_lower:
        return "uv"
    elif 'lux' in topic_lower or 'luminosity' in topic_lower:
        return "luminosity"
    elif 'dendro' in topic_lower or 'diameter' in topic_lower:
        return "diameter"
    elif 'rain' in topic_lower:
        return "rain"
    elif 'wind' in topic_lower:
        return "wind"
    elif 'battery' in topic_lower:
        return "battery"
    elif 'voltage' in topic_lower:
        return "voltage"
    return "unknown"

def ensure_sensor(cursor, topic, value):
    name = sanitize_name(topic)
    sensor_type = detect_type(topic)
    unit = detect_unit(topic, value)
    
    cursor.execute("SELECT id FROM sensors WHERE name = %s", (name,))
    result = cursor.fetchone()
    
    if not result:
        location = topic.split('/')[1] if '/' in topic else "auto"
        cursor.execute(
            "INSERT INTO sensors (name, type, location, unit) VALUES (%s, %s, %s, %s)",
            (name, sensor_type, location, unit)
        )
        print(f"[NEW SENSOR] {name} ({sensor_type}, {unit}) @ {location}")
    
    cursor.execute("SELECT id FROM sensors WHERE name = %s", (name,))
    return cursor.fetchone()[0]

def insert_measurement(cursor, sensor_id, value):
    cursor.execute(
        "INSERT INTO measurements (sensor_id, value) VALUES (%s, %s)",
        (sensor_id, value)
    )

def decode_p2p_payload(data_b64):
    try:
        decoded = base64.b64decode(data_b64).decode('utf-8')
        return json.loads(decoded)
    except Exception as e:
        print(f"[P2P DECODE ERROR] {e}")
        return None

def handle_p2p_message(cursor, topic, payload):
    try:
        msg = json.loads(payload)
        if "data" not in msg:
            return False
        
        decoded = decode_p2p_payload(msg["data"])
        if not decoded:
            return False
        
        eui = msg.get("eui", "unknown")
        rssi = msg.get("rssi", 0)
        lsnr = msg.get("lsnr", 0)
        
        for key, value in decoded.items():
            sensor_name = f"p2p_{eui}_{key}"
            sensor_id = ensure_sensor(cursor, sensor_name, value)
            insert_measurement(cursor, sensor_id, float(value))
        
        ensure_sensor(cursor, f"p2p_{eui}_rssi", rssi)
        insert_measurement(cursor, ensure_sensor(cursor, f"p2p_{eui}_rssi", rssi), float(rssi))
        
        ensure_sensor(cursor, f"p2p_{eui}_lsnr", lsnr)
        insert_measurement(cursor, ensure_sensor(cursor, f"p2p_{eui}_lsnr", lsnr), float(lsnr))
        
        print(f"[P2P] {eui} -> {decoded}")
        return True
    except Exception as e:
        print(f"[P2P ERROR] {e}")
        return False

def on_message(client, userdata, msg):
    try:
        topic = msg.topic
        payload = msg.payload.decode()
        
        if "lora/p2p/rx" in topic:
            db = get_db_connection()
            cursor = db.cursor()
            if handle_p2p_message(cursor, topic, payload):
                db.commit()
                cursor.close()
                db.close()
                return
            cursor.close()
            db.close()
        
        # Parse JSON ou valeur simple
        try:
            data = json.loads(payload)
            if isinstance(data, dict):
                value = data.get("value") or data.get("temperature") or data.get("humidity") or data.get("pressure")
            else:
                value = data
        except:
            try:
                value = float(payload)
            except:
                print(f"[SKIP] Payload non numérique: {payload}")
                return
        
        if value is None:
            return
            
        db = get_db_connection()
        cursor = db.cursor()
        
        sensor_id = ensure_sensor(cursor, topic, value)
        insert_measurement(cursor, sensor_id, float(value))
        db.commit()
        
        print(f"[DB] {topic} -> {value}")
        
        cursor.close()
        db.close()
    except Exception as e:
        print(f"[ERROR] {e}")

client = mqtt.Client()
if MQTT_USER:
    client.username_pw_set(MQTT_USER, MQTT_PASS)
client.on_message = on_message

print(f"Connecté à {MQTT_BROKER}, subscribe à #")
client.connect(MQTT_BROKER, 1883)
client.subscribe("#")
print("En attente des messages...")
client.loop_forever()
