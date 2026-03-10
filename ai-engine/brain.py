import paho.mqtt.client as mqtt
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib
import os
import time
from datetime import datetime

MQTT_BROKER = os.environ.get("MQTT_BROKER", "mosquitto")
MQTT_USER = os.environ.get("MQTT_USER", "")
MQTT_PASS = os.environ.get("MQTT_PASS", "")
MODEL_PATH = "/data/models/"
MIN_SAMPLES = int(os.environ.get("MIN_SAMPLES", "50"))
USE_DB = os.environ.get("USE_DB", "false").lower() == "true"

DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "mariadb-agro"),
    "user": os.environ.get("DB_USER", "XXXX"),
    "password": os.environ.get("DB_PASSWORD", "XXXX"),
    "database": os.environ.get("DB_NAME", "agro_monitoring")
}

os.makedirs(MODEL_PATH, exist_ok=True)

def get_db_connection():
    try:
        import mysql.connector
        return mysql.connector.connect(**DB_CONFIG)
    except:
        return None

def get_or_create_sensor(db, sensor_name, sensor_type="unknown", unit=""):
    try:
        cursor = db.cursor()
        cursor.execute("SELECT id FROM sensors WHERE name = %s", (sensor_name,))
        result = cursor.fetchone()
        if result:
            return result[0]
        cursor.execute(
            "INSERT INTO sensors (name, type, location, unit) VALUES (%s, %s, %s, %s)",
            (sensor_name, sensor_type, "ia_engine", unit)
        )
        db.commit()
        cursor.execute("SELECT id FROM sensors WHERE name = %s", (sensor_name,))
        result = cursor.fetchone()
        print(f"[IA] Capteur créé: {sensor_name}")
        return result[0] if result else 1
    except Exception as e:
        print(f"[IA] Erreur: {e}")
        return 1

def extract_value(payload):
    if isinstance(payload, dict):
        return payload.get("value") or payload.get("temperature") or payload.get("humidity")
    try:
        return float(payload)
    except:
        return None

class SensorModel:
    def __init__(self, name, model_type="temperature"):
        self.name = name
        self.model_type = model_type
        self.model_file = f"{MODEL_PATH}{name}.pkl"
        self.model = self.load_model()
        self.history = []
        
    def load_model(self):
        if os.path.exists(self.model_file):
            return joblib.load(self.model_file)
        return LinearRegression()
    
    def save_model(self):
        joblib.dump(self.model, self.model_file)
    
    def predict(self, forecast, timestamp=None):
        if timestamp is None:
            timestamp = time.time()
        dt = datetime.fromtimestamp(timestamp)
        
        if len(self.history) >= MIN_SAMPLES:
            df = pd.DataFrame(self.history)
            X = df[['forecast', 'hour', 'month']].values
            y = df['real'].values
            self.model.fit(X, y)
            self.save_model()
            pred = self.model.predict([[forecast, dt.hour, dt.month]])[0]
            return round(pred, 2)
        return forecast
    
    def add_data(self, forecast, real):
        dt = datetime.now()
        self.history.append({
            'forecast': forecast,
            'real': real,
            'hour': dt.hour,
            'month': dt.month,
            'timestamp': time.time()
        })
        # Garder 200 dernières données
        if len(self.history) > 200:
            self.history = self.history[-200:]

class AgriBrain:
    def __init__(self):
        self.db = get_db_connection() if USE_DB else None
        self.models = {}
        self.data_buffer = {}
        
    def process(self, topic, value, client):
        # Extraire le nom du sensor depuis le topic
        # topic: sensor/meteo/temperature -> meteo_temperature
        # topic: sensor/soil/10cm -> soil_10cm
        parts = topic.split('/')
        if len(parts) >= 3:
            sensor_name = f"{parts[1]}_{parts[2]}"
        else:
            sensor_name = topic.replace('/', '_')
        
        # Stocker la donnée
        if sensor_name not in self.data_buffer:
            self.data_buffer[sensor_name] = {}
        
        if 'real' in topic or 'measured' in topic or 'actual' in topic:
            self.data_buffer[sensor_name]['real'] = value
        elif 'forecast' in topic or 'openmeteo' in topic or 'predicted' in topic:
            self.data_buffer[sensor_name]['forecast'] = value
        
        # Si on a les 2, on entraîne et prédit
        buf = self.data_buffer[sensor_name]
        if 'real' in buf and 'forecast' in buf:
            if sensor_name not in self.models:
                self.models[sensor_name] = SensorModel(sensor_name)
            
            model = self.models[sensor_name]
            model.add_data(buf['forecast'], buf['real'])
            corrected = model.predict(buf['forecast'])
            
            # Publier le résultat
            result_topic = f"ia/{sensor_name}/corrected"
            client.publish(result_topic, json.dumps({
                "value": corrected,
                "original": buf['forecast'],
                "real": buf['real']
            }))
            print(f"[IA] {sensor_name}: forecast={buf['forecast']} real={buf['real']} corrected={corrected}")
            
            buf.clear()

brain = AgriBrain()

def on_message(client, userdata, msg):
    try:
        payload_str = msg.payload.decode()
        value = extract_value(payload_str)
        
        if value is None:
            return
        
        brain.process(msg.topic, value, client)
    except Exception as e:
        print(f"[ERROR] {e}")

client = mqtt.Client()
if MQTT_USER:
    client.username_pw_set(MQTT_USER, MQTT_PASS)
client.on_message = on_message
client.connect(MQTT_BROKER, 1883)
client.subscribe("#")
print(f"[IA] Connecté à {MQTT_BROKER}, mode auto-discovery (min {MIN_SAMPLES} samples)")
client.loop_forever()
