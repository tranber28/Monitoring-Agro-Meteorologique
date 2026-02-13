import paho.mqtt.client as mqtt
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib
import os
import time

# --- CONFIGURATION ---
MQTT_BROKER = "127.0.0.1" # Change par l'IP de ton Pi si besoin
TOPIC_REAL = "sensor/meteo/temp"        # Ta sonde réelle
TOPIC_FORECAST = "api/openmeteo/temp"   # La prévision HA
TOPIC_CORRECTED = "ia/meteo/temp_corrigee"
MODEL_PATH = "models/temp_corrector.pkl"
DATA_PATH = "models/history.csv"

# --- LOGIQUE IA ---
class AgriBrain:
    def __init__(self):
        self.model = self.load_model()
        self.history = self.load_history()

    def load_model(self):
        if os.path.exists(MODEL_PATH):
            return joblib.load(MODEL_PATH)
        return LinearRegression()

    def load_history(self):
        if os.path.exists(DATA_PATH):
            return pd.read_csv(DATA_PATH).to_dict('records')
        return []

    def train_and_predict(self, forecast, real):
        # Ajouter au jeu de données
        self.history.append({'forecast': forecast, 'real': real, 'timestamp': time.time()})
        df = pd.DataFrame(self.history)
        
        # Sauvegarder l'historique en CSV pour ton GitHub (version data)
        df.to_csv(DATA_PATH, index=False)

        # Si on a assez de points (ex: 50 mesures), on entraîne
        if len(df) > 50:
            X = df[['forecast']].values
            y = df['real'].values
            self.model.fit(X, y)
            joblib.dump(self.model, MODEL_PATH) # On sauve l'IA
            
            prediction = self.model.predict([[forecast]])[0]
            return round(prediction, 2)
        return forecast # Retourne la prévision brute si pas assez de données

brain = AgriBrain()
current_data = {}

# --- MQTT ---
def on_message(client, userdata, msg):
    try:
        val = float(msg.payload.decode())
        if msg.topic == TOPIC_REAL:
            current_data['real'] = val
        elif msg.topic == TOPIC_FORECAST:
            current_data['forecast'] = val

        # Quand on a les deux, l'IA travaille
        if 'real' in current_data and 'forecast' in current_data:
            corrected = brain.train_and_predict(current_data['forecast'], current_data['real'])
            client.publish(TOPIC_CORRECTED, corrected)
            print(f"Stats: Réel={current_data['real']} | Prévu={current_data['forecast']} | IA={corrected}")
            current_data.clear() # Reset pour le prochain cycle
    except Exception as e:
        print(f"Erreur: {e}")

client = mqtt.Client()
client.on_message = on_message
client.connect(MQTT_BROKER, 1883)
client.subscribe([(TOPIC_REAL, 0), (TOPIC_FORECAST, 0)])
client.loop_forever()