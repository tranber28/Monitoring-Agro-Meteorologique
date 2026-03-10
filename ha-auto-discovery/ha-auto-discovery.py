import paho.mqtt.client as mqtt
import json
import os
import time
import base64

MQTT_BROKER = os.environ.get("MQTT_BROKER", "mosquitto")
MQTT_USER   = os.environ.get("MQTT_USER", "")
MQTT_PASS   = os.environ.get("MQTT_PASS", "")
HA_PREFIX   = os.environ.get("HA_PREFIX", "homeassistant")

DISCOVERY_INTERVAL = 60
known_sensors = set()

# ──────────────────────────────────────────────────────────────
#  HELPERS COMMUNS
# ──────────────────────────────────────────────────────────────

def sanitize_name(s):
    return s.replace('/', '_').replace('-', '_').replace(':', '_')

def get_device_class_from_key(key):
    k = key.lower()
    if 'temp'     in k: return "temperature",    "°C",  "measurement"
    if 'hum'      in k: return "humidity",        "%",   "measurement"
    if 'pressure' in k: return "pressure",        "hPa", "measurement"
    if 'sol'      in k: return "moisture",        "%",   "measurement"
    if 'soil'     in k: return "moisture",        "%",   "measurement"
    if 'uv'       in k: return None,              "idx", "measurement"
    if 'lux'      in k: return "illuminance",     "lx",  "measurement"
    if 'bat'      in k: return "voltage",         "V",   "measurement"
    if 'voltage'  in k: return "voltage",         "V",   "measurement"
    if 'rssi'     in k: return "signal_strength", "dBm", "measurement"
    if 'lsnr'     in k: return None,              "dB",  "measurement"
    if 'dist'     in k: return "distance",        "cm",  "measurement"
    if 'level'    in k: return "distance",        "cm",  "measurement"
    if 'pkt'      in k: return None,              "pkts","total_increasing"
    return None, None, None

def publish_sensor_discovery(client, sensor_id, sensor_name, state_topic,
                              value_template, device_class, unit, state_class,
                              device_name, device_identifiers, device_model="MQTT Sensor"):
    if sensor_id in known_sensors:
        return
    known_sensors.add(sensor_id)

    config = {
        "name": sensor_name,
        "state_topic": state_topic,
        "unique_id": f"auto_{sensor_id}",
        "device": {
            "name": device_name,
            "identifiers": device_identifiers,
            "model": device_model
        }
    }
    if value_template:
        config["value_template"] = value_template
    if unit:
        config["unit_of_measurement"] = unit
    if device_class:
        config["device_class"] = device_class
    if state_class:
        config["state_class"] = state_class

    discovery_topic = f"{HA_PREFIX}/sensor/{sensor_id}/config"
    client.publish(discovery_topic, json.dumps(config), retain=True)
    print(f"[HA Discovery] {sensor_id} → {state_topic}")

# ──────────────────────────────────────────────────────────────
#  CHIRPSTACK LORAWAN
#  Topic : application/{app_id}/device/{dev_eui}/event/up
# ──────────────────────────────────────────────────────────────

def handle_chirpstack_uplink(client, topic, payload_str):
    """
    Reçoit un uplink ChirpStack et publie chaque champ de 'object'
    comme un capteur Home Assistant distinct.
    """
    try:
        msg = json.loads(payload_str)
    except Exception as e:
        print(f"[ChirpStack] JSON parse error: {e}")
        return

    device_info  = msg.get("deviceInfo", {})
    dev_eui      = device_info.get("devEui") or msg.get("devEui", "unknown")
    device_name  = device_info.get("deviceName") or msg.get("deviceName", dev_eui)
    app_name     = device_info.get("applicationName") or msg.get("applicationName", "lorawan")
    decoded      = msg.get("object", {})   # payload décodé par ChirpStack

    if not decoded:
        print(f"[ChirpStack] Pas de 'object' dans le message pour {dev_eui} — vérifie ton codec payload")
        return

    for key, value in decoded.items():
        if not isinstance(value, (int, float, str, bool)):
            continue  # ignore les objets imbriqués

        sensor_id    = sanitize_name(f"lorawan_{dev_eui}_{key}")
        sensor_label = f"{device_name} {key}"
        state_topic  = f"ha/lorawan/{dev_eui}/{key}/state"

        device_class, unit, state_class = get_device_class_from_key(key)

        publish_sensor_discovery(
            client,
            sensor_id    = sensor_id,
            sensor_name  = sensor_label,
            state_topic  = state_topic,
            value_template = None,
            device_class = device_class,
            unit         = unit,
            state_class  = state_class,
            device_name  = f"LoRaWAN {device_name}",
            device_identifiers = [f"lorawan_{dev_eui}"],
            device_model = "LoRaWAN OTAA"
        )

        client.publish(state_topic, str(value), retain=True)
        print(f"[ChirpStack] {dev_eui} {key}={value} → {state_topic}")

    # Publie aussi RSSI et SNR si présents dans les rxInfo
    rx_info = msg.get("rxInfo", [])
    if rx_info:
        rssi = rx_info[0].get("rssi")
        snr  = rx_info[0].get("snr")
        if rssi is not None:
            sid = sanitize_name(f"lorawan_{dev_eui}_rssi")
            publish_sensor_discovery(client, sid, f"{device_name} RSSI",
                f"ha/lorawan/{dev_eui}/rssi/state", None,
                "signal_strength", "dBm", "measurement",
                f"LoRaWAN {device_name}", [f"lorawan_{dev_eui}"], "LoRaWAN OTAA")
            client.publish(f"ha/lorawan/{dev_eui}/rssi/state", str(rssi), retain=True)
        if snr is not None:
            sid = sanitize_name(f"lorawan_{dev_eui}_snr")
            publish_sensor_discovery(client, sid, f"{device_name} SNR",
                f"ha/lorawan/{dev_eui}/snr/state", None,
                None, "dB", "measurement",
                f"LoRaWAN {device_name}", [f"lorawan_{dev_eui}"], "LoRaWAN OTAA")
            client.publish(f"ha/lorawan/{dev_eui}/snr/state", str(snr), retain=True)

# ──────────────────────────────────────────────────────────────
#  P2P VIA GATEWAY BRIDGE
#  Topic : eu868/gateway/{gw_id}/event/up
#  Ces paquets sont des JSON P2P encodés en base64 dans phyPayload
#  (paquets non-LoRaWAN qui passent quand même par le gateway bridge)
# ──────────────────────────────────────────────────────────────

def handle_gateway_p2p(client, topic, payload_str):
    """
    Intercepte les paquets P2P qui arrivent sur le topic gateway bridge.
    Le JSON P2P est encodé en base64 dans le champ 'phyPayload'.
    Ex: {"phyPayload":"eyJpZCI6Ij...", "rxInfo": {...}}
    """
    try:
        msg = json.loads(payload_str)
    except Exception as e:
        print(f"[GW-P2P] JSON parse error: {e}")
        return False

    phy = msg.get("phyPayload", "")
    if not phy:
        return False

    # Tente de décoder le phyPayload comme du JSON P2P
    try:
        decoded_bytes = base64.b64decode(phy)
        decoded_str   = decoded_bytes.decode('utf-8')
        # Vérifie que c'est bien du JSON (P2P) et pas du binaire LoRaWAN
        if not decoded_str.startswith('{'):
            return False
        p2p_data = json.loads(decoded_str)
    except Exception:
        return False  # C'est un vrai paquet LoRaWAN binaire, on ignore

    # Extrait l'EUI du capteur P2P depuis le champ 'id' ou 'eui'
    eui = p2p_data.get("id") or p2p_data.get("eui", "unknown")

    # RSSI / SNR depuis rxInfo du gateway
    rx_info = msg.get("rxInfo", {})
    rssi = rx_info.get("rssi")
    snr  = rx_info.get("snr")

    print(f"[GW-P2P] Paquet P2P détecté EUI={eui} : {p2p_data}")

    for key, value in p2p_data.items():
        if key in ("id", "eui") or not isinstance(value, (int, float, str, bool)):
            continue
        converted   = convert_p2p_value(key, value)
        sensor_id   = sanitize_name(f"p2p_{eui}_{key}")
        state_topic = f"ha/p2p/{eui}/{key}/state"
        device_class, unit, state_class = get_device_class_from_key(key)

        publish_sensor_discovery(
            client,
            sensor_id          = sensor_id,
            sensor_name        = f"P2P {eui} {key}",
            state_topic        = state_topic,
            value_template     = None,
            device_class       = device_class,
            unit               = unit,
            state_class        = state_class,
            device_name        = f"LoRa P2P {eui}",
            device_identifiers = [f"p2p_{eui}"],
            device_model       = "LoRa P2P Sensor"
        )
        client.publish(state_topic, str(converted), retain=True)
        print(f"[GW-P2P] {eui} {key}={converted} → {state_topic}")

    # Publie RSSI et SNR
    for meta_key, meta_val in [("rssi", rssi), ("snr", snr)]:
        if meta_val is not None:
            sid = sanitize_name(f"p2p_{eui}_{meta_key}")
            dc, unit, sc = get_device_class_from_key(meta_key)
            publish_sensor_discovery(client, sid, f"P2P {eui} {meta_key}",
                f"ha/p2p/{eui}/{meta_key}/state", None, dc, unit, sc,
                f"LoRa P2P {eui}", [f"p2p_{eui}"], "LoRa P2P Sensor")
            client.publish(f"ha/p2p/{eui}/{meta_key}/state", str(meta_val), retain=True)

    return True

# ──────────────────────────────────────────────────────────────
#  LORA P2P
#  Topic : lora/p2p/rx (ancien chemin, gardé pour compatibilité)
# ──────────────────────────────────────────────────────────────

def decode_p2p_payload(data_b64):
    try:
        decoded = base64.b64decode(data_b64).decode('utf-8')
        return json.loads(decoded)
    except:
        return None

def convert_p2p_value(key, value):
    if 'sol' in key.lower():
        pct = round((1 - value / 1023) * 100, 1)
        return max(0, min(100, pct))
    return value

def handle_p2p_message(client, topic, payload_str):
    try:
        msg = json.loads(payload_str)
    except Exception as e:
        print(f"[P2P] JSON parse error: {e}")
        return False

    if "data" not in msg:
        print(f"[P2P] Pas de champ 'data' dans le message")
        return False

    decoded = decode_p2p_payload(msg["data"])
    if not decoded:
        print(f"[P2P] Impossible de décoder le payload base64")
        return False

    eui = msg.get("eui", "unknown")

    for key, value in decoded.items():
        converted = convert_p2p_value(key, value)
        sensor_id = sanitize_name(f"p2p_{eui}_{key}")
        state_topic = f"ha/p2p/{eui}/{key}/state"
        device_class, unit, state_class = get_device_class_from_key(key)

        publish_sensor_discovery(
            client,
            sensor_id    = sensor_id,
            sensor_name  = f"P2P {eui} {key}",
            state_topic  = state_topic,
            value_template = None,
            device_class = device_class,
            unit         = unit,
            state_class  = state_class,
            device_name  = f"LoRa P2P {eui}",
            device_identifiers = [f"p2p_{eui}"],
            device_model = "LoRa P2P Sensor"
        )
        client.publish(state_topic, str(converted), retain=True)
        print(f"[P2P] {eui} {key}={converted} → {state_topic}")

    # RSSI et SNR radio
    for meta_key in ["rssi", "lsnr"]:
        val = msg.get(meta_key)
        if val is not None:
            sid = sanitize_name(f"p2p_{eui}_{meta_key}")
            dc, unit, sc = get_device_class_from_key(meta_key)
            publish_sensor_discovery(client, sid, f"P2P {eui} {meta_key}",
                f"ha/p2p/{eui}/{meta_key}/state", None, dc, unit, sc,
                f"LoRa P2P {eui}", [f"p2p_{eui}"], "LoRa P2P Sensor")
            client.publish(f"ha/p2p/{eui}/{meta_key}/state", str(val), retain=True)

    print(f"[P2P] {eui} publié dans HA avec succès")
    return True

# ──────────────────────────────────────────────────────────────
#  HANDLER MQTT GÉNÉRIQUE (autres topics)
# ──────────────────────────────────────────────────────────────

def get_device_class(topic):
    return get_device_class_from_key(topic)

def publish_discovery(client, topic, value):
    sensor_name = sanitize_name(topic)
    if sensor_name in known_sensors:
        return
    known_sensors.add(sensor_name)

    device_class, unit, state_class = get_device_class(topic)

    config = {
        "name": sensor_name,
        "state_topic": topic,
        "unique_id": f"auto_{sensor_name}",
        "device": {
            "name": "Auto Discovery",
            "identifiers": ["auto_discovery"],
            "model": "MQTT Auto"
        }
    }
    if unit:         config["unit_of_measurement"] = unit
    if device_class: config["device_class"] = device_class
    if state_class:  config["state_class"] = "measurement"

    client.publish(f"{HA_PREFIX}/sensor/{sensor_name}/config", json.dumps(config), retain=True)
    print(f"[HA Generic] Discovery: {sensor_name}")

# ──────────────────────────────────────────────────────────────
#  ON MESSAGE — routage principal
# ──────────────────────────────────────────────────────────────

def on_message(client, userdata, msg):
    try:
        # Ignore les topics internes HA
        if msg.topic.startswith(HA_PREFIX) or msg.topic.startswith("ha/"):
            return

        # Ignore les binaires (protobuf, etc.)
        try:
            raw = msg.payload.decode('utf-8')
        except UnicodeDecodeError:
            return

        # ── ChirpStack uplink ──
        # Topic : application/{app_id}/device/{dev_eui}/event/up
        if "/event/up" in msg.topic and msg.topic.startswith("application/"):
            handle_chirpstack_uplink(client, msg.topic, raw)
            return

        # ── P2P via gateway bridge ──
        # Topic : eu868/gateway/{gw_id}/event/up
        if "/event/up" in msg.topic and msg.topic.startswith("eu868/gateway/"):
            handle_gateway_p2p(client, msg.topic, raw)
            return


        # ── LoRa P2P ──
        if "lora/p2p/rx" in msg.topic:
            handle_p2p_message(client, msg.topic, raw)
            return

        # ── Topics génériques JSON ──
        try:
            data = json.loads(raw)
            value = data.get("value") if isinstance(data, dict) else data
        except:
            value = raw

        if value is not None:
            publish_discovery(client, msg.topic, value)

    except Exception as e:
        print(f"[ERROR] {e}")

# ──────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────

client = mqtt.Client()
if MQTT_USER:
    client.username_pw_set(MQTT_USER, MQTT_PASS)
client.on_message = on_message
client.connect(MQTT_BROKER, 1883)
client.subscribe("#")
print(f"[HA Auto-Discovery] Connecté à {MQTT_BROKER}, écoute sur tous les topics (#)")
print(f"[HA Auto-Discovery] Support : ChirpStack LoRaWAN + LoRa P2P + MQTT générique")
client.loop_forever()
