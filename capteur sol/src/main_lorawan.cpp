#include <Arduino.h>
#include <SPI.h>
#include <Wire.h>
#include <lmic.h>
#include <hal/hal.h>
#include "I2C_AHT10.h"

#define DEBUG_OUT_ENABLE 1

#define LORA_SCK  13
#define LORA_MISO 12
#define LORA_MOSI 11
#define LORA_CS   10
#define LORA_DIO0 2
#define LORA_DIO1 6
#define LORA_RST  4

#define SENSOR_POWER_PIN 5
#define ADC_PIN A2
#define VOLTAGE_PIN A3

static const u1_t PROGMEM AppEUI[8] = { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
static const u1_t PROGMEM DevEUI[8] = { 0x01, 0x12, 0x61, 0x01, 0x02, 0x80, 0x00, 0x01 };
static const u1_t PROGMEM AppKey[16] = { 0x2B, 0x7E, 0x15, 0x16, 0x28, 0xAE, 0xD2, 0xA6, 0xAB, 0xF7, 0x15, 0x88, 0x09, 0xCF, 0x4F, 0x3C };

void os_getArtEui (u1_t* buf) { memcpy_P(buf, AppEUI, 8); }
void os_getDevEui (u1_t* buf) { memcpy_P(buf, DevEUI, 8); }
void os_getDevKey (u1_t* buf) { memcpy_P(buf, AppKey, 16); }

AHT10 humiditySensor;
bool readSensorStatus = false;
int sensorValue = 0;
int batValue = 0;
int16_t packetnum = 0;
float temperature = 0.0;
float humidity = 0.0;

static osjob_t sendjob;
const unsigned TX_INTERVAL = 60;

void do_send(osjob_t* j);

void onEvent (ev_t ev) {
    switch(ev) {
        case EV_JOINED:
            #if DEBUG_OUT_ENABLE
            Serial.println(F("EV_JOINED"));
            #endif
            LMIC_setLinkCheckMode(0);
            do_send(&sendjob);
            break;
        case EV_TXCOMPLETE:
            #if DEBUG_OUT_ENABLE
            Serial.println(F("EV_TXCOMPLETE"));
            #endif
            os_setTimedCallback(&sendjob, os_getTime()+sec2osticks(TX_INTERVAL), do_send);
            break;
        case EV_JOIN_FAILED:
            #if DEBUG_OUT_ENABLE
            Serial.println(F("EV_JOIN_FAILED"));
            #endif
            break;
        default:
            break;
    }
}

void do_send(osjob_t* j) {
    if (LMIC.opmode & OP_TXRXPEND) {
        #if DEBUG_OUT_ENABLE
        Serial.println(F("TX not ready"));
        #endif
        return;
    }

    digitalWrite(SENSOR_POWER_PIN, HIGH);
    delay(50);

    Wire.begin();
    if(humiditySensor.begin() && humiditySensor.available()) {
        temperature = humiditySensor.getTemperature();
        humidity = humiditySensor.getHumidity();
    }

    sensorValue = analogRead(ADC_PIN);
    batValue = analogRead(VOLTAGE_PIN);
    float battery_v = (float)batValue * 3.3 / 1024.0;

    uint8_t payload[8];
    uint16_t soil = sensorValue;
    uint16_t bat = (uint16_t)(battery_v * 100);
    int16_t temp = (int16_t)(temperature * 10);
    uint16_t hum = (uint16_t)(humidity * 10);

    payload[0] = soil >> 8;
    payload[1] = soil & 0xFF;
    payload[2] = bat >> 8;
    payload[3] = bat & 0xFF;
    payload[4] = temp >> 8;
    payload[5] = temp & 0xFF;
    payload[6] = hum >> 8;
    payload[7] = hum & 0xFF;

    #if DEBUG_OUT_ENABLE
    Serial.print(F("Soil:")); Serial.println(soil);
    Serial.print(F("Bat:")); Serial.println(battery_v);
    Serial.print(F("Temp:")); Serial.println(temperature);
    Serial.print(F("Hum:")); Serial.println(humidity);
    Serial.println(F("Sending..."));
    #endif

    LMIC_setTxData2(1, payload, sizeof(payload), 0);

    digitalWrite(SENSOR_POWER_PIN, LOW);
}

void setup() {
    #if DEBUG_OUT_ENABLE
    Serial.begin(115200);
    Serial.println(F("LoRaWAN start"));
    #endif

    pinMode(SENSOR_POWER_PIN, OUTPUT);
    digitalWrite(SENSOR_POWER_PIN, LOW);

    SPI.begin(LORA_SCK, LORA_MISO, LORA_MOSI, LORA_CS);
    
    os_init();
    LMIC_reset();
    
    LMIC_setupChannel(0, 868100000, DR_RANGE_MAP(DR_SF12, DR_SF7),  BAND_CENTI);
    LMIC_setupChannel(1, 868300000, DR_RANGE_MAP(DR_SF12, DR_SF7),  BAND_CENTI);
    LMIC_setupChannel(2, 868500000, DR_RANGE_MAP(DR_SF12, DR_SF7),  BAND_CENTI);
    LMIC_setupChannel(3, 867100000, DR_RANGE_MAP(DR_SF12, DR_SF7),  BAND_CENTI);
    LMIC_setupChannel(4, 867300000, DR_RANGE_MAP(DR_SF12, DR_SF7),  BAND_CENTI);
    LMIC_setupChannel(5, 867500000, DR_RANGE_MAP(DR_SF12, DR_SF7),  BAND_CENTI);
    LMIC_setupChannel(6, 867700000, DR_RANGE_MAP(DR_SF12, DR_SF7),  BAND_CENTI);
    LMIC_setupChannel(7, 867900000, DR_RANGE_MAP(DR_SF12, DR_SF7),  BAND_CENTI);

    LMIC_setDrTxpow(DR_SF7, 14);

    LMIC_startJoining();
}

void loop() {
    os_runloop_once();
}
