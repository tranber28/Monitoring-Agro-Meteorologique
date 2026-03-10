#include <Arduino.h>
#include <SPI.h>
#include <Wire.h>
#include <RadioLib.h>
#include <avr/wdt.h>
#include <avr/sleep.h>
#include "I2C_AHT10.h"
#include <ArduinoJson.h>

// Prototypes
void Lora_init();
void do_some_work();
void low_power_set();
void all_pins_low();
void watchdog_init();
void send_lora();
bool AHT_init();

String node_id = String("011261") + "010280";

// Sleep time configuration
#define SLEEP_CYCLE 450

// LoRa configuration
#define FREQUENCY 868.1
#define BANDWIDTH 125.0
#define SPREADING_FACTOR 7
#define CODING_RATE 5
#define OUTPUT_POWER 10
#define PREAMBLE_LEN 8
#define GAIN 0

// Pin definitions (328p)
#define DIO0 2
#define DIO1 6
#define LORA_RST 4
#define LORA_CS 10
#define SPI_MOSI 11
#define SPI_MISO 12
#define SPI_SCK 13
#define VOLTAGE_PIN A3
#define PWM_OUT_PIN 9
#define SENSOR_POWER_PIN 5
#define ADC_PIN A2

#define DEBUG_OUT_ENABLE 1

SX1276 radio = new Module(LORA_CS, DIO0, LORA_RST, DIO1);
AHT10 humiditySensor;

String jsonoutput = "";
bool readSensorStatus = false;
int sensorValue = 0;
int batValue = 0;
int count = 0;
int ADC_O_1;
int ADC_O_2;
int16_t packetnum = 0;
float temperature = 0.0;
float humidity = 0.0;

bool AHT_init()
{
    bool ret = false;
    Wire.begin();
    if (humiditySensor.begin() == false)
    {
#if DEBUG_OUT_ENABLE
        Serial.println("AHT10 not detected. Please check wiring.");
#endif
    }

    if (humiditySensor.available() == true)
    {
        temperature = humiditySensor.getTemperature();
        humidity = humiditySensor.getHumidity();
        ret = true;
    }
    if (isnan(humidity) || isnan(temperature))
    {
#if DEBUG_OUT_ENABLE
        Serial.println(F("Failed to read from AHT sensor!"));
#endif
    }
    return ret;
}

void Lora_init() {
    // Fréquence, BW, SF, CR, SyncWord, Power, Preamble
    int state = radio.begin(868.1, 125.0, 7, 5, 0x34, 17, 8, 0);
    
    if (state == ERR_NONE) {
        radio.setSyncWord(0x34); // Standard LoRaWAN
        radio.invertIQ(true);    // OBLIGATOIRE pour SX1303
        radio.setCRC(true);      // La gateway rejette si pas de CRC
        
        // Optionnel : Si l'Atmega est instable, on augmente la tolérance
        // radio.setFrequencyDeviation(0.5); 
        
#if DEBUG_OUT_ENABLE
        Serial.println(F("LoRa init OK (Mode Gateway)"));
#endif
    }
}
void setup()
{
#if DEBUG_OUT_ENABLE
    Serial.begin(115200);
    Serial.println("Soil sensor P2P start.");
#endif
    delay(100);

    // Set up Timer 1
    pinMode(PWM_OUT_PIN, OUTPUT);
    TCCR1A = bit(COM1A0);
    TCCR1B = bit(WGM12) | bit(CS10);
    OCR1A = 1;

    pinMode(LORA_RST, OUTPUT);
    digitalWrite(LORA_RST, HIGH);
    delay(100);

    pinMode(SENSOR_POWER_PIN, OUTPUT);
    digitalWrite(SENSOR_POWER_PIN, HIGH);
    delay(100);

    Lora_init();

    Wire.begin();
    if (humiditySensor.begin() == false)
    {
#if DEBUG_OUT_ENABLE
        Serial.println("AHT10 not detected.");
#endif
    }
#if DEBUG_OUT_ENABLE
    else
        Serial.println("AHT10 acknowledged.");
#endif

    do_some_work();

#if DEBUG_OUT_ENABLE
    Serial.println("[Set]Sleep Mode Set");
#endif
    low_power_set();
}

void loop()
{
    wdt_disable();

    if (count > SLEEP_CYCLE)
    {
#if DEBUG_OUT_ENABLE
        Serial.println("Code start>>");
#endif

        do_some_work();
        all_pins_low();

#if DEBUG_OUT_ENABLE
        Serial.println("Code end<<");
#endif
        count = 0;
    }

    low_power_set();
}

ISR(WDT_vect)
{
#if DEBUG_OUT_ENABLE
    Serial.print("[Watch dog]");
    Serial.println(count);
#endif
    delay(100);
    count++;
    wdt_disable();
}

void low_power_set()
{
    all_pins_low();
    delay(10);
    ADCSRA = 0;

    sleep_enable();
    watchdog_init();
    set_sleep_mode(SLEEP_MODE_PWR_DOWN);
    delay(10);
    noInterrupts();
    sleep_enable();

    MCUCR = bit(BODS) | bit(BODSE);
    MCUCR = bit(BODS);
    interrupts();

    sleep_cpu();
    sleep_disable();
}

void watchdog_init()
{
    MCUSR = 0;
    WDTCSR = bit(WDCE) | bit(WDE);
    WDTCSR = bit(WDIE) | bit(WDP3) | bit(WDP0);
    wdt_reset();
}

void do_some_work()
{
    digitalWrite(SENSOR_POWER_PIN, HIGH);
    digitalWrite(LORA_RST, HIGH);
    delay(5);
    
    pinMode(PWM_OUT_PIN, OUTPUT);
    TCCR1A = bit(COM1A0);
    TCCR1B = bit(WGM12) | bit(CS10);
    OCR1A = 1;

    Lora_init();
    delay(50);

    // ADC2 AVCC as reference voltage
    ADMUX = _BV(REFS0) | _BV(MUX1);
    ADCSRA = _BV(ADEN) | _BV(ADPS1) | _BV(ADPS0);
    delay(50);
    
    for (int i = 0; i < 3; i++)
    {
        ADCSRA |= (1 << ADSC);
        delay(10);

        if ((ADCSRA & 0x40) == 0)
        {
            ADC_O_1 = ADCL;
            ADC_O_2 = ADCH;
            sensorValue = (ADC_O_2 << 8) + ADC_O_1;
            ADCSRA |= 0x40;
#if DEBUG_OUT_ENABLE
            Serial.print("ADC:");
            Serial.println(sensorValue);
#endif

            if (readSensorStatus == false)
                readSensorStatus = AHT_init();
        }
        ADCSRA |= (1 << ADIF);
        delay(50);
    }

    // ADC3 internal 1.1V as ADC reference voltage
    ADMUX = _BV(REFS1) | _BV(REFS0) | _BV(MUX1) | _BV(MUX0);
    delay(50);
    
    for (int i = 0; i < 3; i++)
    {
        ADCSRA |= (1 << ADSC);
        delay(10);
        
        if ((ADCSRA & 0x40) == 0)
        {
            ADC_O_1 = ADCL;
            ADC_O_2 = ADCH;
            batValue = (ADC_O_2 << 8) + ADC_O_1;
            ADCSRA |= 0x40;
#if DEBUG_OUT_ENABLE
            Serial.print("BAT:");
            Serial.println(batValue);
            float bat = (float)batValue * 3.3 / 1024.0;
            Serial.print(bat);
            Serial.println("V");
#endif
        }
        ADCSRA |= (1 << ADIF);
        delay(50);
    }
    
    send_lora();
    delay(1000);
    radio.sleep();

    packetnum++;
    readSensorStatus = false;
    digitalWrite(SENSOR_POWER_PIN, LOW);
    delay(100);
}

void all_pins_low()
{
    pinMode(PWM_OUT_PIN, INPUT);
    pinMode(A4, INPUT_PULLUP);
    pinMode(A5, INPUT_PULLUP);
    delay(50);
}

void send_lora() {
    // 1. Calcul du niveau de batterie pour le JSON
    float battery_v = (float)batValue * 3.3 / 1024.0;

    // 2. Construction du JSON complet (sans espaces pour la légèreté)
    // Nous incluons : ID, Sol (s), Batterie (b), Température (t), Humidité Air (h), Paquet (p)
    String msg = "{";
    msg += "\"id\":\"" + node_id + "\"";
    msg += ",\"sol\":" + String(sensorValue);
    msg += ",\"bat\":" + String(battery_v, 2);
    msg += ",\"temp\":" + String(temperature, 1);
    msg += ",\"hum\":" + String(humidity, 1);
    msg += ",\"pkt\":" + String(packetnum);
    msg += "}";

    // 3. Configuration Radio (Standard Gateway)
    radio.setSyncWord(0x34);
    radio.invertIQ(true);
    radio.setCRC(true);

#if DEBUG_OUT_ENABLE
    Serial.print(F("Envoi complet vers HA : "));
    Serial.println(msg);
#endif

    // 4. Transmission
    int state = radio.transmit(msg);
    
    if (state == ERR_NONE) {
        Serial.println(F("OK!"));
        packetnum++; // On incrémente après un envoi réussi
    } else {
        Serial.print(F("Erreur Radio : "));
        Serial.println(state);
    }
}