CREATE DATABASE IF NOT EXISTS agro_monitoring;
USE agro_monitoring;

CREATE TABLE IF NOT EXISTS sensors (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    type VARCHAR(50) NOT NULL,
    location VARCHAR(100),
    unit VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS measurements (
    id INT AUTO_INCREMENT PRIMARY KEY,
    sensor_id INT NOT NULL,
    value FLOAT NOT NULL,
    forecast_value FLOAT NULL,
    corrected_value FLOAT NULL,
    measured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (sensor_id) REFERENCES sensors(id)
);

CREATE TABLE IF NOT EXISTS weather_forecast (
    id INT AUTO_INCREMENT PRIMARY KEY,
    source VARCHAR(50) NOT NULL,
    temperature FLOAT,
    humidity FLOAT,
    pressure FLOAT,
    forecast_for TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO sensors (name, type, location, unit) VALUES
    ('temp_meteo', 'temperature', 'meteo_station', '°C'),
    ('humidity_meteo', 'humidity', 'meteo_station', '%'),
    ('pressure_meteo', 'pressure', 'meteo_station', 'hPa'),
    ('soil_10cm', 'soil_moisture', 'verger_profondeur_10', '%'),
    ('soil_30cm', 'soil_moisture', 'verger_profondeur_30', '%'),
    ('soil_60cm', 'soil_moisture', 'verger_profondeur_60', '%'),
    ('dendrometer', 'diameter', 'verger_tronc', 'mm'),
    ('uv_index', 'uv', 'meteo_station', 'index'),
    ('lux', 'luminosity', 'meteo_station', 'lux')
ON DUPLICATE KEY UPDATE name=name;
