#!/usr/bin/env python3
"""
Food Spoilage Detector - Raspberry Pi Inference Script
Runs on Raspberry Pi Zero 2W with MQ-135 + DHT22 sensors
Uses TensorFlow Lite INT8 quantized model for edge AI
"""

import time
import numpy as np
import board
import busio
import digitalio
import adafruit_mcp3xxx.mcp3008 as MCP
from adafruit_mcp3xxx.analog_in import AnalogIn
import adafruit_dht
import tflite_runtime.interpreter as tflite

# Sensor Configuration
MQ135_CHANNEL = MCP.P0  # MQ-135 connected to CH0
DHT_PIN = board.D4      # DHT22 connected to GPIO4

# Model Configuration
MODEL_PATH = 'food_spoilage_int8.tflite'
CLASS_NAMES = ['Fresh', 'Spoiling', 'Spoiled']

def initialize_sensors():
    """Initialize MQ-135 and DHT22 sensors"""
    print("Initializing sensors...")

    # SPI for MCP3008 (MQ-135)
    spi = busio.SPI(board.SCK, MOSI=board.MOSI, MISO=board.MISO)
    cs = digitalio.DigitalInOut(board.D5)  # Chip select on GPIO5
    mcp = MCP.MCP3008(spi, cs)
    mq135 = AnalogIn(mcp, MQ135_CHANNEL)

    # DHT22
    dht = adafruit_dht.DHT22(DHT_PIN)

    print("Sensors initialized successfully!")
    return mq135, dht

def collect_sensor_data(mq135, dht, duration=60):
    """Collect sensor readings over specified duration (seconds)"""
    print(f"Collecting sensor data for {duration} seconds...")

    readings = []
    start_time = time.time()

    while time.time() - start_time < duration:
        try:
            # Read MQ-135 (resistance in ohms)
            mq135_value = mq135.value  # Raw ADC value (0-65535)
            # Convert to resistance (assuming voltage divider with 10k resistor)
            # This is approximate - calibrate for your setup
            mq135_resistance = (mq135_value / 65535) * 10000  # Rough conversion

            # Read DHT22
            temperature = dht.temperature
            humidity = dht.humidity

            # Get timestamp
            minute = time.time() / 60  # Convert to minutes since epoch

            readings.append({
                'mq135': mq135_resistance,
                'temperature': temperature,
                'humidity': humidity,
                'minute': minute
            })

            time.sleep(1)  # 1 reading per second

        except RuntimeError as e:
            print(f"Sensor reading error: {e}")
            time.sleep(1)
            continue

    print(f"Collected {len(readings)} readings")
    return readings

def extract_features(readings):
    """Extract 5D feature vector from sensor readings"""
    if len(readings) < 2:
        raise ValueError("Need at least 2 readings for feature extraction")

    # Convert to DataFrame-like structure
    mq135_values = [r['mq135'] for r in readings]
    temp_values = [r['temperature'] for r in readings]
    hum_values = [r['humidity'] for r in readings]
    minutes = [r['minute'] for r in readings]

    # Use the last reading for features (simplified - in production use sliding window)
    i = len(readings) - 1
    window_data = mq135_values[max(0, i-60):i+1]  # Last 60 readings or available

    R_now = mq135_values[i]
    R_prev = mq135_values[i-1] if i > 0 else R_now
    T_now = temp_values[i]
    H_now = hum_values[i]
    minute = minutes[i]

    # Feature 1: R_norm (normalized resistance)
    R_min = min(window_data)
    R_max = max(window_data)
    R_norm = (R_now - R_min) / (R_max - R_min + 1e-3)

    # Feature 2: dR/dt (rate of resistance change)
    dR_dt = (R_now - R_prev) / 60.0  # Change per minute
    dR_dt_norm = dR_dt / 1.0  # Scale to reasonable range

    # Feature 3: T_comp (temperature compensation)
    T_comp = max(T_now - 4.0, 0)  # Above fridge baseline
    T_comp_norm = T_comp / 40.0  # Normalize

    # Feature 4: H_norm (humidity normalized)
    H_norm = H_now / 100.0

    # Feature 5: Hour (time-of-day factor)
    hour = (minute % 1440) / 1440.0  # Fraction of day

    features = np.array([[R_norm, dR_dt_norm, T_comp_norm, H_norm, hour]], dtype=np.float32)
    return features

def load_model():
    """Load TensorFlow Lite model"""
    print("Loading TFLite model...")
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("Model loaded successfully!")
    return interpreter, input_details, output_details

def run_inference(interpreter, input_details, output_details, features):
    """Run model inference"""
    # Quantize input for INT8 model
    input_scale = input_details[0]['quantization'][0]
    input_zero_point = input_details[0]['quantization'][1]

    if input_scale != 0:  # Check if quantization is enabled
        input_quant = (features / input_scale + input_zero_point).astype(np.uint8)
    else:
        input_quant = features.astype(np.float32)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_quant)

    # Run inference
    start_time = time.time()
    interpreter.invoke()
    inference_time = (time.time() - start_time) * 1000  # ms

    # Get output
    output_quant = interpreter.get_tensor(output_details[0]['index'])

    # Dequantize output
    output_scale = output_details[0]['quantization'][0]
    output_zero_point = output_details[0]['quantization'][1]

    if output_scale != 0:
        output = (output_quant.astype(np.float32) - output_zero_point) * output_scale
    else:
        output = output_quant.astype(np.float32)

    return output, inference_time

def main():
    """Main detection loop"""
    print("Food Spoilage Detector Starting...")
    print("=" * 50)

    try:
        # Initialize sensors
        mq135, dht = initialize_sensors()

        # Load model
        interpreter, input_details, output_details = load_model()

        print("\nStarting detection loop...")
        print("Press Ctrl+C to stop\n")

        while True:
            # Collect 60 seconds of data
            readings = collect_sensor_data(mq135, dht, duration=60)

            if len(readings) < 2:
                print("Insufficient data, skipping...")
                continue

            # Extract features
            features = extract_features(readings)

            # Run inference
            output, inference_time = run_inference(interpreter, input_details, output_details, features)

            # Get prediction
            predicted_class = np.argmax(output[0])
            confidence = np.max(output[0])
            prediction = CLASS_NAMES[predicted_class]

            # Display results
            print("=" * 50)
            print("FOOD SPOILAGE DETECTION RESULT")
            print("=" * 50)
            print(f"Prediction: {prediction}")
            print(".2f")
            print(".2f")
            print(f"Raw outputs: {output[0]}")
            print("=" * 50)

            # Status indicators (you can connect LEDs here)
            if prediction == 'Fresh':
                print("🟢 STATUS: Food is FRESH - Safe to eat!")
            elif prediction == 'Spoiling':
                print("🟡 STATUS: Food is SPOILING - Use soon!")
            else:
                print("🔴 STATUS: Food is SPOILED - Discard immediately!")

            print("\nWaiting 60 seconds before next check...\n")
            time.sleep(60)

    except KeyboardInterrupt:
        print("\nStopping detector...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()