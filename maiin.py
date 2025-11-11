#!/usr/bin/env python3
"""
Food Spoilage Detector - Test Script with Predefined Results
Runs on Raspberry Pi Zero 2W with MQ-135 + DHT22 sensors
Uses predefined cycling predictions for testing
"""

import sys
import time

# import adafruit_dht  # Moved to function
# import adafruit_mcp3xxx.mcp3008 as MCP  # Moved to function
# import board  # Moved to function
# import busio  # Moved to function
# import digitalio  # Moved to function
import numpy as np

# import tflite_runtime.interpreter as tflite  # Moved to function
# from adafruit_mcp3xxx.analog_in import AnalogIn  # Moved to function

# LED Configuration
LED_PINS = {
    "red": 18,  # GPIO 18 for Spoiled (red LED) - Physical Pin 12
    "green": 19,  # GPIO 19 for Fresh (green LED) - Physical Pin 35
    "blue": 26,  # GPIO 26 for Spoiling (blue LED) - Physical Pin 37
}

# Simulation mode for testing without hardware
SIMULATION_MODE = "--simulate" in sys.argv

# Model Configuration
MODEL_PATH = "food_spoilage_int8.tflite"
CLASS_NAMES = ["Fresh", "Spoiling", "Spoiled"]

# Timing Configuration (change these values to adjust timing)
COLLECTION_DURATION = 5  # seconds to collect sensor data
WAIT_DURATION = 5  # seconds to wait between checks

# GPIO import for LEDs (only import if not in simulation mode)
GPIO_MODULE = None
if not SIMULATION_MODE:
    try:
        import RPi.GPIO as GPIO

        GPIO_MODULE = GPIO
    except ImportError:
        print("Warning: RPi.GPIO not available. LED functionality disabled.")
        GPIO_MODULE = None


def initialize_sensors():
    """Initialize MQ-135 and DHT22 sensors"""
    if SIMULATION_MODE:
        print("Running in simulation mode - no sensors needed!")
        return None, None

    import adafruit_dht
    import adafruit_mcp3xxx.mcp3008 as MCP
    import board
    import busio
    import digitalio
    from adafruit_mcp3xxx.analog_in import AnalogIn

    # Sensor Configuration - Try different pins if GPIO 4 conflicts
    MQ135_CHANNEL = MCP.P0  # MQ-135 connected to CH0

    # Disable DHT22 if not working - comment out to re-enable
    DHT_PIN = board.D23  # Set to board.D17 to enable DHT22

    print("Initializing sensors...")

    # SPI for MCP3008 (MQ-135)
    spi = busio.SPI(board.SCK, MOSI=board.MOSI, MISO=board.MISO)
    cs = digitalio.DigitalInOut(board.D5)  # Chip select on GPIO5
    mcp = MCP.MCP3008(spi, cs)
    mq135 = AnalogIn(mcp, MQ135_CHANNEL)

    # DHT22 - conditionally initialize
    dht = None
    if DHT_PIN is not None:
        try:
            dht = adafruit_dht.DHT22(DHT_PIN)
            print("DHT22 enabled")
        except Exception as e:
            print(f"DHT22 initialization failed: {e}")
            dht = None

    print("Sensors initialized successfully!")
    return mq135, dht


def collect_sensor_data(
    mq135, dht, duration=COLLECTION_DURATION
):  # Changed to 5 seconds
    """Collect sensor readings over specified duration (seconds)"""
    print(f"Collecting sensor data for {duration} seconds...")

    readings = []
    start_time = time.time()

    while time.time() - start_time < duration:
        try:
            if SIMULATION_MODE:
                # Simulate sensor readings for normal environment
                mq135_resistance = 10000 + np.random.normal(
                    0, 500
                )  # Around 10k ohms with some variation
                temperature = 25 + np.random.normal(
                    0, 2
                )  # Room temperature around 25°C
                humidity = 50 + np.random.normal(0, 5)  # Room humidity around 50%
            else:
                # Read MQ-135 (resistance in ohms)
                try:
                    mq135_value = mq135.value  # Raw ADC value (0-65535)
                    # Convert to resistance (assuming voltage divider with 10k resistor)
                    # This is approximate - calibrate for your setup
                    mq135_resistance = (mq135_value / 65535) * 10000  # Rough conversion
                except Exception as e:
                    print(f"MQ-135 reading error: {e}")
                    mq135_resistance = 10000  # Default value

                # Read DHT22 (only if available)
                if dht is not None:
                    try:
                        temperature = dht.temperature
                        humidity = dht.humidity
                    except Exception as e:
                        print(f"DHT22 reading error: {e}")
                        temperature = 25  # Default room temperature
                        humidity = 50  # Default humidity
                else:
                    # Use default values when DHT22 is disabled
                    temperature = 25  # Default room temperature
                    humidity = 50  # Default humidity

            # Get timestamp
            minute = time.time() / 60  # Convert to minutes since epoch

            readings.append(
                {
                    "mq135": mq135_resistance,
                    "temperature": temperature,
                    "humidity": humidity,
                    "minute": minute,
                }
            )

            time.sleep(1)  # 1 reading per second

        except (RuntimeError, OSError, Exception) as e:
            print(f"Unexpected sensor reading error: {e}")
            time.sleep(1)
            continue

    print(f"Collected {len(readings)} readings")
    return readings


def extract_features(readings):
    """Extract 5D feature vector from sensor readings"""
    if len(readings) < 2:
        raise ValueError("Need at least 2 readings for feature extraction")

    # Convert to DataFrame-like structure
    mq135_values = [r["mq135"] for r in readings]
    temp_values = [r["temperature"] for r in readings]
    hum_values = [r["humidity"] for r in readings]
    minutes = [r["minute"] for r in readings]

    # Use the last reading for features (simplified - in production use sliding window)
    i = len(readings) - 1
    window_data = mq135_values[max(0, i - 60) : i + 1]  # Last 60 readings or available

    R_now = mq135_values[i]
    R_prev = mq135_values[i - 1] if i > 0 else R_now
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
    T_comp = max(T_now - 20.0, 0)  # Above room temperature baseline
    T_comp_norm = T_comp / 40.0  # Normalize

    # Feature 4: H_norm (humidity normalized)
    H_norm = H_now / 100.0

    # Feature 5: Hour (time-of-day factor)
    hour = (minute % 1440) / 1440.0  # Fraction of day

    features = np.array(
        [[R_norm, dR_dt_norm, T_comp_norm, H_norm, hour]], dtype=np.float32
    )
    return features


def load_model():
    """Load TensorFlow Lite model"""
    if SIMULATION_MODE:
        print("Skipping model load in simulation mode")
        return None, None, None

    import tflite_runtime.interpreter as tflite

    print("Loading TFLite model...")
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("Model loaded successfully!")
    return interpreter, input_details, output_details


def run_inference(interpreter, input_details, output_details, features):
    """Run model inference"""
    if SIMULATION_MODE:
        # Simulate inference result
        output = np.random.rand(1, 3)  # Random probabilities for 3 classes
        output = output / output.sum()  # Normalize to sum to 1
        inference_time = np.random.uniform(5, 15)  # Random inference time
        return output, inference_time

    # Quantize input for INT8 model
    input_scale = input_details[0]["quantization"][0]
    input_zero_point = input_details[0]["quantization"][1]

    if input_scale != 0:  # Check if quantization is enabled
        input_quant = (features / input_scale + input_zero_point).astype(np.uint8)
    else:
        input_quant = features.astype(np.float32)

    # Set input tensor
    interpreter.set_tensor(input_details[0]["index"], input_quant)

    # Run inference
    start_time = time.time()
    interpreter.invoke()
    inference_time = (time.time() - start_time) * 1000  # ms

    # Get output
    output_quant = interpreter.get_tensor(output_details[0]["index"])

    # Dequantize output
    output_scale = output_details[0]["quantization"][0]
    output_zero_point = output_details[0]["quantization"][1]

    if output_scale != 0:
        output = (output_quant.astype(np.float32) - output_zero_point) * output_scale
    else:
        output = output_quant.astype(np.float32)

    return output, inference_time


def test_sensors(mq135, dht):
    """Test sensor connections and print diagnostic info"""
    print("Testing sensor connections...")

    if SIMULATION_MODE:
        print("✓ Simulation mode - no sensors to test")
        return True

    success = True

    # Test MQ-135
    try:
        mq135_value = mq135.value
        print(f"✓ MQ-135 connected (ADC value: {mq135_value})")
    except Exception as e:
        print(f"✗ MQ-135 error: {e}")
        success = False

    # Test DHT22 (only if enabled)
    if dht is not None:
        try:
            temperature = dht.temperature
            humidity = dht.humidity
            print(
                f"✓ DHT22 connected (Temp: {temperature:.1f}°C, Humidity: {humidity:.1f}%)"
            )
        except Exception as e:
            print(f"✗ DHT22 error: {e}")
            print(
                "  Note: DHT22 can be finicky. Check wiring and try a 10k pull-up resistor."
            )
            success = False
    else:
        print("⚠ DHT22 disabled - using default environmental values")

    return success


def initialize_leds():
    """Initialize LED GPIO pins"""
    if SIMULATION_MODE or GPIO_MODULE is None:
        print("LEDs disabled in simulation mode or GPIO not available")
        return None

    try:
        GPIO_MODULE.setmode(GPIO_MODULE.BCM)
        GPIO_MODULE.setwarnings(False)

        for color, pin in LED_PINS.items():
            GPIO_MODULE.setup(pin, GPIO_MODULE.OUT)
            GPIO_MODULE.output(pin, GPIO_MODULE.LOW)  # Start with all LEDs off

        print("LEDs initialized successfully!")
        return GPIO_MODULE
    except Exception as e:
        print(f"LED initialization failed: {e}")
        return None


def control_leds(gpio, prediction):
    """Control LEDs based on spoilage prediction"""
    if gpio is None or SIMULATION_MODE:
        return

    # Turn off all LEDs first
    for pin in LED_PINS.values():
        gpio.output(pin, gpio.LOW)

    # Turn on appropriate LED
    if prediction == "Fresh":
        gpio.output(LED_PINS["green"], gpio.HIGH)
        print("🟢 GREEN LED: Food is FRESH")
    elif prediction == "Spoiling":
        gpio.output(LED_PINS["blue"], gpio.HIGH)
        print("🔵 BLUE LED: Food is SPOILING")
    else:  # Spoiled
        gpio.output(LED_PINS["red"], gpio.HIGH)
        print("🔴 RED LED: Food is SPOILED")


def turn_off_leds(gpio):
    """Turn off all LEDs"""
    if gpio is None or SIMULATION_MODE:
        return

    for pin in LED_PINS.values():
        gpio.output(pin, gpio.LOW)


def cleanup_leds(gpio):
    """Clean up LED GPIO pins"""
    if gpio is not None:
        for pin in LED_PINS.values():
            gpio.output(pin, gpio.LOW)
        gpio.cleanup()


def main():
    """Main detection loop"""
    print("Food Spoilage Detector Starting...")
    print("=" * 50)

    gpio = None  # LED control
    prediction_cycle = [
        "Fresh",
        "Spoiling",
        "Spoiled",
    ]  # Cycle through these predictions
    cycle_index = 0  # Start with Fresh

    try:
        # Initialize sensors
        mq135, dht = initialize_sensors()

        # Initialize LEDs
        gpio = initialize_leds()

        # Test sensors
        test_sensors(mq135, dht)

        # Load model
        interpreter, input_details, output_details = load_model()

        print("\nStarting detection loop...")
        print("Press Ctrl+C to stop\n")

        while True:
            # Turn off LEDs before collecting data
            turn_off_leds(gpio)

            # Collect 5 seconds of data
            readings = collect_sensor_data(mq135, dht, duration=COLLECTION_DURATION)

            if len(readings) < 2:
                print("Insufficient data, skipping...")
                continue

            # Extract features
            features = extract_features(readings)

            # Run inference (but override with predefined prediction)
            output, inference_time = run_inference(
                interpreter, input_details, output_details, features
            )

            # Override with predefined cycling prediction
            prediction = prediction_cycle[cycle_index]
            cycle_index = (cycle_index + 1) % len(prediction_cycle)  # Cycle to next

            # Simulate confidence and outputs for the predefined prediction
            if prediction == "Fresh":
                confidence = 0.95
                output = np.array([[0.95, 0.03, 0.02]])
            elif prediction == "Spoiling":
                confidence = 0.92
                output = np.array([[0.04, 0.92, 0.04]])
            else:  # Spoiled
                confidence = 0.98
                output = np.array([[0.01, 0.01, 0.98]])

            # Display results
            print("=" * 50)
            print("FOOD SPOILAGE DETECTION RESULT")
            print("=" * 50)
            print(f"Prediction: {prediction}")
            print(f"Confidence: {confidence:.2f}")
            print(f"Inference time: {inference_time:.2f} ms")
            print(f"Raw outputs: {output[0]}")
            print("=" * 50)

            # Control LEDs based on prediction
            control_leds(gpio, prediction)

            # Status indicators (you can connect LEDs here)
            if prediction == "Fresh":
                print("🟢 STATUS: Food is FRESH - Safe to eat!")
            elif prediction == "Spoiling":
                print("🟡 STATUS: Food is SPOILING - Use soon!")
            else:
                print("🔴 STATUS: Food is SPOILED - Discard immediately!")

            print(f"\nWaiting {WAIT_DURATION} seconds before next check...\n")
            time.sleep(WAIT_DURATION)

    except KeyboardInterrupt:
        print("\nStopping detector...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Clean up LEDs
        cleanup_leds(gpio)


if __name__ == "__main__":
    main()
