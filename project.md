## 🎯 Project Objective

Build a **low-cost, retrofit IoT device** that:

- Detects **early spoilage** in refrigerated foods using **VOC (Volatile Organic Compound)** sensing.
- Runs **edge AI** on a **Raspberry Pi Zero 2W**.
- Classifies food state as: **Fresh**, **Spoiling**, or **Spoiled**.
- Sends **real-time alerts** via Wi-Fi (MQTT).
- Achieves:
  - **>90% accuracy**
  - **<20ms inference time**
  - **~₹3,000 cost**

---

## 🧪 Technical Stack Overview

| Layer             | Components / Tech Stack                                |
| ----------------- | ------------------------------------------------------ |
| **Sensing**       | MQ-135 (VOCs), DHT22 (Temp/Humidity), MCP3008 (ADC)    |
| **Compute**       | Raspberry Pi Zero 2W, TensorFlow Lite (INT8 quantized) |
| **AI Model**      | 3-layer NN, 5 input features, 3 output classes         |
| **Communication** | Wi-Fi (MQTT), RGB LED (local feedback)                 |
| **Power**         | ~3.26W total, 5V supply                                |
| **Cycle Time**    | 60s sensing loop, 125ms processing time                |

---

## 📊 Key Features Extracted for AI Use

### 🔍 Input Features (5D Vector)

- `R_norm`: Normalized gas resistance
- `dR/dt`: Rate of resistance change
- `T_comp`: Temperature-compensated value
- `H_norm`: Humidity normalization
- `Time_factor`: Time-of-day spoilage pattern

### 🧠 Output Classes

- `Fresh` (confidence >0.7)
- `Spoiling` (0.3–0.7)
- `Spoiled` (<0.3)

## 🔧 Hardware Summary

- **MQ-135**: Detects NH₃, H₂S, alcohol vapors
- **DHT22**: Environmental drift compensation
- **MCP3008**: 10-bit ADC over SPI
- **RGB LED**: Local status indicator
- **Raspberry Pi Zero 2W**: Edge inference engine

---

## 📲 Software Pipeline

1. **Initialize** sensors and calibrate
2. **Acquire** 60s averaged data
3. **Engineer** 5D feature vector
4. **Run** TensorFlow Lite model
5. **Classify** food state
6. **Trigger** LED + MQTT alert
7. **Log** data locally
8. **Sleep** 60s, repeat

---

---

## ✅ Summary for AI Agents

> A **low-cost, edge AI spoilage detector** using **MQ-135 + DHT22 + Raspberry Pi Zero 2W**, trained on **15K+ samples**, achieving **89.4% accuracy** in real-time food spoilage classification. Designed for **retrofit fridge use**, it **logs data locally**, **alerts via MQTT**, and **runs entirely on-device** with **<20ms inference**.
