# Autonomous Waste Segregation Boat 

Welcome to the **Autonomous Waste Segregation Boat** project, an innovative solution addressing water pollution by autonomously classifying and collecting waste in water bodies. Designed with deep learning and robotics, this boat navigates independently, identifying waste as biodegradable or non-biodegradable with a high level of accuracy. This work is a significant contribution to environmental tech, and it’s patented as a **Waste Segregation System using Deep Learning**.

---

## 📜 Table of Contents
- [Overview](#overview)
- [Project Highlights](#project-highlights)
- [Technical Details](#technical-details)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Patent & License](#patent--license)

---

## 🌍 Overview
Water pollution is one of today’s most pressing environmental issues. This project aims to create an effective, scalable, and real-time solution using a robotic boat to autonomously segregate waste floating in water bodies, enhancing waste management and contributing to cleaner ecosystems.

## 🌟 Project Highlights
- **Real-time Waste Detection**: The boat employs TensorFlow and a ResNet deep learning model to perform real-time waste classification, distinguishing biodegradable from non-biodegradable waste.
- **High Classification Accuracy**: With an accuracy rate of 93.07%, the system sets a new standard in autonomous waste identification.
- **Autonomous Navigation**: Using Raspberry Pi for processing and navigation, the boat operates independently without human intervention.
- **Patented Innovation**: This project is protected under a published patent, establishing it as a novel approach in environmental waste management technology.

## ⚙️ Technical Details

### Technology Stack
- **Languages**: Python
- **Machine Learning Frameworks**: TensorFlow, ResNet for image classification
- **Computer Vision**: OpenCV
- **Hardware**: Raspberry Pi, integrated camera system for real-time image capture and processing
- **Robotic Components**: Custom hardware setup with motors and sensors for navigation and waste collection

### System Architecture
The project employs:
1. **Deep Learning**: For accurate, real-time waste classification.
2. **Camera System**: Captures continuous video feed for image processing.
3. **Raspberry Pi Control**: Manages navigation and processing for autonomous operation.

## 🚀 Getting Started

### Prerequisites
- Raspberry Pi setup and basic Python environment.
- Hardware components for navigation (refer to hardware setup guide).
- Install necessary libraries with:
   ```bash
   pip install -r requirements.txt
