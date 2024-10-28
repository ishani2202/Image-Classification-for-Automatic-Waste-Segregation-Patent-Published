🚤 Autonomous Waste Segregation Boat 🌊
Welcome to the Autonomous Waste Segregation Boat project, an innovative solution addressing water pollution by autonomously classifying and collecting waste in water bodies. Designed with deep learning and robotics, this boat navigates independently, identifying waste as biodegradable or non-biodegradable with a high level of accuracy. This work is a significant contribution to environmental tech, and it’s patented as a Waste Segregation System using Deep Learning.

📜 Table of Contents
Overview
Project Highlights
Technical Details
Getting Started
Usage
Patent & License
🌍 Overview
Water pollution is one of today’s most pressing environmental issues. This project aims to create an effective, scalable, and real-time solution using a robotic boat to autonomously segregate waste floating in water bodies, enhancing waste management and contributing to cleaner ecosystems.

🌟 Project Highlights
Real-time Waste Detection: The boat employs TensorFlow and a ResNet deep learning model to perform real-time waste classification, distinguishing biodegradable from non-biodegradable waste.
High Classification Accuracy: With an accuracy rate of 93.07%, the system sets a new standard in autonomous waste identification.
Autonomous Navigation: Using Raspberry Pi for processing and navigation, the boat operates independently without human intervention.
Patented Innovation: This project is protected under a published patent, establishing it as a novel approach in environmental waste management technology.
⚙️ Technical Details
Technology Stack
Languages: Python
Machine Learning Frameworks: TensorFlow, ResNet for image classification
Computer Vision: OpenCV
Hardware: Raspberry Pi, integrated camera system for real-time image capture and processing
Robotic Components: Custom hardware setup with motors and sensors for navigation and waste collection
System Architecture
The project employs:

Deep Learning: For accurate, real-time waste classification.
Camera System: Captures continuous video feed for image processing.
Raspberry Pi Control: Manages navigation and processing for autonomous operation.
🚀 Getting Started
Prerequisites
Raspberry Pi setup and basic Python environment.
Hardware components for navigation (refer to hardware setup guide).
Install necessary libraries with:
bash
Copy code
pip install -r requirements.txt
Installation
Clone the repository to your local machine:

bash
Copy code
git clone https://github.com/yourusername/autonomous-waste-segregation-boat.git
📌 Usage
Power up the boat system and run the main script:

bash
Copy code
python main.py
The system will capture video, classify waste, and display output in real-time. Data is logged for further analysis.

For detailed configurations, refer to the config.yaml file.

📑 Patent & License
Patent: This project has a published patent for its unique waste segregation methodology using deep learning, underscoring its originality and technical innovation.

Usage License: While the project code is available for educational and non-commercial research purposes, any commercial application requires prior authorization due to the patent protection. Please contact us for commercial inquiries.


